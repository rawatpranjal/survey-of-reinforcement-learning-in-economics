## Policy Mirror Descent for Reinforcement Learning: Linear Convergence, New Sampling Complexity, and Generalized Problem Classes

## Guanghui Lan

Submitted: Feb 5, 2021; Revised: Oct 26, 2021; Accepted: April 5, 2022.

Abstract We present new policy mirror descent (PMD) methods for solving reinforcement learning (RL) problems with either strongly convex or general convex regularizers. By exploring the structural properties of these overall highly nonconvex problems we show that the PMD methods exhibit fast linear rate of convergence to the global optimality. We develop stochastic counterparts of these methods, and establish an O (1 //epsilon1 ) (resp., O (1 //epsilon1 2 )) sampling complexity for solving these RL problems with strongly (resp., general) convex regularizers using different sampling schemes, where /epsilon1 denote the target accuracy. We further show that the complexity for computing the gradients of these regularizers, if necessary, can be bounded by O{ (log γ /epsilon1 )[(1 -γ ) L/µ ] 1 / 2 log(1 //epsilon1 ) } (resp., O{ (log γ /epsilon1 )( L//epsilon1 ) 1 / 2 } ) for problems with strongly (resp., general) convex regularizers. Here γ denotes the discounting factor. To the best of our knowledge, these complexity bounds, along with our algorithmic developments, appear to be new in both optimization and RL literature. The introduction of these convex regularizers also greatly enhances the flexibility and thus expands the applicability of RL models.

## 1 Introduction

In this paper, we study a general class of reinforcement learning (RL) problems involving either covex or strongly convex regularizers in their cost functions. Consider the finite Markov decision process M = ( S , A , P , c, γ ), where S is a finite state space, A is a finite action space, P : S × S × A → R is transition model, c : S × A → R is the cost function, and γ ∈ (0 , 1) is the discount factor. A policy π : A×S → R determines the probability of selecting a particular action at a given state.

For a given policy π , we measure its performance by the action-value function ( Q -function) Q π : S × A → R defined as

<!-- formula-not-decoded -->

Here h π is a closed convex function w.r.t. the policy π , i.e., there exist some µ ≥ 0 s.t.

<!-- formula-not-decoded -->

where 〈· , ·〉 denotes the inner product over the action space A , ( h ′ ) π ′ ( s, · ) denotes a subgradient of h ( s ) at π ′ , and D π π ′ ( s ) is the Bregman's distance or Kullback-Leibler (KL) divergence between π and π ′ (see Subsection 1.1 for more discussion).

Clearly, if h π = 0, then Q π becomes the classic action-value function. If h π ( s ) = µD π π 0 ( s ) for some µ &gt; 0, then Q π reduces to the so-called entropy regularized action-value function. The incorporation of a more general

This research was partially supported by the NSF grants 1909298 and 1953199 and NIFA grant 2020-67021-31526. The paper was first released at https://arxiv.org/abs/2102.00135 on 01/30/2021.

H. Milton Stewart School of Industrial and Systems Engineering, Georgia Institute of Technology, Atlanta, GA, 30332. (email: george.lan@isye.gatech.edu ).

Address(es) of author(s) should be given

convex regularizer h π allows us to not only unify these two cases, but also to greatly enhance the expression power and thus the applicability of RL. For example, by using either the indicator function, quadratic penalty or barrier functions, h π can model the set of constraints that an optimal policy should satisfy. It can describe the correlation among different actions for different states. h π can also model some risk or utility function associated with the policy π . Throughout this paper, we say that h π is a strongly convex regularizer if µ &gt; 0. Otherwise, we call h π a general convex regularizer. Clearly the latter class of problems covers the regular case with h π = 0.

We define the state-value function V π : S → R associated with π as

<!-- formula-not-decoded -->

It can be easily seen from the definitions of Q π and V π that

<!-- formula-not-decoded -->

The main objective in RL is to find an optimal policy π ∗ : S × A → R s.t.

<!-- formula-not-decoded -->

for any s ∈ S . Here ∆ |A| denotes the simplex constraint given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By examining Bellman's optimality condition for dynamic programming ([3] and Chapter 6 of [19]), we can show the existence of a policy π ∗ which satisfies (1.6) simultaneously for all s ∈ S . Hence, we can formulate (1.6) as an optimization problem with a single objective by taking the weighted sum of V π over s (with weights ρ s &gt; 0 and ∑ s ∈ S ρ s = 1):

While the weights ρ can be arbitrarily chosen, a reasonable selection of ρ would be the stationary state distribution induced by the optimal policy π ∗ , denoted by ν ∗ ≡ ν ( π ∗ ). As such, problem (1.8) reduces to

<!-- formula-not-decoded -->

It has been observed recently (eg., [14]) that one can simplify the analysis of various algorithms by setting ρ to ν ∗ . As we will also see later, even though the definition of the objective f in (1.9) depends on ν ∗ and hence the unknown optimal policy π ∗ , the algorithms for solving (1.6) and (1.9) do not really require the input of π ∗ .

Recently, there has been considerable interest in the development of first-order methods for solving RL problems in (1.8) -(1.9). While these methods have been derived under various names (e.g., policy gradient, natural policy gradient, trust region policy optimization), they all utilize the gradient information of f (i.e., Q function) in some form to guide the search of optimal policy (e.g., [21,9,7,1,20,5,23,15]). As pointed out by a few authors recently, many of these algorithms are intrinsically connected to the classic mirror descent method originally presented by Nemirovski and Yudin [17,2,16], and some analysis techniques in mirror descent method have thus been adapted to reinforcement learning [20,23,22]. In spite of the popularity of these methods in practice, a few significant issues remain on their theoretical studies. Firstly, most policy gradient methods converge only sublinearly, while many other classic algorithms (e.g., policy iteration) can converge at a linear rate due to the contraction properties of the Bellman operator. Recently, there are some interesting works relating first-order methods with the Bellman operator to establish their linear convergence [4,5]. However, in a nutshell these developments rely on the contraction of the Bellman operator, and as a consequence, they either require unrealistic algorithmic assumptions (e.g., exact line search [4]) or apply only for some restricted problem classes (e.g., entropy regularized problems [5]). Secondly, the convergence of stochastic policy gradient methods has not been well-understood in spite of intensive research effort. Due to unavoidable bias, stochastic policy gradient methods exhibit much slower rate of convergence than related methods, e.g., stochastic Q-learning.

Our contributions in this paper mainly exist in the following several aspects. Firstly, we present a policy mirror descent (PMD) method and show that it can achieve a linear rate of convergence for solving RL problems with strongly convex regularizers. We then develop a more general form of PMD, namely approximate policy

mirror descent (APMD) method, obtained by applying an adaptive perturbation term into PMD, and show that it can achieve a linear rate of convergence for solving RL problems with general convex regularizers. Even though the overall problem is highly nonconvex, we exploit the generalized monotonicity [6,13,11] associated with the variational inequality (VI) reformulation of (1.8)-(1.9) (see [8] for a comprehensive introduction to VI). As a consequence, our convergence analysis does not rely on the contraction properties of the Bellman operator. This fact not only enables us to define h π as a general (strongly) convex function of π and thus expand the problem classes considered in RL, but also facilitates the study of PMD methods under the stochastic settings.

Secondly, we develop the stochastic policy mirror descent (SPMD) and stochastic approximate policy mirror descent (SAPMD) method to handle stochastic first-order information. One key idea of SPMD and SAPMD is to handle separately the bias and expected error of the stochastic estimation of the action-value functions in our convergence analysis, since we can usually reduce the bias term much faster than the total expected error. We establish general convergence results for both SPMD and SAPMD applied to solve RL problems with strongly convex and general convex regularizers, under different conditions about the bias and expected error associated with the estimation of value functions.

Thirdly, we establish the overall sampling complexity of these algorithms by employing different schemes to estimate the action-value function. More specifically, we present an O ( |S||A| /µ/epsilon1 ) and O ( |S||A| //epsilon1 2 ) sampling complexity for solving RL problems with strongly convex and general convex regularizers, when one has access to multiple independent sampling trajectories. To the best of our knowledge, the former sampling complexity is new in the RL literature, while the latter one has not been reported before for policy gradient type methods. We further enhance a recently developed conditional temporal difference (CTD) method [12] so that it can reduce the bias term faster. We show that with CTD, the aforementioned O (1 /µ/epsilon1 ) and O (1 //epsilon1 2 ) sampling complexity bounds can be achieved in the single trajectory setting with Markovian noise under certain regularity assumptions.

Fourthly, observe that unless h π is relatively simple (e.g., h π does not exist or it is given as the KL divergence), the subproblems in the SPMD and SAPMD methods do not have an explicit solution in general and require an efficient solution procedure to find some approximate solutions. We establish the general conditions on the accuracy for solving these subproblems, so that the aforementioned linear rate of convergence and new sampling complexity bounds can still be maintained. We further show that if h π is a smooth convex function, by employing an accelerated gradient descent method for solving these subproblems, the overall gradient computations for h π can be bounded by O{ (log γ /epsilon1 ) √ (1 -γ ) L/µ log(1 //epsilon1 ) } and O{ (log γ /epsilon1 ) √ L//epsilon1 } , respectively, for the case when h π is a strongly convex and general convex function. To the best of our knowledge, such gradient complexity has not been considered before in the RL and optimization literature.

This paper is organized as follows. In Section 2, we discuss the optimality conditions and generalized monotonicity about RL with convex regularizers. Sections 3 and 4 are dedicated to the deterministic and stochastic policy mirror descent methods, respectively. In Section 5 we establish the sampling complexity bounds under different sampling schemes, while the gradient complexity of computing ∇ h π is shown in Section 6. Some concluding remarks are made in Section 7.

## 1.1 Notation and terminology

For any two points π ( ·| s ) , π ′ ( ·| s ) ∈ ∆ |A| , we measure their Kullback-Leibler (KL) divergence by

<!-- formula-not-decoded -->

Observe that the KL divergence can be viewed as is a special instance of the Bregman's distance (or proxfunction) widely used in the optimization literature. Let the distance generating function ω ( π ( ·| s )) := ∑ a ∈A π ( a | s ) log π ( a | s ) 1 . The Bregman's distance associated with ω is given by

<!-- formula-not-decoded -->

1 It is worth noting that we do not enforce π ( a | s ) &gt; 0 when defining ω ( π ( ·| s )) as all the search points generated by our algorithms will satisfy this assumption.

where the last equation follows from the fact that ∑ a ∈A ( π ( a | s ) -π ′ ( a | s )) = 0. Therefore, we will use the KL divergence KL( π ( ·| s ) ‖ π ′ ( ·| s )) and Bregman's distance D π π ′ ( s ) interchangeably throughout this paper. It should be noted that our algorithmic framework allows us to use other distance generating functions, such as ‖ · ‖ 2 p for some p &gt; 1, which, different from the KL divergence, has a bounded prox-function over ∆ |A| .

## 2 Optimality Conditions and Generalized Monotonicity

It is well-known that the value function V π ( s ) in (1.3) is highly nonconvex w.r.t. π , because the components of π ( ·| s ) are multiplied by each other in their definitions (see also Lemma 3 of [1] for an instructive counterexample). However, we will show in this subsection that problem (1.9) can be formulated as a variational inequality (VI) which satisfies certain generalized monotonicity properties (see [6], Section 3.8.2 of [13] and [11]).

Let us first compute the gradient of the value function V π ( s ) in (1.3). For simplicity, we assume for now that h π is differentiable and will relax this assumption later. For a given policy π , we define the discounted state visitation distribution by

<!-- formula-not-decoded -->

where Pr π ( s t = s | s 0 ) denotes the state visitation probability of s t = s after we follow the policy π starting at state s 0 . Let P π denote the transition probability matrix associated with policy π , i.e., P π ( i, j ) = ∑ a ∈A π ( a | i ) P ( j | i, a ), and e i be the i -th unit vector. Then Pr π ( s t = s | s 0 ) = e T s 0 ( P π ) t e s and

<!-- formula-not-decoded -->

Lemma 1 For any ( s 0 , s, a ) ∈ S × S × A , we have

<!-- formula-not-decoded -->

where ∇ h π ( s, · ) denotes the gradient of h π ( s ) w.r.t. π .

Proof . It follows from (1.4) that

<!-- formula-not-decoded -->

Also the relation in (1.5) implies that

<!-- formula-not-decoded -->

Combining the above two relations, we obtain

<!-- formula-not-decoded -->

/negationslash where the second equality follows by expanding ∂V π ( s ′ ) ∂π ( a | s ) recursively, and the third equality follows from the definition of d π s 0 ( s ) in (2.1), and the last identity follows from ∂π ( a ′ | x ) ∂π ( a | s ) = 0 for x = s or a ′ = a , and ∂h π ( x ) ∂π ( a | s ) = 0 for x = s .

/negationslash

/negationslash

In view of Lemma 1, the gradient of the objective f ( π ) in (1.9) at the optimal policy π ∗ is given by

<!-- formula-not-decoded -->

where the third identity follows from (2.2) and the last one follows from the fact that ( ν ∗ ) T ( P π ∗ ) t = ( ν ∗ ) T for any t ≥ 0 since ν ∗ is the steady state distribution of π ∗ . Therefore, the optimality condition of (1.9) suggests us to solve the following variational inequality

<!-- formula-not-decoded -->

However, the above VI requires h π to be differentiable. In order to handle the possible non-smoothness of h π , we instead solve the following problem

<!-- formula-not-decoded -->

It turns out this variational inequality satisfies certain generalized monotonicity properties thanks to the following performance difference lemma obtained by generalizing some previous results (e.g., Lemma 6.1 of [9]).

Lemma 2 For any two feasible policies π and π ′ , we have where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof . For simplicity, let us denote ξ π ′ ( s 0 ) the random process ( s t , a t , s t +1 ), t ≥ 0, generated by following the policy π ′ starting with the initial state s 0 . It then follows from the definition of V π ′ that

<!-- formula-not-decoded -->

where (a) follows by taking the term V π ( s 0 ) outside the summation, (b) follows from the fact that E ξ π ′ ( s ) [ V π ( s 0 )] = V π ( s ) since the random process starts with s 0 = s , and (c) follows from (1.5). The previous conclusion, together with (2.6) and the definition d π ′ s in (2.1), then imply that

<!-- formula-not-decoded -->

which immediately implies the result.

We are now ready to prove the generalized monotonicity for the variational inequality in (2.5).

Lemma 3 The VI problem in (2.5) satisfies

<!-- formula-not-decoded -->

Proof . It follows from Lemma 2 (with π ′ = π ∗ ) that

<!-- formula-not-decoded -->

Let e denote the vector of all 1's. Then, we have

<!-- formula-not-decoded -->

where the first identity follows from the definition of A π ( s ′ , · ) in (2.6), the second equality follows from the fact that 〈 e, π ∗ ( ·| s ′ ) 〉 = 1, and the third equality follows from the definition of V π in (1.3). Combining the above two relations and taking expectation w.r.t. ν ∗ , we obtain

<!-- formula-not-decoded -->

where the second identity follows similarly to (2.3) since ν ∗ is the steady state distribution induced by π ∗ . The result then follows by rearranging the terms.

Since V π ( s ) -V π ∗ ( s ) ≥ 0 for any feasible policity π , we conclude from Lemma 3 that

<!-- formula-not-decoded -->

Therefore, the VI in (2.5) satisfies the generalized monotonicity. In the next few sections, we will exploit the generalized monotonicity and some other structural properties to design efficient algorithms for solving the RL problem.

## 3 Deterministic Policy Mirror Descent

In this section, we present the basic schemes of policy mirror descent (PMD) and establish their convergence properties.

## 3.1 Prox-mapping

In the proposed PMD methods, we will update a given policy π to π + through the following proximal mapping:

<!-- formula-not-decoded -->

Here η &gt; 0 denotes a certain stepsize (or learning rate), and G π can be the operator for the VI formulation, e.g., G π ( s, · ) = Q π ( s, · ) or its approximation.

It is well-known that one can solve (3.1) explicitly for some interesting special cases, e.g., when h p ( s ) = 0 or h p ( s ) = τD p π 0 ( s ) for some τ &gt; 0 and given π 0 . For both these cases, the solution of (3.1) boils down to solving a problem of the form

<!-- formula-not-decoded -->

for some g ∈ R |A| . It can be easily checked from the Karush-Kuhn-Tucker conditions that its optimal solution is given by p ∗ i = exp( -g i ) / [ ∑ |A| i =1 exp( -g i )] . (3.2) For more general convex functions h p , problem (3.1) usually does not have an explicit solution, and one can only solve it approximately. In fact, we will show in Section 6 that by applying the accelerated gradient descent method, we only need to compute a small number of updates in the form of (3.2) in order to approximately solve (3.1) without slowing down the efficiency of the overall PMD algorithms.

## 3.2 Basic PMD method

As shown in Algorithm 1, each iteration of the PMD method applies the prox-mapping step discussed in Subsection 3.1 to update the policy π k . It involves the stepsize parameter η k and requires the selection of an initial point π 0 . For the sake of simplicity, we will assume throughout the paper that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In this case, we have

Observe also that we can replace Q π k ( s, · ) in (3.5) with A π k ( s, a ) defined in (2.6) without impacting the updating of π k +1 ( s, · ), since this only introduces an extra constant into the objective function of (3.5).

## Algorithm 1 The policy mirror descent (PMD) method

Input: initial points π 0 and stepsizes η k ≥ 0.

for k = 0 , 1 , . . . , do end for

Below we establish some general convergence properties about the PMD method. Different from the classic policy iteration or value iteration method used in Markov Decision Processes, our analysis does not rely on the contraction properties of the Bellman's operator, but on the so-called three-point lemma associated with the optimality condition of problem (3.5) (see Lemma 4). Our analysis also significantly differs from the one for the classic mirror descent method in convex optimization (see, e.g., Chapter 3 of [13]). First, the classic mirror descent method requires the convexity of the objective function, while the analysis of PMD utilizes the generalized monotonicity in Lemma 3. Second, the classic mirror descent utilizes the Lipschitz or smoothness properties of the objective function, while in the PMD method, we show the progress made in each iteration of this algorithm (see Lemma 5) by using the performance difference lemma (c.f., Lemma 2) and the three-point lemma (c.f., Lemma 4). As a result, we make no assumptions about the smoothness properties of the objective function at all.

The following result characterizes the optimality condition of problem (3.5) (see Lemma 3.5 of [13]). We add a proof for the sake of completeness.

<!-- formula-not-decoded -->

Lemma 4 For any p ( ·| s ) ∈ ∆ |A| , we have

<!-- formula-not-decoded -->

Proof . By the optimality condition of (3.5),

<!-- formula-not-decoded -->

where ( h ′ ) π k +1 denotes the subgradient of h at π k +1 and ∇ D π k +1 π k ( s, · ) denotes the gradient of D π k +1 π k ( s ) at π k +1 . Using the definition of Bregman's distance, it is easy to verify that

<!-- formula-not-decoded -->

The result then immediately follows by combining the above two relations together with (1.2).

Lemma 5 For any s ∈ S , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof . It follows from Lemma 2 (with π ′ = π k +1 , π = π k and τ = τ k ) that

<!-- formula-not-decoded -->

Similarly to (2.8), we can show that

<!-- formula-not-decoded -->

Combining the above two identities, we then obtain

<!-- formula-not-decoded -->

Now we conclude from Lemma 4 applied to (3.5) with p ( ·| s ′ ) = π k ( ·| s ′ ) that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The previous two conclusions then clearly imply the result in (3.7). It also follows from (3.11) that

<!-- formula-not-decoded -->

where the last inequality follows from the fact that d π k +1 s ( s ) ≥ (1 -γ ) due to the definition of d π k +1 s in (2.1). The result in (3.7) then follows immediately from (3.10) and the above inequality.

Now we show that with a constant stepsize rule, the PMD method can achieve a linear rate of convergence for solving RL problems with strongly convex regularizers (i.e., µ &gt; 0).

Theorem 1 Suppose that η k = η for any k ≥ 0 in the PMD method with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then we have for any k ≥ 0 , where

Proof . By Lemma 4 applied to (3.5) (with η k = η and p = π ∗ ), we have

<!-- formula-not-decoded -->

which, in view of (3.8), then implies that

<!-- formula-not-decoded -->

Taking expectation w.r.t. ν ∗ on both sides of the above inequality and using Lemma 3, we arrive at

<!-- formula-not-decoded -->

Noting V π k +1 ( s ) -V π k ( s ) = V π k +1 ( s ) -V π ∗ ( s ) -[ V π k ( s ) -V π ∗ ( s )] and rearranging the terms in the above inequality, we have

<!-- formula-not-decoded -->

which, in view of the assumption (3.13) and the definition of f in (1.9)

<!-- formula-not-decoded -->

Applying this relation recursively and using the bound in (3.4) we then conclude the result.

According to Theorem 1, the PMD method converges linearly in terms of both function value and the distance to the optimal solution for solving RL problems with strongly convex regularizers. Now we show that a direct application of the PMD method only achieves a sublinear rate of convergence for the case when µ = 0.

Theorem 2 Suppose that η k = η in the PMD method. Then we have

<!-- formula-not-decoded -->

for any k ≥ 0 .

Proof . It follows from (3.15) with µ = 0 that

<!-- formula-not-decoded -->

Taking the telescopic sum of the above inequalities and using the fact that V π k +1 ( s ) ≤ V π k ( s ) due to (3.7) , we obtain

<!-- formula-not-decoded -->

which clearly implies the result in view of the definition of f in (1.9) and the bound on D π ∗ π 0 in (3.4).

The result in Theorem 2 shows that the PMD method requires O (1 / (1 -γ ) /epsilon1 ) iterations to find an /epsilon1 -solution for general RL problems. This bound already matches, in terms of its dependence on (1 -γ ) and /epsilon1 , the previously best-known complexity for natural policy gradient methods [1]. We will further enhance the PMD method so that it can achieve a linear rate of convergence for the case when µ = 0 in next subsection.

3.3 Approximate policy mirror descent method

In this subsection, we propose to enhance the basic PMD method by adding adaptively a perturbation term into the definition of the value functions or the proximal-mapping.

For some τ ≥ 0 and a given initial policy π 0 ( a | s ) &gt; 0, ∀ s ∈ S , a ∈ A , we define the perturbed action-value and state-value functions, respectively, by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Clearly, if τ = 0, then the perturbed value functions reduce to the usual value functions, i.e.,

<!-- formula-not-decoded -->

The following result relates the value functions with different τ .

Lemma 6 For any given τ, τ ′ ≥ 0 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof . By the definitions of V π τ and d π s , we have

<!-- formula-not-decoded -->

which together with the bound on D π π 0 in (3.4) then imply (3.19).

As shown in Algorithm 2, the approximate policy mirror descent (APMD) method is obtained by replacing Q π k ( s, · ) with its approximation Q π k τ k ( s, · ) and adding the perturbation τ k D π π 0 ( s t ) for the updating of π k +1 in the basic PMD method. As discussed in Subsection 3.1, the incorporation of the perturbation term does not impact the difficulty of solving the subproblem in (3.20). In fact, the APMD method can be viewed as a general form of the PMD method since it reduces to the PMD method when τ k = 0. In fact, the perturbation parameter τ k used to define the action-value function Q π k τ k ( s, · ) is not necessarily the same as the one used in the regularization term τ k D p π 0 ( s t ), yielding more flexibility to the design and analysis for this class of algorithms.

## Algorithm 2 The approximate policy mirror descent (APMD) method

Input: initial points π 0 , stepsizes η k ≥ 0 and perturbation τ k ≥ 0. for k = 0 , 1 , . . . , do

<!-- formula-not-decoded -->

end for

Our goal in the remaining part of this subsection is to show that the APMD method, when employed with proper selection of τ k , can achieve a linear rate of convergence for solving general RL problems. Note that in the classic mirror descent method, adding a perturbation term into the objective function usually would not improve its rate of convergence from sublinear to linear. However, the linear rate of convergence in PMD depends on the

As a consequence, if τ ≥ τ ′ ≥ 0 then

discount factor rather than the strongly convex modulus of the regularization term, which makes it possible for us to show a linear rate of convergence for the APMD method.

First we observe that Lemma 3 can still be applied to the perturbed value functions. The difference between the following result and Lemma 3 exists in that the RHS of (3.21) is no longer nonnegative, i.e., V π τ ( s ) -V π ∗ τ ( s ) /notgreaterorslnteql 0. However, this relation will be approximately satisfied if τ is small enough.

Lemma 7 The VI problem in (2.5) satisfies

<!-- formula-not-decoded -->

Proof . The proof is the same as that for Lemma 3 except that we will apply the performance difference lemma (i.e., Lemma 2) to the perturbed value function V π τ .

Next we establish some general convergence properties about the APMD method. Lemma 8 below characterizes the optimal solution of (3.20) (see, e.g., Lemma 3.5 of [13]).

Lemma 8 Let π k +1 ( ·| s ) be defined in (3.20). For any p ( ·| s ) ∈ ∆ |A| , we have

<!-- formula-not-decoded -->

Lemma 9 below is similar to Lemma 5 for the PMD method.

Lemma 9 For any s ∈ S , we have

<!-- formula-not-decoded -->

Proof . By applying Lemma 2 to the perturbed value function V π τ and using an argument similar to (3.10), we can show that

<!-- formula-not-decoded -->

Now we conclude from Lemma 8 with p ( ·| s ′ ) = π k ( ·| s ′ ) that

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

] where the last inequality follows from the fact that d π k +1 s ( s ) ≥ (1 -γ ) due to the definition of d π k +1 s in (2.1). The result in (3.22) then follows immediately from (3.23) and the above inequality.

The following general result holds for different stepsize rules for APMD.

Lemma 10 Suppose 1 + η k τ k = 1 /γ and τ k ≥ τ k +1 in the APMD method. Then for any k ≥ 0 , we have

<!-- formula-not-decoded -->

Proof . By Lemma 8 with p = π ∗ , we have

<!-- formula-not-decoded -->

Moreover, by Lemma 9,

<!-- formula-not-decoded -->

Combining the above two relations, we obtain

<!-- formula-not-decoded -->

Taking expectation w.r.t. ν ∗ on both sides of the above inequality and using Lemma 7, we arrive at

<!-- formula-not-decoded -->

Noting V π k +1 τ k ( s ) -V π k τ k ( s ) = V π k +1 τ k ( s ) -V π ∗ τ k ( s ) -[ V π k τ k ( s ) -V π ∗ τ k ( s )] and rearranging the terms in the above inequality, we have

<!-- formula-not-decoded -->

Using the above inequality, the assumption τ k ≥ τ k +1 and (3.19), we have

<!-- formula-not-decoded -->

which implies the result by the assumption 1 + η k τ k = 1 /γ .

We are now ready to establish the rate of convergence of the APMD method with dynamic stepsize rules to select η k and τ k for solving general RL problems.

Theorem 3 Suppose that τ k = τ 0 γ k for some τ 0 ≥ 0 and that 1 + η k τ k = 1 /γ for any k ≥ 0 in the APMD method. Then for any k ≥ 0 , we have

<!-- formula-not-decoded -->

Proof . Applying the result in Lemma 10 recursively, we have

<!-- formula-not-decoded -->

Noting that V π k τ k ( s ) ≥ V π k ( s ), V π ∗ τ k ( s ) ≤ V π ∗ ( s ) + τ k 1 -γ log |A| , and V π ∗ τ 0 ( s ) ≥ V π ∗ ( s ) due to (3.18), and that V π 0 τ 0 ( s ) = V π 0 ( s ) due to D π 0 π 0 ( s ) = 0, we conclude from the previous inequality that

<!-- formula-not-decoded -->

The result in (3.29) immediately follows from the above relation, the definition of f in (1.9), and the selection of τ k .

According to (3.29), if τ 0 is a constant, then the rate of convergence of the APMD method is O ( kγ k ). If the total number of iterations k is given a priori, we can improve the rate of convergence to O ( γ k ) by setting τ 0 = 1 /k . Below we propose a different way to specify τ k for the APMD method so that it can achieve this O ( γ k ) rate of convergence without fixing k a priori.

We first establish a technical result that will also be used later for the analysis of stochastic PMD methods.

Lemma 11 Assume that the nonnegative sequences { X k } k ≥ 0 , { Y k } k ≥ 0 and { Z k } k ≥ 0 satisfy

<!-- formula-not-decoded -->

Let us denote l = ⌈ log γ 1 4 ⌉ . If Y k = Y · 2 -( /floorleft k/l /floorright +1) and Z k = Z · 2 -( /floorleft k/l /floorright +2) for some Y ≥ 0 and Z ≥ 0 , then

<!-- formula-not-decoded -->

Proof . Let us group the indices { 0 , . . . , k } into ¯ p ≡ /floorleft k/l /floorright +1 epochs with each of the first ¯ p -1 epochs consisting of l iterations. Let p = 0 , . . . , ¯ p be the epoch indices. We first show that for any p = 0 , . . . , ¯ p -1,

<!-- formula-not-decoded -->

This relation holds obviously for p = 0. Let us assume that (3.33) holds at the beginning of epoch p ad examine the progress made in epoch p . Note that for any indices k = pl, . . . , ( p +1) l -1 in epoch p , we have Y k = Y · 2 -( p +1) and Z = Z · 2 -( p +2) . By applying (3.31) recursively, we have

<!-- formula-not-decoded -->

where the second inequality follows from the definition of Z pl and γ l ≥ 0, the third one follows from γ l ≤ 1 / 4, the fourth one follows by induction hypothesis, and the last one follows by regrouping the terms. Since k = (¯ p -1) l + k (mod l ), we have

<!-- formula-not-decoded -->

which implies the result.

We are now ready to present a more convenient selection of τ k and η k for the APMD method.

Theorem 4 Let us denote l := ⌈ log γ 1 4 ⌉ . If τ k = 2 -( /floorleft k/l /floorright +1) and 1 + η k τ k = 1 /γ , then

<!-- formula-not-decoded -->

Proof . By using Lemma 10 and Lemma 11 (with X k = E s ∼ ν ∗ [ V π k τ k ( s ) -V π ∗ τ k ( s ) + τ k 1 -γ D π ∗ π k ( s )] and Y k = τ k 1 -γ log |A| ), we have

<!-- formula-not-decoded -->

Noting that V π k τ k ( s ) ≥ V π k ( s ), V π ∗ τ k ( s ) ≤ V π ∗ ( s ) + τ k 1 -γ log |A| , V π ∗ τ 0 ( s ) ≥ V π ∗ ( s ) due to (3.18), and that V π 0 τ 0 ( s ) = V π 0 ( s ) due to D π 0 π 0 ( s ) = 0, we conclude from the previous inequality and the definition of τ k that

<!-- formula-not-decoded -->

In view of Theorem 4, a policy ¯ π s.t. f (¯ π ) -f ( π ∗ ) ≤ /epsilon1 will be found in at most O (log(1 //epsilon1 )) epochs and hence at most O ( l log(1 //epsilon1 )) = O (log γ ( /epsilon1 )) iterations, which matches the one for solving RL problems with strongly convex regularizers. However, for general RL problems, we cannot guarantee the linear convergence of D π ∗ π k +1 ( s ) since its coefficient τ k will become very small eventually. By using the continuity of the objective function and the compactness of the feasible set, we can possibly show that the solution sequence converges to the true optimal policy asymptotically as the number of iterations increases. On the other hand, the rate of convergence associated with the solution sequence of the PMD method for general RL problems cannot be established unless more structural properties of the RL problems can be further explored.

## 4 Stochastic Policy Mirror Descent

The policy mirror descent methods described in the previous section require the input of the exact action-value functions Q π k . This requirement can hardly be satisfied in practice even for the case when P is given explicitly, since Q π k is defined as an infinite sum. In addition, in RL one does not know the transition dynamics P and thus only stochastic estimators of action-value functions are available. In this section, we propose stochastic versions for the PMD and APMD methods to address these issues.

## 4.1 Basic stochastic policy mirror descent

In this subsection, we assume that for a given policy π k , there exists a stochastic estimator Q π k ,ξ k s.t.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some σ k ≥ and ς k ≥ 0, where ξ k denotes the random vector used to generate the stochastic estimator Q π k ,ξ k . Clearly, if σ k = 0, then we have exact information about Q π k . One key insight we have for the stochastic PMD methods is to handle separately the bias term ς k from the overall expected error term σ k , because one can reduce the bias term much faster than the total error. This makes the analysis of the stochastic PMD method considerably different from that of the classic stochastic mirror descent method. While in this section we focus on the convergence analysis of the algorithms, we will show in next section that such separate treatment of bias and total error enables us to substantially improve the sampling complexity for solving RL problems by using policy gradient type methods.

The stochastic policy mirror descent (SPMD) is obtained by replacing Q π k in (3.5) with its stochastic estimator Q π k ,ξ k , i.e.,

<!-- formula-not-decoded -->

In the sequel, we denote ξ /ceilingleft k /ceilingright the sequence of random vectors ξ 0 , . . . , ξ k and define

<!-- formula-not-decoded -->

By using the assumptions in (4.1) and (4.3) and the decomposition

<!-- formula-not-decoded -->

we can see that

<!-- formula-not-decoded -->

Similar to Lemma 5, below we show some general convergence properties about the SPMD method. Unlike PMD, SPMD does not guarantee the non-increasing property of V π k ( s ) anymore.

Lemma 12 For any s ∈ S , we have

<!-- formula-not-decoded -->

Proof . Observe that (3.10) still holds, and hence that

<!-- formula-not-decoded -->

where the first inequality follows from Young's inequality and the second one follows from the strong convexity of D π k π k +1 w.r.t. to ‖·‖ 1 . Moreover, we conclude from Lemma 4 applied to (4.4) with Q π k replaced by Q π k ,ξ k and p ( ·| s ′ ) = π k ( ·| s ′ ) that

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

where the last inequality follows from the fact that d π k +1 s ( s ) ≥ 1 -γ due to the definition of d π k +1 s in (2.1). The result in (4.7) then follows immediately from (4.8) and the above inequality.

We now establish an important recursion about the SPMD method.

Lemma 13 For any k ≥ 0 , we have

<!-- formula-not-decoded -->

Proof . By applying Lemma 4 to (3.5) (with Q π k replaced by Q π k ,ξ k and p = π ∗ ), we have

<!-- formula-not-decoded -->

which, in view of (4.7), then implies that

<!-- formula-not-decoded -->

Taking expectation w.r.t. ξ /ceilingleft k /ceilingright and ν ∗ on both sides of the above inequality, and using Lemma 3 and the relation in (4.6), we arrive at

<!-- formula-not-decoded -->

Noting V π k +1 ( s ) -V π k ( s ) = V π k +1 ( s ) -V π ∗ ( s ) -[ V π k ( s ) -V π ∗ ( s )], rearranging the terms in the above inequality, and using the definition of f in (1.9), we arrive at the result.

We are now ready to establish the convergence rate of the SPMD method. We start with the case when µ &gt; 0 and state a constant stepsize rule which requires both ς k and σ k , k ≥ 0, to be small enough to guarantee the convergence of the SPMD method.

Theorem 5 Suppose that η k = η = 1 -γ γµ in the SPMD method. If ς k = 2 -( /floorleft k/l /floorright +2) and σ 2 k = 2 -( /floorleft k/l /floorright +2) for any k ≥ 0 with l := ⌈ log γ (1 / 4) ⌉ , then

Proof . By Lemma 13 and the selection of η , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which, in view of Lemma 11 with X k = E ξ /ceilingleft k -1 /ceilingright [ f ( π k ) -f ( π ∗ ) + µ 1 -γ D ( π k , π ∗ ) and Z k = 2 ς k + σ 2 k 2 γµ , then implies that

<!-- formula-not-decoded -->

We now turn our attention to the convergence properties of the SPMD method for the case when µ = 0.

Theorem 6 Suppose that η k = η for any k ≥ 0 in the SPMD method. If ς k ≤ ς and σ k ≤ σ for any k ≥ 0 , then we have

<!-- formula-not-decoded -->

where R denotes a random number uniformly distributed between 1 and k . In particular, if the number of iterations k is given a priori and η = ( 2(1 -γ ) log |A| kσ 2 ) 1 / 2 , then

<!-- formula-not-decoded -->

Proof . By Lemma 13 and the fact that µ = 0, we have

<!-- formula-not-decoded -->

Taking the telescopic sum of the above relations, we have

<!-- formula-not-decoded -->

We add some remarks about the results in Theorem 6. In comparison with the convergence results of SPMD for the case µ &gt; 0, there exist some possible shortcomings for the case when µ = 0. Firstly, one needs to output a randomly selected π R from the trajectory. Secondly, since the first term in (4.11) converges sublinearly, one has to update π k +1 at least O (1 //epsilon1 ) times, which may also impact the gradient complexity of computing ∇ h π if π k +1 cannot be computed explicitly. We will address these issues by developing the stochastic APMD method in next subsection.

Dividing both sides by (1 -γ ) k and using the definition of R , we obtain the result in (4.10).

## 4.2 Stochastic approximate policy mirror descent

The stochastic approximate policy mirror descent (SAPMD) method is obtained by replacing Q π k τ k in (3.20) with its stochastic estimator Q π k ,ξ k τ k . As such, its updating formula is given by

<!-- formula-not-decoded -->

With a little abuse of notation, we still denote δ k := Q π k ,ξ k τ k -Q π k τ k and assume that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some σ k ≥ and ς k ≥ 0. Similarly to (4.6) we have

<!-- formula-not-decoded -->

Lemma 14 and Lemma 15 below show the improvement for each SAPMD iteration.

Lemma 14 For any k ≥ 0 , we have

<!-- formula-not-decoded -->

Proof . The proof is similar to the one for Lemma 12 except that we will apply Lemma 2 to the perturbed value functions V π τ k instead of V π .

Lemma 15 If 1 + η k τ k = 1 /γ and τ k ≥ τ k +1 in the SAPMD method, then for any k ≥ 0 ,

<!-- formula-not-decoded -->

Proof . By Lemma 8 with p = π ∗ and Q π k τ k replaced by Q π k ,ξ k τ k , we have

<!-- formula-not-decoded -->

which, in view of (4.17), implies that

<!-- formula-not-decoded -->

Taking expectation w.r.t. ξ /ceilingleft k /ceilingright and ν ∗ on both sides of the above inequality, and using Lemma 7 and the relation in (4.16), we arrive at

<!-- formula-not-decoded -->

Noting V π k +1 τ k ( s ) -V π k τ k ( s ) = V π k +1 τ k ( s ) -V π ∗ τ k ( s ) -[ V π k τ k ( s ) -V π ∗ τ k ( s )] and rearranging the terms in the above inequality, we have

<!-- formula-not-decoded -->

which, in view of the assumption τ k ≥ τ k +1 and (3.19), then implies that

<!-- formula-not-decoded -->

The result then immediately follows from the assumption that 1 + η k τ k = 1 /γ .

We are now ready to establish the convergence of the SAPMD method.

Theorem 7 Suppose that η k = 1 -γ γτ k in the SAPMD method. If τ k = 1 √ γ log |A| 2 -( /floorleft k/l /floorright +1) , ς k = 2 -( /floorleft k/l /floorright +2) , and σ 2 k = 4 -( /floorleft k/l /floorright +2) with l := ⌈ log γ (1 / 4) ⌉ , then

<!-- formula-not-decoded -->

Proof . By Lemma 15 and the selection of τ k , ς k and σ k , we have

<!-- formula-not-decoded -->

Using the above inequality and Lemma 11 (with X k = E s ∼ ν ∗ ,ξ /ceilingleft k -1 /ceilingright [ γ [ V π k τ k ( s ) -V π ∗ τ k ( s ) + τ k 1 -γ D π ∗ π k ( s )]], Y k = τ k 1 -γ log |A| and Z k = (2 + √ log |A| 2 √ γ )2 -( /floorleft k/l /floorright +2) ), we conclude

<!-- formula-not-decoded -->

Noting that V π k τ k ( s ) ≥ V π k ( s ), V π ∗ τ k ( s ) ≤ V π ∗ ( s ) + τ k 1 -γ log |A| , V π ∗ τ 0 ( s ) ≥ V π ∗ ( s ) due to (3.18), and that V π 0 τ 0 ( s ) = V π 0 ( s ) due to D π 0 π 0 ( s ) = 0, we conclude from the previous inequality and the definition of τ k that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

from which the result immediately follows.

A few remarks about the convergence of the SAPMD method are in place.

First, in view of Theorem 7, the SAPMD method does not need to randomly output a solution as most existing nonconvex stochastic gradient descent methods did. Instead, the linear rate of convergence in (4.20) has been established for the last iterate π k generated by this algorithm. The convergence for the last iterate indicates that the SAMPD method will continuously improve the policy deployed by the system for implementation and evaluation. This is not the case for the convergence of the average or random iterate, since the average iterate will not be implemented and evaluated, and the convergence of the random iterate does not warrant continuous improvement of the generated policies.

Second, both Theorems 5 and 7 allow us to establish some strong large-deviation properties associated with the convergence of SPMD and SAPMD. Let us focus on the SAPMD method. For a given confidence level λ ∈ (0 , 1) and accuracy level /epsilon1 &gt; 0, if the number of iterations k satisfies

<!-- formula-not-decoded -->

then by (4.20) and Markov's inequality, we have

<!-- formula-not-decoded -->

In other words, with probability greater than 1 -λ , we have f ( π k ) -f ( π ∗ ) ≤ /epsilon1 . On the other hand, it is more difficult to derive a similar large deviation result for SPMD directly applied to unregularized problems (c.f. Theorem 6). Due to the sublinear rate of convergence and random selection of output, we need to run the algorithm for a few times to general several candidate solutions and apply a post-optimization procedure to choose from these candidate solutions in order to improve the the reliability of the algorithm (see Chapter 6 of [13] for more discussions).

## 5 Stochastic Estimation for Action-value Functions

In this section, we discuss the estimation of the action-value functions Q π or Q π τ through two different approaches. In Subsection 5.1, we assume the existence of a generative model for the Markov Chains so that we can estimate value functions by generating multiple independent trajectories starting from an arbitrary pair of state and action. In Subsection 5.2, we consider a more challenging setting where we only have access to a single trajectory observed when the dynamic system runs online. In this case, we employ and enhance the conditional temporal difference (CTD) method recently developed in [12] to estimate value functions. Throughout the section we assume that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 5.1 Multiple independent trajectories

In the multiple trajectory setting, starting from state-action pair ( s, a ) and following policy π k , we can generate M k independent trajectories of length T k , denoted by

<!-- formula-not-decoded -->

Let ξ k := { ζ i k ( s, a ) , i = 1 , . . . , M k , s ∈ S , a ∈ A} denote all these random variables. We can estimate Q π k in the SPMD method by

<!-- formula-not-decoded -->

We can show that Q π k ,ξ k satisfy (4.1)-(4.3) with

<!-- formula-not-decoded -->

for some absolute constant κ &gt; 0 (see Proposition 7 in the Appendix). By choosing T k and M k properly, we can show the convergence of the SPMD method employed with different stepsize rules as stated in Theorems 5 and 6.

Proposition 1 Suppose that η k = 1 -γ γµ in the SPMD method. If T k and M k are chosen such that

<!-- formula-not-decoded -->

with l := ⌈ log γ (1 / 4) ⌉ , then the relation in (4.9) holds. As a consequence, an /epsilon1 -solution of (1.9), i.e., a solution ¯ π s.t. E [ f (¯ π ) -f ( π ∗ ) + µ 1 -γ D (¯ π, π ∗ )] ≤ /epsilon1 , can be found in at most O (log γ /epsilon1 ) SPMD iterations. In addition, the total number of samples for ( s t , a t ) pairs can be bounded by

<!-- formula-not-decoded -->

Proof . Using the fact that γ l ≤ 1 / 4, we can easily check from (5.3) and the selection of T k and M k that (4.1)-(4.3) hold with ς k = 2 -( /floorleft k/l /floorright +2) and σ 2 k = 2 -( /floorleft k/l /floorright +2) . Suppose that an /epsilon1 -solution ¯ π will be found at the ¯ k iteration. By (4.9), we have

<!-- formula-not-decoded -->

which implies that the number of iterations is bounded by O ( l /floorleft ¯ k/l /floorright ) = O (log γ /epsilon1 ). Moreover by the definition of T k and M k , the total number of samples is bounded by

<!-- formula-not-decoded -->

To the best of our knowledge, this is the first time in the literature that an O (log(1 //epsilon1 ) //epsilon1 ) sampling complexity, after disregarding all constant factors, has been obtained for solving RL problems with strongly convex regularizers, even though problem (1.9) is still nonconvex. The previously best-known sampling complexity for RL problems with entropy regularizer was ˜ O ( |S||A| 2 //epsilon1 3 ) [20], and the author was not aware of an ˜ O (1 //epsilon1 ) sampling complexity results for any RL problems.

Below we discuss the sampling complexities of SPMD and SAPMD for solving RL problems with general convex regularizers.

Proposition 2 Consider the general RL problems with µ = 0 . Suppose that the number of iterations k is given a priori and η k = ( 2(1 -γ ) log |A| kσ 2 ) 1 / 2 . If T k ≥ T ≡ log γ (1 -γ ) /epsilon1 3(¯ c + ¯ h ) and M k = 1 , then an /epsilon1 -solution of problem of (1.9), i.e., a solution ¯ π s.t. E [ f (¯ π ) -f ( π ∗ )] ≤ /epsilon1 , can be found in at most O (log |A| / [(1 -γ ) 5 /epsilon1 2 ]) SPMD iterations. In addition, the total number of state-action samples can be bounded by

<!-- formula-not-decoded -->

Proof . We can easily check from (5.3) and the selection of T k and M k that (4.1)-(4.3) holds with ς k = /epsilon1/ 3 and σ 2 k = 2( /epsilon1 2 3 2 + 2(¯ c +¯ h ) 2 (1 -γ ) 2 ). Using these bounds in (4.10), we conclude that an /epsilon1 -solution will be found in at most

<!-- formula-not-decoded -->

iterations. Moreover, the total number of samples is bounded by |S||A| T ¯ k and hence by (5.5).

We can also establish the iteration and sampling complexities of the SAPMD method, in which we estimate Q π k τ k by

<!-- formula-not-decoded -->

Since τ 0 ≥ τ k , similar to (5.3), we can show that Q π k ,ξ k τ k satisfy (4.13)-(4.15) with

<!-- formula-not-decoded -->

for some absolute constant κ &gt; 0.

Proposition 3 Suppose that η k = 1 -γ γτ k and τ k = 1 √ γ log |A| 2 -( /floorleft k/l /floorright +1) in the SAPMD method. If T k and M k are chosen such that

<!-- formula-not-decoded -->

with l := ⌈ log γ (1 / 4) ⌉ , then the relation in (4.20) holds. As a consequence, an /epsilon1 -solution of (1.9), i.e., a solution ¯ π s.t. E [ f (¯ π ) -f ( π ∗ )] ≤ /epsilon1 , can be found in at most O (log γ /epsilon1 ) SAPMD iterations. In addition, the total number of samples for ( s t , a t ) pairs can be bounded by

<!-- formula-not-decoded -->

Proof . Using the fact that γ l ≤ 1 / 4, we can easily check from (5.7) and the selection of T k and M k that (4.13)-(4.15) hold with ς k = 2 -( /floorleft k/l /floorright +2) and σ 2 k = 4 -( /floorleft k/l /floorright +2) . Suppose that an /epsilon1 -solution ¯ π will be found at the ¯ k iteration. By (4.20), we have

<!-- formula-not-decoded -->

which implies that the number of iterations is bounded by O ( l /floorleft ¯ k/l /floorright ) = O (log γ /epsilon1 ). Moreover by the definition of T k and M k , the number of samples is bounded by

<!-- formula-not-decoded -->

To the best of our knowledge, the results in Propositions 2 and 3 appear to be new for policy gradient type methods. The previously best-known sampling complexity for policy gradient methods for RL problems was ˜ O ( |S||A| 2 //epsilon1 4 ) (e.g., [20]) although some improvements have been made under certain specific settings (e.g., [25]). Observe that the sampling complexity in (5.16) is slightly better than the one in (5.5) in the logarithmic terms. In fact, one can possibly further improve the dependence of the sampling complexity on γ in (5.5) by a factor of 1 / (1 -γ ) by allowing a slightly worse iteration complexity than the one in (5.6). This indicates that one needs to carefully consider the tradeoff between iteration and sampling complexities when implementing PMD type algorithms.

## 5.2 Conditional temporal difference

In this subsection, we enhance a recently developed temporal different (TD) type method, i.e., conditional temporal difference (CTD) method, and use it to estimate the action-value functions in an online manner. We focus on estimating Q π in SPMD since the estimation of Q π τ in SAPMD is similar.

For a given policy π , we denote the Bellman operator

<!-- formula-not-decoded -->

The action value function Q π corresponding to policy π satisfies the Bellman equation

<!-- formula-not-decoded -->

We also need to define a positive-definite weighting matrix M π ∈ R n × n to define the sampling scheme to evaluate policies using TD-type methods. A natural weighting matrix is the diagonal matrix M π = Diag( ν ( π )) ⊗ Diag( π ), where ν ( π ) is the steady state distribution induced by π and ⊗ denotes the Kronecker product.

Assumption 1 We make the following assumptions about policy π : (a) ν ( π )( s ) ≥ ν for some ν &gt; 0 , which holds when the Markov chain employed with policy π has a single ergodic class with unique stationary distribution, i.e., ν ( π ) = ν ( π ) P π ; and (b) π is sufficiently random, i.e., π ( s, a ) ≥ π for some π &gt; 0 , which can be enforced, for example, by adding some corresponding constraints through h π .

Note that Assumption 1.a) is widely accepted for evaluating policies using TD type methods in the RL literature, and that Assumption 1.b) requires that π assigns a non-zero probability to each action. We will discuss how to possibly relax these assumptions, especially Assumption 1.b) later in Remark 1.

In view of Assumption 1 we have M π /follows 0. With this weighting matrix M π , we define the operator F π as

<!-- formula-not-decoded -->

where T π is the Bellman operator defined in (5.9). Our goal is to find the root θ ∗ ≡ Q π of F ( θ ), i.e., F ( θ ∗ ) = 0. We can show that F is strongly monotone with strong monotonicity modulus bounded from below by Λ min := (1 -γ ) λ min ( M π ) . Here λ min ( A ) denotes the smallest eigenvalue of A . It can also be easily seen that F π is Lipschitz continuous with Lipschitz constant bounded by Λ max := (1 -γ ) λ max ( M π ) , where λ max ( A ) denotes the largest eigenvalue of A .

At time instant t ∈ Z + , we define the stochastic operator of F π as

<!-- formula-not-decoded -->

where ζ t = ( s t , a t , s t +1 , a t +1 ) denotes the state transition steps following policy π and e ( s t , a t ) denotes the unit vector. The CTD method uses the stochastic operator ˜ F π ( θ t , ζ t ) to update the parameters θ t iteratively as shown in Algorithm 3. It involves two algorithmic parameters: α ≥ 0 determines how often θ t is updated and β t ≥ 0 defines the learning rate. Observe that if α = 0, then CTD reduces to the classic TD learning method.

## Algorithm 3 Conditional Temporal Difference (CTD) for evaluating policy π

Let θ 1 , the nonnegative parameters α and { β t } be given.

for

t

= 1

, . . . , T

do

Collect α state transition steps without updating { θ t } , denoted as { ζ 1 t , ζ 2 t , . . . , ζ α t } . Set end for

When applying the general convergence results of CTD to our setting, we need to handle the following possible pitfalls. Firstly, current analysis of TD-type methods only provides bounds on E [ ‖ θ t -θ ∗ ‖ 2 2 ], which gives an upper bound on E [ ‖ θ t -Q π ‖ 2 ∞ ] and thus the bound on the total expected error (c.f., (4.2)). One needs to develop a tight enough bound on the bias ‖ E [ θ t ] -θ ∗ ‖ ∞ (c.f., (4.3)) to derive the overall best rate of convergence for the SPMD method. Secondly, the selection of α and { β t } that gives the best rate of convergence in terms of E [ ‖ θ t -θ ∗ ‖ 2 2 ] does not necessarily result in the best rate of convergence for SPMD, since we need to deal with the bias term explicitly.

The following result can be shown similarly to Lemma 4.1 of [12].

Lemma 16 Given the single ergodic class Markov chain ζ 1 1 , . . . , ζ α 1 , ζ 2 2 , . . . , ζ α 2 , . . . , there exists a constant C &gt; 0 and ρ ∈ [0 , 1) such that for every t, α ∈ Z + with probability 1,

<!-- formula-not-decoded -->

We can also show that the variance of ˜ F π is bounded as follows.

<!-- formula-not-decoded -->

The following result has been shown in Proposition 6.2 of [12].

<!-- formula-not-decoded -->

Lemma 17 If the algorithmic parameters in CTD are chosen such that

<!-- formula-not-decoded -->

with t 0 = 8max { Λ 2 max , 8(1 + γ ) 2 } /Λ 2 min , then

<!-- formula-not-decoded -->

where σ 2 F := 4(1 + γ ) 2 R 2 + ‖ θ ∗ ‖ 2 2 + 2(¯ c + ¯ h ) 2 and R 2 := 8 ‖ θ 1 -θ ∗ ‖ 2 2 + 3[ ‖ θ ∗ ‖ 2 2 +2(¯ c +¯ h ) 2 ] 4(1+ γ ) 2 . Moreover, we have E [ ‖ θ t -θ ∗ ‖ 2 2 ] ≤ R 2 for any t ≥ 1 .

We now enhance the above result with a bound on the bias term given by ‖ E [ θ t +1 ] -θ ∗ ‖ 2 . The proof of this result is put in the appendix since it is more technical.

Lemma 18 Suppose that the algorithmic parameters in CTD are set according to Lemma 17. Then we have

<!-- formula-not-decoded -->

We are now ready to establish the convergence of the SMPD method by using the CTD method to estimate the action-value functions. We focus on the case when µ &gt; 0, and the case for µ = 0 can be shown similarly.

Proposition 4 Suppose that η k = 1 -γ γµ in the SPMD method. If the initial point of CTD is set to θ 1 = 0 and the number of iterations T and the parameter α in CTD are set to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where l := ⌈ log γ (1 / 4) ⌉ and ¯ θ := √ n ¯ c +¯ h 1 -γ , then the relation in (4.9) holds. As a consequence, an /epsilon1 -solution of (1.9), i.e., a solution ¯ π s.t. E [ f (¯ π ) -f ( π ∗ ) + µ 1 -γ D (¯ π, π ∗ )] ≤ /epsilon1 , can be found in at most O (log γ /epsilon1 ) SPMD iterations. In addition, the total number of samples for ( s t , a t ) pairs can be bounded by

<!-- formula-not-decoded -->

Proof . Using the fact that γ l ≤ 1 / 4, we can easily check from Lemma 17, Lemma 18, and the selection of T and α that (4.1)-(4.3) hold with ς k = 2 -( /floorleft k/l /floorright +2) and σ 2 k = 2 -( /floorleft k/l /floorright +2) . Suppose that an /epsilon1 -solution ¯ π will be found at the ¯ k iteration. By (4.9), we have

<!-- formula-not-decoded -->

which implies that the number of iterations is bounded by O ( l /floorleft ¯ k/l /floorright ) = O (log γ /epsilon1 ). Moreover by the definition of T k and α k , the number of samples is bounded by

<!-- formula-not-decoded -->

The following result shows the convergence properties of the SAMPD method when the action-value function is estimated by using the CTD method.

Proposition 5 Suppose that η k = 1 -γ γτ k and τ k = 1 √ γ log |A| 2 -( /floorleft k/l /floorright +1) in the SAPMD method. If the initial point of CTD is set to θ 1 = 0 , the number of iterations T is set to

<!-- formula-not-decoded -->

and the parameter α in CTD is set to (5.15), where l := ⌈ log γ (1 / 4) ⌉ and ¯ θ := ¯ c + ¯ h + τ 0 log |A| 1 -γ , then the relation in (4.20) holds. As a consequence, an /epsilon1 -solution of (1.9), i.e., a solution ¯ π s.t. E [ f (¯ π ) -f ( π ∗ )] ≤ /epsilon1 , can be found in at most O (log γ /epsilon1 ) SPMD iterations. In addition, the total number of samples for ( s t , a t ) pairs can be bounded by

<!-- formula-not-decoded -->

Proof . The proof is similar to that of Proposition 4 except that we will show that (4.1)-(4.3) hold with ς k = 2 -( /floorleft k/l /floorright +2) and σ 2 k = 4 -( /floorleft k/l /floorright +2) . Moreover, we will use (4.20) instead of (4.9) to bound the number of iterations.

To the best of our knowledge, the complexity result in (5.16) is new in the RL literature, while the one in (5.18) is new for policy gradient type methods. It seems that this bound significantly improves the previously best-known O (1 //epsilon1 3 ) sampling complexity result for stochastic policy gradient methods (see [25] and Appendix C of [10] for more explanation).

Remark 1 In this subsection we focus on the more restrictive assumption M π /follows 0 in order to compare our results with the existing ones in the literature. Here we discuss how one can possibly relax this assumption.

If ν ( π )( s ) · π ( s, a ) = 0 for some ( s, a ) ∈ S ×A , one may define the weighting matrix M π = (1 -λ )Diag( ν ( π )) ⊗ Diag( π ) + λ n I for some sufficiently small λ ∈ (0 , 1) which depends on the target accuracy for solving the RL problem, where n = |S|×|A| . As a result, the algorithmic frameworks of CTD and SPMD, and their convergence analysis are still applicable to this more general setting. Obviously, the selection of λ will impact the efficiency estimate for policy evaluation.

An alternative approach that can relax Assumption 1.b), would be to first run the enhanced CTD method to the following equation

<!-- formula-not-decoded -->

to evaluate the state-value function V π . Then we estimate the action-value function Q π by using (1.5), i.e.,

<!-- formula-not-decoded -->

In order to use the above identity, we need to define an estimator of P ( s ′ | s, a ) by using a uniform policy π 0 ( ·| s ) := { 1 / | A | , . . . , 1 / | A |} . The sample size required to estimate the transition kernel from a single trajectory is an active research topic (see [24] and references therein). Current research has been focused only on bounding on the total error for estimating P ( s ′ | s, a ) for a given sample size, and there do not exist separate and tighter bounds on the bias for these estimators. Therefore, it is still not evident whether the same sampling complexity bounds in Propositions 4 and 5 can be maintained using this alternative approach to relax the assumption of non-zero probability to each action.

Remark 2 For problems of high dimension (i.e., n ≡ |S| × |A| is large), one often resorts to a parametric approximation of the value function. In this case it is possible to define a more general operator F π ( θ ) := Φ T M π ( Φθ -T π Φθ ) for some feature matrix Φ to evaluate the value functions (see Section 4 of [12] for a discussion about CTD with function approximation). Unless the column space of Φ spans the true value functions, an additional bias term will be introduced into the computation of gradients, resulting into an extra error term in the overall rate of convergence of PMD methods. In other words, these methods can only be guaranteed to converge to a neighborhood of the optimal solution. Nevertheless, the application of function approximation will significantly reduce the dependence of gradient computation on the problem dimension, i.e., from |S| × |A| to the number of columns of Φ .

## 6 Efficient Solution for General Subproblems

In this section, we study the convergence properties of the PMD methods for the situation where we do not have exact solutions for prox-mapping subprobems. Throughout this section, we assume that h π is differentiable and its gradients are Lipschitz continuous with Lipschitz constant L . We will first review Nesterov's accelerated gradient descent (AGD) method [18], and then discuss the overall gradient complexity of using this method for solving prox-mapping in the PMD methods. We will focus on the stochastic PMD methods since they cover deterministic methods as certain special cases.

6.1 Review of accelerated gradient descent

Let us denote X ≡ ∆ |A| and consider the problem of

<!-- formula-not-decoded -->

where φ : X → R is a smooth convex function such that

<!-- formula-not-decoded -->

Moreover, we assume that χ : X → R satisfies

<!-- formula-not-decoded -->

for some µ χ ≥ 0. Given ( x t -1 , y t -1 ) ∈ X × X , the accelerated gradient method performs the following updates:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some q t ∈ [0 , 1], r t ≥ 0, and ρ t ∈ [0 , 1] .

Below we slightly generalize the convergence results for the AGD method so that they depend on the distance D x x 0 rather than Φ ( y 0 ) -Φ ( x ) for any x ∈ X . This result better fits our need to analyze the convergence of inexact SPMD and SAPMD methods in the next two subsections.

Lemma 19 Let us denote µ Φ := µ φ + µ χ and t 0 := /floorleft 2 √ L φ /µ Φ -1 /floorright . If then for any x ∈ X ,

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof . Using the discussions in Corollary 3.5 of [13] (and the possible strong convexity of χ ), we can check that the conclusions in Theorems 3.6 and 3.7 of [13] hold for the AGD method applied to problem (6.1). It then follows from Theorem 3.6 of [13] that

<!-- formula-not-decoded -->

Moreover, it follows from Theorem 3.7 of [13] that for any t ≥ t 0 ,

<!-- formula-not-decoded -->

where the last inequality follows from (6.7) (with t = t 0 ) and the facts that for any 2 t t

<!-- formula-not-decoded -->

≤ ≤ 0 . The result then follows by combining these observations.

6.2 Convergence of inexact SPMD

In this subsection, we study the convergence properties of the SPMD method when its subproblems are solved inexactly by using the AGD method (see Algorithm 4). Observe that we use the same initial point π 0 whenever calling the AGD method. To use a dynamic initial point (e.g., v k ) will make the analysis more complicated since we do not have a uniform bound on the KL divergence D π v k for an arbitrary v k . To do so probably will require us to use other distance generating functions than the entropy function.

## Algorithm 4 The SPMD method with inexact subproblem solutions

Input: initial points π 0 = v 0 and stepsizes η k ≥ 0.

<!-- formula-not-decoded -->

Apply T k AGD iterations (with initial points x 0 = y 0 = π 0 ) to

<!-- formula-not-decoded -->

Set ( π k +1 , v k +1 ) = ( y T k +1 , x T k +1 ). end for

In the sequel, we will denote ε k ≡ ε ( T k ) to simplify notations. The following result will take place of Lemma 4 in our convergence analysis.

Lemma 20 For any π ( ·| s ) ∈ X , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof . It follows from Lemma 19 (with µ Φ = 1 + µη k and L φ = L ) that

<!-- formula-not-decoded -->

Using the definition of Φ k , we have

<!-- formula-not-decoded -->

which proves (6.9). Setting π = π k and π = π k +1 respectively, in the above conclusion, we obtain

<!-- formula-not-decoded -->

Then (6.10) follows by combining these two inequalities.

Proposition 6 For any s ∈ S , we have

<!-- formula-not-decoded -->

Moreover, we have

Proof . Similar to (4.8), we have

<!-- formula-not-decoded -->

It follows from (6.10) that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies that where the last inequality follows from the fact that d π k +1 s ( s ) ≥ (1 -γ ) due to the definition of d π k +1 s in (2.1). The result in (6.11) then follows immediately from (6.12) and the above inequality.

We now establish an important recursion about the inexact SPMD method in Algorithm 4.

Lemma 21 Suppose that η k = η = 1 -γ γµ and ε k ≤ ε k -1 for any k ≥ 0 in the inexact SPMD method, we have

<!-- formula-not-decoded -->

Proof . By (6.9) (with p = π ∗ ), we have

<!-- formula-not-decoded -->

which, in view of (4.7), then implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking expectation w.r.t. ξ /ceilingleft k /ceilingright and ν ∗ on both sides of the above inequality, and using Lemma 3 and the relation in (4.6), we arrive at

<!-- formula-not-decoded -->

Noting V π k +1 ( s ) -V π k ( s ) = V π k +1 ( s ) -V π ∗ ( s ) -[ V π k ( s ) -V π ∗ ( s )], rearranging the terms in the above inequality, and using the definition of f in (1.9), we arrive at

<!-- formula-not-decoded -->

The result then follows immediately by the selection of η and the assumption ε k ≤ ε k -1 .

We now are now ready to state the convergence rate of the SPMD method with inexact prox-mapping. We focus on the case when µ &gt; 0.

<!-- formula-not-decoded -->

Proof

<!-- formula-not-decoded -->

. The result follows as an immediate consequence of Proposition 21 and Lemma 11.

In view of Theorem 8, the inexact solutions of the subproblems barely affect the iteration and sampling complexities of the SPMD method as long as ε k ≤ (1 -γ ) 2 2 -( /floorleft ( k +1) /l /floorright +2) . Notice that an /epsilon1 -solution of problem (1.9), i.e., a solution ¯ π s.t. E [ f (¯ π ) -f ( π ∗ )] ≤ /epsilon1 , can be found at the ¯ k -th iteration with

Also observe that the condition number of the subproblem is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining these observations with Lemma 19, we conclude that the total number of gradient computations of h can be bounded by

<!-- formula-not-decoded -->

## 6.3 Convergence of inexact SAPMD

In this subsection, we study the convergence properties of the SAPMD method when its subproblems are solved inexactly by using the AGD method (see Algorithm 5).

## Algorithm 5 The Inexact SAPMD method

Input: initial points π 0 = v 0 , stepsizes η k ≥ 0, and regularization parameters τ k ≥ 0.

for k = 0 , 1 , . . . , do

Apply T k AGD iterations (with initial points x 0 = y 0 = π 0 ) to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the sequel, we will still denote ε k ≡ ε ( T k ) to simplify notations. The following result has the same role as Lemma 20 in our convergence analysis.

Lemma 22 For any π ( ·| s ) ∈ X , we have

<!-- formula-not-decoded -->

Moreover, we have

<!-- formula-not-decoded -->

Proof . The proof is the same as that for Lemma 20 except that Lemma 19 will be applied to problem (6.15) (with µ Φ = 1 + τ k η k and L φ = L ).

Lemma 23 For any s ∈ S , we have

<!-- formula-not-decoded -->

Proof . The proof is similar to that for Lemma 6 with the following two exceptions: (a) we will apply Lemma 2 (i.e., the performance difference lemma) to the perturbed value functions V π τ k instead of V π to obtain a result similar to (6.12); and (b) we will use use (6.17) in place of (6.10) to derive a bound similar to (6.13).

Lemma 24 Suppose that 1 + η k τ k = 1 /γ and ε k ≤ ε k -1 in the SAPMD method. Then for any k ≥ 0 , we have

<!-- formula-not-decoded -->

Proof . By (6.16) (with p = π ∗ ), we have

<!-- formula-not-decoded -->

which, in view of (6.18), implies that

<!-- formula-not-decoded -->

Taking expectation w.r.t. ξ /ceilingleft k /ceilingright and ν ∗ on both sides of the above inequality, and using Lemma 3 (with h π replaced by h π + τ k D π π 0 ( s t ) and Q π replaced by Q π τ ) and the relation in (4.6), we arrive at

<!-- formula-not-decoded -->

Noting V π k +1 τ k ( s ) -V π k τ k ( s ) = V π k +1 τ k ( s ) -V π ∗ τ k ( s ) -[ V π k τ k ( s ) -V π ∗ τ k ( s )] and rearranging the terms in the above inequality, we have

<!-- formula-not-decoded -->

The result then follows from 1 + η τ = 1 /γ , the assumptions τ τ , ε ε

<!-- formula-not-decoded -->

k k k ≥ k +1 k ≤ k -1 and (3.19).

Proof . The result follows as an immediate consequence of Lemma 24, Lemma 11, and an argument similar to the one to prove Theorem 7.

In view of Theorem 9, the inexact solution of the subproblem barely affect the iteration and sampling complexities of the SAPMD method as long as ε k ≤ (1 -γ ) 2 2 γ 2 (1+ γ ) . Notice that an /epsilon1 -solution of problem (1.9), i.e., a solution ¯ π s.t. E [ f (¯ π ) -f ( π ∗ )] ≤ /epsilon1 , can be found at the ¯ k -th iteration with

<!-- formula-not-decoded -->

Also observe that the condition number of the subproblem is given by

<!-- formula-not-decoded -->

Combining these observations with Lemma 19, we conclude that the total number of gradient computations of h can be bounded by

<!-- formula-not-decoded -->

## 7 Concluding Remarks

In this paper, we present the policy mirror descent (PMD) method and show that it can achieve the linear and sublinear rate of convergence for RL problems with strongly convex or general convex regularizers, respectively. We then present a more general form of the PMD method, referred to as the approximate policy mirror descent (APMD) method, obtained by adding adaptive perturbations to the action-value functions and show that it can achieve the linear convergence rate for RL problems with general convex regularizers. We develop the stochastic PMD and APMD methods and derive general conditions on the bias and overall expected error to guarantee the convergence of these methods. Using these conditions, we establish new sampling complexity bounds of RL problems by using two different sampling schemes, i.e., either using a straightforward generative model or a more involved conditional temporal different method. The latter setting requires us to establish a bound on the bias for estimating action-value functions, which might be of independent interest. Finally, we establish the conditions on the accuracy required for the prox-mapping subproblems in these PMD type methods, as well as the overall complexity of computing the gradients of the regularizers. In the future, it will be interesting to study how to incorporate exploration into policy mirror descent to handle rarely visited states and actions. Moreover, since this paper focuses on the theoretical studies, it will be also rewarding to derive simplified PMD algorithms and conduct numerical experiments to demonstrate possible advantages of the proposed algorithms.

Acknowledgement: The author appreciates very much Caleb Ju, Sajad Khodaddadian, Tianjiao Li, Yan Li and two anonymous reviewers for their careful reading and a few suggested corrections for earlier versions of this paper.

## Reference

1. A. Agarwal, S. M. Kakade, J. D. Lee, and G. Mahajan. On the theory of policy gradient methods: Optimality, approximation, and distribution shift. arXiv , pages arXiv-1908.00261, 2019.
2. A. Beck and M. Teboulle. Mirror descent and nonlinear projected subgradient methods for convex optimization. SIAM Journal on Optimization , 27:927-956, 2003.
3. R. Bellman and S. Dreyfus. Functional approximations and dynamic programming. Mathematical Tables and Other Aids to Computation , 13(68):247-251, 1959.
4. Jalaj Bhandari and Daniel Russo. A Note on the Linear Convergence of Policy Gradient Methods. arXiv e-prints , page arXiv:2007.11120, July 2020.
5. Shicong Cen, Chen Cheng, Yuxin Chen, Yuting Wei, and Yuejie Chi. Fast Global Convergence of Natural Policy Gradient Methods with Entropy Regularization. arXiv e-prints , page arXiv:2007.06558, July 2020.
6. C. D. Dang and G. Lan. On the convergence properties of non-euclidean extragradient methods for variational inequalities with generalized monotone operators. Computational Optimization and Applications , 60(2):277-310, 2015.
7. Eyal Even-Dar, Sham. M. Kakade, and Yishay Mansour. Online markov decision processes. Mathematics of Operations Research , 34(3):726-736, 2009.
8. F. Facchinei and J. Pang. Finite-Dimensional Variational Inequalities and Complementarity Problems, Volumes I and II . Comprehensive Study in Mathematics. Springer-Verlag, New York, 2003.
9. S. Kakade and J. Langford. Approximately optimal approximate reinforcement learning. In Proc. International Conference on Machine Learning (ICML) , 2002.
10. S. Khodadadian, Z. Chen, and S. T. Maguluri. Finite-sample analysis of off-policy natural actor-critic algorithm. arXiv , pages arXiv-2102.09318, 2021.
11. G. Kotsalis, G. Lan, and T. Li. Simple and optimal methods for stochastic variational inequalities, I: operator extrapolation. arXiv , pages arXiv-2011.02987, 2020.
12. G. Kotsalis, G. Lan, and T. Li. Simple and optimal methods for stochastic variational inequalities, II: Markovian noise and policy evaluation in reinforcement learning. arXiv , pages arXiv-2011.08434, 2020.
13. G. Lan. First-order and Stochastic Optimization Methods for Machine Learning . Springer Nature, Switzerland AG, 2020.
14. B. Liu, Q. Cai, Z. Yang, and Z. Wang. Neural proximal/trust region policy optimization attains globally optimal policy. arXiv , pages arXiv-1906.10306, 2019.
15. Jincheng Mei, Chenjun Xiao, Csaba Szepesvari, and Dale Schuurmans. On the Global Convergence Rates of Softmax Policy Gradient Methods. arXiv e-prints , page arXiv:2005.06392, May 2020.
16. A. S. Nemirovski, A. Juditsky, G. Lan, and A. Shapiro. Robust stochastic approximation approach to stochastic programming. SIAM Journal on Optimization , 19:1574-1609, 2009.
17. A. S. Nemirovski and D. Yudin. Problem complexity and method efficiency in optimization . Wiley-Interscience Series in Discrete Mathematics. John Wiley, XV, 1983.
18. Y. E. Nesterov. A method for unconstrained convex minimization problem with the rate of convergence O (1 /k 2 ). Doklady AN SSSR , 269:543-547, 1983.
19. Martin L. Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming . John Wiley &amp; Sons, Inc., USA, 1st edition, 1994.
20. Lior Shani, Yonathan Efroni, and Shie Mannor. Adaptive trust region policy optimization: Global convergence and faster rates for regularized mdps. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020 , pages 5668-5675. AAAI Press, 2020.
21. R.S. Sutton, D. McAllester, S. Singh, and Y. Mansour. Policy gradient methods for reinforcement learning with function approximation. In NIPS'99: Proceedings of the 12th International Conference on Neural Information Processing Systems , pages 1057-1063, 1999.
22. M. Tomar, Lior Shani, Yonathan Efroni, and Mohammad Ghavamzadeh. Mirror descent policy optimization. ArXiv , abs/2005.09814, 2020.
23. L. Wang, Q. Cai, Zhuoran Yang, and Zhaoran Wang. Neural policy gradient methods: Global optimality and rates of convergence. ArXiv , abs/1909.01150, 2020.
24. G. Wolfer and A. Kontorovich. Statistical estimation of ergodic markov chain kernel over discrete state space. arXiv , pages arXiv-1809.05014v6, 2020.
25. T. Xu, Zhe Wang, and Yingbin Liang. Improving sample complexity bounds for actor-critic algorithms. ArXiv , abs/2004.12956, 2020.

## Appendix A: Concentration Bounds for l ∞ -bounded Noise

We first show how to bound the expectation of the maximum for a finite number of sub-exponential variables.

Lemma 25 Let ‖ X ‖ ψ 1 := inf { t &gt; 0 : exp( | X | /t ) ≤ exp(2) } denote the sub-exponential norm of X . For a given sequence of sub-exponential variables { X i } n i =1 with E [ X i ] ≤ v and ‖ X i ‖ ψ 1 ≤ σ , we have

<!-- formula-not-decoded -->

where C denotes an absolute constant.

Proof . By the property of sub-exponential random variables (Section 2.7 of [ ? ]), we know that Y i = X i -E [ X i ] is also sub-exponential with ‖ Y i ‖ ψ 1 ≤ C 1 ‖ X i ‖ ψ 1 ≤ C 1 σ for some absolute constant C 1 &gt; 0. Hence by Proposition 2.7.1 of [ ? ], there exists an absolute constant C &gt; 0 such that E [exp( λY i )] ≤ exp( C 2 σ 2 λ 2 ) , ∀| λ | ≤ 1 / ( Cσ ) . Using the previous observation, we have

<!-- formula-not-decoded -->

which implies E [max i Y i ] ≤ log n/λ + C 2 σ 2 λ, ∀| λ | ≤ 1 / ( Cσ ) . Choosing λ = 1 / ( Cσ ), we obtain E [max i Y i ] ≤ Cσ (log n +1) . By combining this relation with the definition of Y i , we conclude that E [max i X i ] ≤ E [max i Y i ]+ v ≤ Cσ (log n +1)+ v.

Proposition 7 For δ k := Q π k ,ξ k -Q π k ∈ R |S|×|A| , we have

<!-- formula-not-decoded -->

where κ &gt; 0 denotes an absolute constant.

Proof . To proceed, we denote δ k s,a := Q π k ,ξ k ( s, a ) -Q π k ( s, a ), and hence

<!-- formula-not-decoded -->

Note that by definition, for each ( s, a ) pair, we have M k independent trajectories of length T k starting from ( s, a ). Let us denote Z i := ∑ T k -1 t =0 γ t [ c ( s i t , a i t ) + h π k ( s i t ) ] , i = 1 , . . . , M k . Hence,

Since each Z i -Q π k ( s, a ) is independent of each other, it is immediate to see that Y s,a := ( δ k s,a ) 2 is a subexponential with ‖ Y s,a ‖ ψ 1 ≤ ( c + h ) 2 (1 -γ ) 2 M k . Also note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus in view of Lemma 25, with σ = ( c + h ) 2 (1 -γ ) 2 M k , and v = ( c + h ) 2 (1 -γ ) 2 M k + ( c + h ) 2 (1 -γ ) 2 γ 2 T k , we conclude that

<!-- formula-not-decoded -->

Appendix B: Bias for Conditional Temporal Difference Methods

## Proof of Lemma 18.

Proof . For simplicity, let us denote ¯ θ t ≡ E [ θ t ], ζ t ≡ ( ζ 1 t , . . . , ζ α t ) and ζ /ceilingleft t /ceilingright = ( ζ 1 , . . . , ζ t ). Also let us denote δ F t := F π ( θ t ) -E [ ˜ F π ( θ t , ζ α t ) | ζ /ceilingleft t -1 /ceilingright ] and ¯ δ F t = E ζ /ceilingleft t -1 /ceilingright [ δ F t ]. It follows from Jensen's ienquality and Lemma 17 that

<!-- formula-not-decoded -->

Also by Jensen's inequality, Lemma 16 and Lemma 17, we have

<!-- formula-not-decoded -->

Notice that then imply that

<!-- formula-not-decoded -->

where the last inequality follows from

<!-- formula-not-decoded -->

due to the selection of β t in (5.13). Now let us denote Γ t := { 1 t = 0 , (1 -3 t + t 0 -1 ) Γ t -1 t ≥ 1 , or equivalently, Γ t :=

( t 0 -1)( t 0 -2)( t 0 -3) ( t + t 0 -1)( t + t 0 -2)( t + t 0 -3)) . Dividing both sides of (7.3) by Γ t and taking the telescopic sum, we have

Noting that

<!-- formula-not-decoded -->

we conclude

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

from which the result holds since ¯ θ 1 = θ 1 .

<!-- formula-not-decoded -->

Now conditional on ζ /ceilingleft t -1 /ceilingright , taking expectation w.r.t. ζ t on (5.11), we have E [ θ t +1 | ζ /ceilingleft t -1 /ceilingright ] = θ t -β t F π ( θ t ) + β t δ F t . Taking further expectation w.r.t. ζ /ceilingleft t -1 /ceilingright and using the linearity of F , we have ¯ θ t +1 = ¯ θ t -β t F π ( ¯ θ t )+ β t ¯ δ F t , which implies

<!-- formula-not-decoded -->

The above inequality, together with (7.1), (7.2) and the facts that

<!-- formula-not-decoded -->