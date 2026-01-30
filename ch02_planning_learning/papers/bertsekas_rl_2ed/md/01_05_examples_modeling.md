# 1.6.1-1.6.4: Examples & Modeling

**Source:** Bertsekas, "A Course in Reinforcement Learning" (2nd ed., 2025)
**Pages:** 79-91
**Topics:** modeling, termination state, discrete optimization, finite to infinite horizon, reformulations

---

or equivalently K = F ( K ) [cf. Eq. (1.41)].

In conclusion, when restricted to quadratic functions J ( x ) = Kx 2 with K ≥ 0, the Bellman equation (1.45) is equivalent to the equation

<!-- formula-not-decoded -->

We refer to this equation as the Riccati equation and to the function F as the Riccati operator . ‡ Moreover, the policy corresponding to K ∗ , as per Eqs. (1.47)-(1.48), attains the minimum in Bellman's equation, and is given by Eqs. (1.43)-(1.44).

The Riccati equation can be visualized and solved graphically as illustrated in Fig. 1.5.1. As shown in the figure, the quadratic coe ffi cient K ∗ that corresponds to the optimal cost function J * [cf. Eq. (1.40)] is the unique solution of the Riccati equation K = F ( K ) within the nonnegative real line.

## The Riccati Equation for a Stable Linear Policy

We can also characterize the cost function of a policy θ that is linear of the form θ ( x ) = Lx , and is also stable, in the sense that the scalar L satisfies ♣ a + bL ♣ &lt; 1, so that the corresponding closed-loop system

<!-- formula-not-decoded -->

is stable (its state x k converges to 0 as k →∞ ). In particular, we can show that its cost function has the form

<!-- formula-not-decoded -->

This is an algebraic form of the Riccati di ff erential equation, which was invented in its one-dimensional form by count Jacopo Riccati in the 1700s, and has played an important role in control theory. It has been studied extensively in its di ff erential and di ff erence matrix versions; see the book by Lancaster and Rodman [LR95], and the paper collection by Bittanti, Laub, and Willems [BLW91], which also includes a historical account by Bittanti [Bit91] of Riccati's remarkable life and accomplishments.

‡ The Riccati operator is a special case of the Bellman operator , denoted by T , which transforms a function J into the right side of Bellman's equation:

<!-- formula-not-decoded -->

Thus the Bellman operator T transforms a function J of x into another function TJ also of x . Bellman operators allow a succinct abstract description of the problem's data, and are fundamental in the theory of abstract DP (see the author's monographs [Ber22a] and [Ber22b]). We may view the Riccati operator as the restriction of the Bellman operator to the subspace of quadratic functions of x .

arK

7+ 62K +

ar + 9

45° Line

Riccati Operator

Figure 1.5.1 Graphical construction of the solutions of the Riccati equation (1.41)-(1.42) for the linear quadratic problem. The optimal cost function is J ∗ ( x ) = K ∗ x 2 , where the scalar K ∗ solves the fixed point equation K = F ( K ) ↪ with F being the Riccati operator given by

<!-- image -->

<!-- formula-not-decoded -->

Note that F is concave and monotonically increasing in the interval ( -rglyph[triangleleft]b 2 ↪ ∞ ) and 'flattens out' as K → ∞ , as shown in the figure. The quadratic Riccati equation K = F ( K ) also has another solution, denoted by ¯ K , which is negative and therefore of no interest.

where K L solves the equation

<!-- formula-not-decoded -->

with F L defined by

<!-- formula-not-decoded -->

This equation is called the Riccati equation for the stable policy θ ( x ) = Lx . It is illustrated in Fig. 1.5.2, and it is linear, with linear coe ffi cient ( a + bL ) 2 that is strictly less than 1. Hence the line that represents the graph of F L intersects the 45-degree line at a unique point, which defines the quadratic cost coe ffi cient K L .

The Riccati equation (1.50)-(1.51) for θ ( x ) = Lx may be justified by verifying that it is in fact the Bellman equation for θ ,

<!-- formula-not-decoded -->

62

+9

Unstable L

a + 6L| &gt; 1

0

FI(K) for L Corresponding to an Unstable and a Stable Policy

Stable L

a +6L &lt;1

<!-- image -->

K

L

Figure 1.5.2 Illustration of the construction of the cost function of a linear policy θ ( x ) = Lx , which is stable, i.e., ♣ a + bL ♣ &lt; 1 glyph[triangleright] The cost function J θ ( x ) has the form

<!-- formula-not-decoded -->

with K L obtained as the unique solution of the linear equation K = F L ( K ) ↪ where

<!-- formula-not-decoded -->

/negationslash is the Riccati equation operator corresponding to θ ( x ) = Lx . If θ is not stable, i.e., ♣ a + bL ♣ ≥ 1 ↪ we have J θ ( x ) = ∞ for all x = 0, but the equation has K = F L ( K ) still has a solution that is of no interest within our context.

/negationslash

[cf. Eq. (1.34)], restricted to quadratic functions of the form J ( x ) = Kx 2 . We note, however, that J θ ( x ) = K L x 2 is the solution of the Riccati equation (1.50)-(1.51) only when θ ( x ) = Lx is stable. If θ is not stable, i.e., ♣ a + bL ♣ ≥ 1, then (since q &gt; 0 and r &gt; 0) we have J θ ( x ) = ∞ for all x = 0. Then, the Riccati equation (1.50)-(1.51) is still defined, but its solution is negative and is of no interest within our context.

## Value Iteration

The VI algorithm for our linear quadratic problem is given by

<!-- formula-not-decoded -->

When J k is quadratic of the form J k ( x ) = K k x 2 with K k ≥ 0, it can be seen that the VI iterate J k +1 is also quadratic of the form J k +1 ( x ) = K k +1 x 2 ,

Corresponding to an Unstable and a Stable Policy

a-rK

+ 62K +

Kk+1

0

K

Kk Kk+1

Riccati Operator

Figure 1.5.3 Graphical illustration of value iteration for the linear quadratic problem. It has the form K k +1 = F ( K k ), where F is the Riccati operator,

<!-- image -->

<!-- formula-not-decoded -->

The algorithm converges to K ∗ starting from any K 0 ≥ 0.

where

<!-- formula-not-decoded -->

with F being the Riccati operator of Eq. (1.49). The algorithm is illustrated in Fig. 1.5.3. As can be seen from the figure, when starting from any K 0 ≥ 0, the algorithm generates a sequence ¶ K k ♦ of nonnegative scalars that converges to K ∗ .

## 1.5.1 Visualizing Approximation in Value Space - Region of Stability

The use of Riccati equations allows insightful visualization of approximation in value space. This visualization, although specialized to linear quadratic problems, is consistent with related visualizations for more general infinite horizon problems; this is a recurring theme in what follows. In particular, in the books [Ber20a] and [Ber22a], Bellman operators, which define the Bellman equations, are used in place of Riccati operators, which define the Riccati equations.

In summary, we will aim to show that:

- (a) Approximation in value space with one-step lookahead can be viewed as a Newton step for solving the Bellman equation, and maps the

ON-LINE PLAY

OFF-LINE TRAINING

Figure 1.5.4 Illustration of the interpretation of approximation in value space with one-step lookahead as a Newton step that maps ˜ J to the cost function J ˜ θ of the one-step lookahead policy.

<!-- image -->

terminal cost function approximation ˜ J to the cost function J ˜ θ of the one-step lookahead policy; see Fig. 1.5.4.

- (b) Approximation in value space with multistep lookahead and truncated rollout can be viewed as a Newton step for solving the Bellman equation, and maps the result of multiple VI iterations starting with the terminal cost function approximation ˜ J to the cost function J ˜ θ of the multistep lookahead policy; see Fig. 1.5.5.

Our derivation will be given for the one-dimensional linear quadratic problem, but applies far more generally . The reason is that the Bellman equation is valid universally in DP, and the corresponding Bellman operator has a concavity property that is well-suited for the application of Newton's method; see the books [Ber20a] and [Ber22a], where the connection of approximation in value space with Newton's method was first developed in detail.

Let us consider one-step lookahead minimization with any terminal cost function approximation of the form ˜ J ( x ) = Kx 2 , where K ≥ 0. We have derived the one-step lookahead policy θ K ( x ) in Eqs. (1.47)-(1.48), by minimizing the right side of Bellman's equation when J ( x ) = Kx 2 :

<!-- formula-not-decoded -->

We can break this minimization into a sequence of two minimizations as

NEWTON STEP for Bellman Eq. 2-Step Lookahead Minimization

ON-LINE PLAY

ON-LINE PLAY Lookahead Tree States

<!-- image -->

OFF-LINE TRAINING

Figure 1.5.5 Illustration of the interpretation of approximation in value space with multistep lookahead and truncated rollout as a Newton step, which maps the result of multiple VI iterations starting with the terminal cost function approximation ˜ J to the cost function J ˜ θ of the multistep lookahead policy.

follows:

<!-- formula-not-decoded -->

From this equation, it follows that

<!-- formula-not-decoded -->

where the function F L ( K ) is defined by

<!-- formula-not-decoded -->

Figure 1.5.6 illustrates the relation (1.52)-(1.53), and shows how the graph of the Riccati operator F can be obtained as the lower envelope of the linear operators F L , as L ranges over the real numbers.

## One-Step Lookahead Minimization and Newton's Method

Let us now fix the terminal cost function approximation to some ˜ Kx 2 , where ˜ K ≥ 0, and consider the corresponding one-step lookahead policy,

abK

Tangent Riccati

Operator at K

abK

[=--

Tangent Riccati Operator at

FL (K) = (a+bL)2K+9+rI2

arK

Figure 1.5.6 Illustration of how the graph of the Riccati operator F can be obtained as the lower envelope of the linear operators

<!-- image -->

<!-- formula-not-decoded -->

as L ranges over the real numbers. We have

<!-- formula-not-decoded -->

cf. Eq. (1.52). Moreover, for any fixed ˜ K , the scalar ˜ L that attains the minimum is given by

<!-- formula-not-decoded -->

[cf. Eq. (1.48)], and is such that the line corresponding to the graph of F ˜ L is tangent to the graph of F at ˜ K , as shown in the figure.

which we will denote by ˜ θ . Figure 1.5.7 illustrates the corresponding linear function F ˜ L , and shows that its graph is a tangent line to the graph of F at the point ˜ K [cf. Fig. 1.5.6 and Eq. (1.53)].

Thus the function F ˜ L can be viewed as a linearization of F at the point ˜ K , and defines a linearized problem: to find a solution of the equation

<!-- formula-not-decoded -->

The important point now is that the solution of this equation, denoted K ˜ L , is the same as the one obtained from a single iteration of Newton's method for solving the Riccati equation, starting from the point ˜ K . This is illustrated in Fig. 1.5.7, and is also justified analytically in Exercise 1.7.

abK

дІ(Ук)

+=

0

Fi (K)

K

K

abK

r + 62 K

Figure 1.5.7 Illustration of approximation in value space with one-step lookahead for the linear quadratic problem. Given a terminal cost approximation ˜ J ( x ) = ˜ Kx 2 , we compute the corresponding linear policy ˜ θ ( x ) = ˜ Lx , where

<!-- image -->

<!-- formula-not-decoded -->

and the corresponding cost function K ˜ L x 2 , using the Newton step shown.

To explain this connection, we note that the classical form of Newton's method for solving a fixed point problem of the form y = T ( y ), where y is an n -dimensional vector, operates as follows: At the current iterate y k , we linearize T and find the solution y k +1 of the corresponding linear fixed point problem. Assuming T is di ff erentiable, the linearization is obtained by using a first order Taylor expansion:

<!-- formula-not-decoded -->

where ∂ T ( y k ) glyph[triangleleft] ∂ y is the n × n Jacobian matrix of T evaluated at the vector y k , as indicated in Fig. 1.5.7.

The most commonly given convergence rate property of Newton's method is quadratic convergence . It states that near the solution y ∗ , we have

<!-- formula-not-decoded -->

where ‖ · ‖ is the Euclidean norm, and holds assuming the Jacobian matrix exists and is Lipschitz continuous (see [Ber16], Section 1.4). There are extensions of Newton's method that are based on solving a linearized

T. :

=

abK1

7+ 62K1

K

Fi (K)

K1

Starting point enhancement osition Evaluator Engine Newton step Starting point enhancement

abK1

7 + 62K1

Figure 1.5.8 Illustration of approximation in value space with two-step lookahead for the linear quadratic problem. Starting with a terminal cost approximation ˜ J ( x ) = ˜ Kx 2 , we obtain K 1 using a single value iteration, thereby enhancing the starting point of the Newton step. We then compute the corresponding linear policy ˜ θ ( x ) = ˜ Lx , where

<!-- image -->

<!-- formula-not-decoded -->

and the corresponding cost function K ˜ L x 2 , using the Newton step shown. The figure shows that for any K ≥ 0, the corresponding /lscript -step lookahead policy will be stable for all /lscript larger than some threshold.

system at the current iterate, but relax the di ff erentiability requirement to piecewise di ff erentiability, and/or component concavity, while maintaining the either a quadratic or a similarly fast superlinear convergence property of the method; see the monograph [Ber22a] (Appendix A) and the paper [Ber22c], which also provide a convergence analysis.

Note also that if the one-step lookahead policy is stable, i.e., ♣ a + b ˜ L ♣ &lt; 1 ↪ then K ˜ L is the quadratic cost coe ffi cient of its cost function, i.e.,

<!-- formula-not-decoded -->

/negationslash

The reason is that J ˜ θ solves the Bellman equation for policy ˜ θ . On the other hand, if ˜ θ is not stable, then in view of the positive definite quadratic cost per stage, we have J ˜ θ ( x ) = ∞ for all x = 0.

## Multistep Lookahead

In the case of /lscript -step lookahead minimization, a similar Newton step inter-

[=

Riccati Equation Formulas for One-Dimensional Problems

Riccati equation for minimization [cf. Eqs. (1.41) and (1.42)]

<!-- formula-not-decoded -->

Riccati equation for a linear policy θ ( x ) = Lx

<!-- formula-not-decoded -->

Cost coe ffi cient K L of a stable linear policy θ ( x ) = Lx

<!-- formula-not-decoded -->

Linear coe ffi cient L K of the one-step lookahead linear policy θ K for K in the region of stability [cf. Eq. (1.48)]

<!-- formula-not-decoded -->

Quadratic cost coe ffi cient ˜ K of a one-step lookahead linear policy θ K for K in the region of stability

Obtained as the solution of the linearized Riccati equation

<!-- formula-not-decoded -->

or equivalently by a Newton iteration starting from K .

pretation is possible. Instead of linearizing F at ˜ K , we linearize at

<!-- formula-not-decoded -->

i.e., the result of /lscript -1 successive applications of F starting with ˜ K . Each application of F corresponds to a value iteration. Thus the e ff ective starting point for the Newton step is F /lscript -1 ( ˜ K ). Figure 1.5.8 depicts the case /lscript = 2.

## Region of Stability

It is also useful to define the region of stability as the set of K ≥ 0 such

Unstable Policy

Slope=1

Instability Region Stability Region Slope=1

also

Region of Convergence of

Newton's Method

Line Stable Policy Unstable Policy Region of stability

Also Region of Convergence of Newton's Method

Stable Policy

Figure 1.5.9 Illustration of the region of stability, i.e., the set of K ≥ 0 such that the one-step lookahead policy θ K is stable. This is also the set of initial conditions for which Newton's method converges to K ∗ asymptotically.

<!-- image -->

that

Line Stable Policy Unstable Policy Region of stability

<!-- formula-not-decoded -->

where L K is the linear coe ffi cient of the one-step lookahead policy corresponding to K ; cf. Eq. (1.48). The region of stability may also be viewed as the region of convergence of Newton's method . It is the set of starting points K for which Newton's method, applied to the Riccati equation F = F ( K ), converges to K ∗ asymptotically, and with a quadratic convergence rate (asymptotically as K → K ∗ ). Note that for our one-dimensional problem, the region of stability is the interval ( K S ↪ ∞ ) that is characterized by the single point K S where F has derivative equal to 1; see Fig. 1.5.9.

For multidimensional problems, the region of stability may not be characterized as easily. Still, however, it is generally true that the region of stability is enlarged as the length of the lookahead increases .

Indeed, with increased lookahead, the e ff ective starting point

<!-- formula-not-decoded -->

is pushed more and more within the region of stability. In particular, for any given K ≥ 0 , the corresponding /lscript -step lookahead policy will be stable for all /lscript larger than some threshold ; see Fig. 1.5.8. The book [Ber22a], Section 3.3, contains a broader discussion of the region of stability and the role of multistep lookahead in enhancing it; see also Exercises 1.8 and 1.9.

## Newton Step Interpretation of Approximation in Value Space in General Infinite Horizon Problems

The interpretation of approximation in value space as a Newton step, and related notions of stability that we have discussed in this section admit a broad generalization to the infinite horizon problems that we consider in this book and beyond. The key fact in this respect is that our DP problem formulation allows arbitrary state and control spaces, both discrete and continuous, and can be extended even further to general abstract models with a DP structure; see the abstract DP book [Ber22b].

Within this context, the Riccati operator is replaced by an abstract Bellman operator, and valuable insight can be obtained from graphical interpretations of the Bellman equation, the VI and PI algorithms, onestep and multistep approximation in value space, the region of stability, and exceptional behavior; see the book [Ber22a] for an extensive discussion. Naturally, the graphical interpretations and visualizations are limited to one dimension. However, they provide insight, and motivate conjectures and mathematical analysis, much of which is given in the book [Ber20a].

## The Importance of the First Step in Multistep Lookahead

The Newton step interpretation of approximation in value space leads to an important insight into the special character of the initial step in /lscript -step lookahead implementations. In particular, it is only the first step that acts as the Newton step , and needs to be implemented with precision; cf. Fig. 1.5.5. The subsequent /lscript -1 steps are a sequence of value iterations starting with ˜ J , and only serve to enhance the quality of the starting point of the Newton step. As a result, their precise implementation is not critical , a major point in the narrative of the author's book [Ber22a].

This idea suggests that we can simplify (within reason) the lookahead steps after the first with small (if any) performance loss for the multistep lookahead policy. An important example of such a simplification is the use of certainty equivalence, which will be discussed later in various contexts (Sections 1.6.9, 2.7.2, 2.8.3). Other possibilities include the 'pruning' of the lookahead tree after the first step; see Section 2.4. In practical terms, simplifications after the first step of the multistep lookahead can save a lot of on-line computation, which can be fruitfully invested in extending the length of the lookahead. This insight is supported by substantial computational experimentation, starting with the paper by Bertsekas and Casta˜ non [BeC99], which verified the beneficial e ff ect of using certainty equivalence after the first step.

## 1.5.2 Rollout and Policy Iteration

We will now consider the rollout algorithm for the linear quadratic problem, starting from a linear stable base policy θ . It generates the rollout policy

Figure 1.5.10 Illustration of the rollout algorithm for the linear quadratic problem. Starting from a linear stable base policy θ , it generates a stable rollout policy ˜ θ . The quadratic cost coe ffi cient of ˜ θ is obtained from the quadratic cost coe ffi cient of θ with a Newton step for solving the Riccati equation.

<!-- image -->

˜ θ by using a policy improvement operation, which by definition, yields the one-step lookahead policy that corresponds to terminal cost approximation ˜ J = J θ . Figure 1.5.10 illustrates the rollout algorithm. It can be seen from the figure that the rollout policy is in fact an improved policy, in the sense that J ˜ θ ( x ) ≤ J θ ( x ) for all x . Among others, this implies that the rollout policy is stable, since θ is assumed stable so that J θ ( x ) &lt; ∞ for all x .

Since the rollout policy is a one-step lookahead policy, it can also be described using the formulas that we developed earlier in this section. In particular, let the base policy have the form

<!-- formula-not-decoded -->

where L 0 is a scalar. We require that the base policy must be stable, i.e., ♣ a + bL 0 ♣ &lt; 1. From our earlier calculations, we have that the cost function of θ 0 is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

Moreover, the rollout policy θ 1 has the form θ 1 ( x ) = L 1 x↪ where

<!-- formula-not-decoded -->