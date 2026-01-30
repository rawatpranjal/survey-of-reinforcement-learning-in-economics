## Mathematics Department, Princeton University

An Iterative Method of Solving a Game

Author(s): Julia Robinson

Source: Annals of Mathematics, Second Series, Vol. 54, No. 2 (Sep., 1951), pp. 296-301

Published by: Mathematics Department, Princeton University

Stable URL: https://www.jstor.org/stable/1969530

Accessed: 18-01-2020 15:09 UTC

JSTOR is a not-for-profit service that helps scholars, researchers, and students discover, use, and build upon a wide range of content in a trusted digital archive. We use information technology and tools to increase productivity and facilitate new forms of scholarship. For more information about JSTOR, please contact support@jstor.org.

Your use of the JSTOR archive indicates your acceptance of the Terms &amp; Conditions of Use, available at https://about.jstor.org/terms

<!-- image -->

Mathematics Department, Princeton University is collaborating with JSTOR to digitize, preserve and extend access to Annals of Mathematics

## AN ITERATIVE METHOD OF SOLVING A GAME

## BY JULIA ROBINSON

(July 28, 1950)

A two-person game' can be represented by its pay-off matrix A = (ai first player chooses one of the m rows and the second player simultaneously chooses one of the n columns. If the ith row and the jth column are chosen, the second player pays the first player aij.

If the first player plays the ith row with probability xi and the second player plays the jth column with probability yj , where xi \_ 0, EZx = 1, yj &gt; 0, and Eyj = 1, then the expectation of the first player is Zaijxiyj. Furthermore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The minimax theorem of game theory (see [1] page 153) asserts that for some set of probabilities X = (x1 , ***, X,) and Y = (yi , .**, yn) the equality holds in (1). Such a pair (X, Y) is called a solution of the game. The value v of the game is defined by

<!-- formula-not-decoded -->

where (X, Y) is a solution of the game.

In this paper, we shall show the validity of an iterative procedure suggested by George W. Brown [2]. This method corresponds to each player choosing in turn the best pure strategy against the accumulated mixed strategy of his opponent up to then.

Let A = (aij) be an m X n matrix. Ai. will denote the ith row of A and A.j, the jth column. Similarly, if V(t) is a vector, then vj(t) is the jth component. Let max V(t) = maxj vj(t) and min V(t) = mini vj(t). In this notation, (1) can be rewritten as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

whenever xi\_ 0, Exi = 1, y, \_ 0, and Zyj = 1.

DEFINITION 1. A system (U, V) consisting of a sequence of n-dimensional vectors U(O), U(1), * * * and a sequence of m-dimensional vectors V(O), V(1), * is called a vector system for A provided that

<!-- formula-not-decoded -->

1 More technically, a finite two-person zero-sum game. See [1] in the bibliography at end of the paper.

and

<!-- formula-not-decoded -->

where i and j satisfy the conditions

<!-- formula-not-decoded -->

Thus a vector system for A can be formed recursively from a given U(0) and V(0). At each step, the row added to U is determined by a maximum component of V and the column added to V is determined by a minimum component of U.

An alternate notion of vector system is obtained if the condition on j in Definition 1 is replaced by

<!-- formula-not-decoded -->

A vector system of this new type can also be built up recursively. The only difference is that here successive U and V are determined alternately while in the other definition U and V could be obtained simultaneously. In all the following proofs and theorems, either definition may be used.

In the special case U(0) = 0 and V(0) = 0, we see that U(t)/t is a weighted average of the rows of A and V(t)/t is a weighted average of the columns. Hence for every t and t',

<!-- formula-not-decoded -->

If for some t and t', these two bounds are equal, we have a solution of the game. Unfortunately, this is not always the case. However George Brown [2] conjectured that as t and l' tend to o, the two bounds approach v. The main result of this paper is to prove this for any vector system. In numerical examples, vector systems of the second kind appear to converge more rapidly than the first.

THEOREM.' If (U, V) is a vector system for A, then

<!-- formula-not-decoded -->

The proof will be divided into four lemmas.

LeMMa 1. If (U, V) is a vector system for a matrix A, then

<!-- formula-not-decoded -->

PRooF. For each t,

<!-- formula-not-decoded -->

^ The solution to Problem 5 in the RAND Mathematical Problem Series II is contained as a special case of this theorem.

and

Hence and

<!-- formula-not-decoded -->

Therefore, and

and and

where

Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

DEFINITION 2. If (U, V) is a vector system for A, then we say that the ita row is eligible in the interval (t, t) provided that there exists t with

<!-- formula-not-decoded -->

Similarly, the jth column is eligible in the interval (t, t') if there exists t with

<!-- formula-not-decoded -->

LEMMa 2. Given a vector system (U, V) for A, then if all the rows and columns of A are eligible in the interval (s, 8 + t),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

ProoF. Let j be such that

<!-- formula-not-decoded -->

Choose t' with s ≤ t ≤s + t so that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

since the change in the ith component in t steps is not more than at. But

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 3. If all the rows and columns of A are eligible in (s, s + t) for a given vector system (U, V), then

<!-- formula-not-decoded -->

PRooF. By Lemma 2,

<!-- formula-not-decoded -->

Hence it is sufficient to show that min V(s + t) ≤ max U(s + t). Now applying

(2) to the transpose of A, we have

whenever xi ≥ 0, Ex; = 1, yj ≥ 0, and [yj = 1. In particular, choose xi

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then

<!-- formula-not-decoded -->

LeMMa 4. To every matrix A and e &gt; 0, there exists to such that for any vector system (U, V),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

ProoF. The theorem holds for matrices of order 1 since U(t) = Vt) for all t. Assume the theorem holds for all submatrices of A, then we will show by induction that it holds for A. Choose t* so that for any vector system (U', V') corresponding to a submatrix A' of A, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We shall prove that if in the given vector system (U, V) for A, some row or column is not eligible in the interval (s, s + f*), then

<!-- formula-not-decoded -->

so that

Similarly, and y satisfying

Suppose, for example, that the lith row is not eligible in the interval (s, s + f*). Then we can construct a vector system (U', V') for the matrix A' obtained by deleting the lth row of A, in the following way:

<!-- formula-not-decoded -->

where C' is the n-dimensional vector all of whose components are equal to max V(s) - min U(s) and Proj.V is the vector obtained from V by omitting the kith component. The rows of A' will be numbered 1,2, ..., k - 1, k + 1. ... , m. Notice first that min U'(0) = max V'(0). Furthermore, if

<!-- formula-not-decoded -->

then

Hence max V(s + f*) - min U(s + f*)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We can now show that given any vector system (U, V) for A,

<!-- formula-not-decoded -->

Consider t &gt; t* . Let o with 0 ≤ 0 &lt; 1 and q a positive integer be so chosen that t = (0 + 9)t*.

CasE 1. Suppose there is a positive integers ≤ q so that all rows and columns of A are eligible in the interval ((0 + s - 1)t* , (0 + s)*). Take the largest such s, then

<!-- formula-not-decoded -->

We obtain this inequality by repeated application of (3), since in each of the intervals

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Also vi(s + t) = max V(s + t) if and only if v'(t) = max V'(t) and u,(s + t) = min U(s + t) if and only if ust) = min '(t) for 0 ≤ t ≤ **. Hence we see that U' and V' must satisfy the recursive restrictions of the definition of a vector system for 0 ≤ t ≤ t*, since U and V do. Naturally, we may continue

U' and V' indefinitely to form a vector system for A'.

Now by the choice of t*, we know that

<!-- formula-not-decoded -->

some row or column of A is not eligible. From Lemma 3 and the choice of s, we have

<!-- formula-not-decoded -->

From (4) and (5), we obtain

<!-- formula-not-decoded -->

CASE 2. If there is no such s, then in each interval ((0 + r - 1)t* , (O + r)t*) some row or column of A is not eligible. Hence

<!-- formula-not-decoded -->

Therefore, in either case,

<!-- formula-not-decoded -->

From Lemmas 1 and 4, we see that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which completes the proof of the theorem.

THE RAND CORPORATION,

SANTA MONICA, CALIFORNIA

But from (1),

Hence

## BIBLIOGRAPHY

- [1] J. VON NEUMANN and 0. MORGENSTERN, Theory of Games and Economic Behavior, Princeton University Press.
- [2] G. W. BROWN, Some notes on computation of Games Solutions, RAND Report P-78, April 1949, The RAND Corporation, Santa Monica, California.