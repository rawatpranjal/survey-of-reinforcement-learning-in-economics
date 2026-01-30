## Convergence of Stochastic Iterative Dynamic Programming Algorithms

Tommi Jaakkola'" Michael I.  Jordan Satinder P.  Singh

Department of Brain and Cognitive Sciences Massachusetts  Institute of Technology Cambridge, MA  02139

## Abstract

Increasing attention has recently been  paid to algorithms based on dynamic programming (DP) due to the suitability of DP for learn› ing  problems involving control. In stochastic  environments  where the system being  controlled is  only incompletely known,  however, a  unifying theoretical account  of these  methods has  been  missing. In this  paper  we  relate  DP-based learning algorithms to the  pow› erful techniques of stochastic approximation via a new convergence theorem,  enabling us  to establish  a  class of convergent  algorithms to which both TD("\)  and Q-Iearning belong.

## 1 INTRODUCTION

Learning to predict  the future  and to find  an optimal way of controlling it  are  the basic  goals of learning systems  that  interact  with  their  environment.  A  variety of algorithms are  currently  being  studied  for  the  purposes  of prediction  and  control in  incompletely specified,  stochastic environments.  Here  we  consider learning algo› rithms defined in Markov environments.  There are actions or controls (u) available for  the  learner  that  affect  both  the  state  transition  probabilities,  and  the  proba› bility distribution for  the immediate, state dependent costs (Ci( u)) incurred by the learner. Let Pij  (u) denote  the  probability of a  transition  to  state j when  control u is  executed  in state i. The learning problem is  to predict  the expected  cost  of a

...  E-mail:  tommi@psyche.mit.edu

fixed  policy  p  (a function  from states  to  actions),  or  to obtain the  optimal policy (p*) that minimizes the expected cost  of interacting with the environment.

If the learner were  allowed to know the transition probabilities as well as the imme› diate  costs  the control  problem could  be solved directly  by  Dynamic Programming (see  e.g.,  Bertsekas,  1987). However,  when  the  underlying  system  is  only  incom› pletely  known,  algorithms such  as  Q-Iearning  (Watkins,  1989)  for  prediction  and control,  and  TD(&gt;.)  (Sutton,  1988) for  prediction,  are needed.

One  of the  central  problems  in  developing  a  theoretical  understanding  of these algorithms  is  to  characterize  their  convergence;  that  is,  to  establish  under  what conditions they are  ultimately able to obtain correct  predictions or optimal control policies. The stochastic  nature  of these  algorithms immediately suggests  the  use of stochastic  approximation  theory  to  obtain  the  convergence  results. However, there exists  no  directly  available stochastic  approximation techniques for  problems involving the maximum norm that plays a crucial role in learning algorithms based on  DP.

In  this  paper,  we  extend  Dvoretzky's (1956)  formulation of the  classical  Robbins› Munro (1951) stochastic approximation theory to obtain a  class of converging pro› cesses  involving  the  maximum norm. In  addition,  we  show  that  Q-Iearning  and both  the  on-line  and  batch  versions  of TD(&gt;.)  are  realizations  of  this  new  class. This approach keeps  the convergence  proofs simple and does  not rely on  construc› tions specific to particular algorithms.  Several other authors have recently presented results  that  are  similar to  those  presented  here: Dayan and  Sejnowski  (1993)  for TD(A),  Peng and Williams (1993)  for  TD(A),  and Tsitsiklis (1993)  for  Q-Iearning. Our results  appear  to be closest  to those  of Tsitsiklis (1993).

## 2 Q-LEARNING

The Q-Iearning  algorithm produces  values-"Q-values"-by which  an  optimal ac› tion  can  be  determined  at  any  state.  The  algorithm is  based  on  DP  by  rewriting Bellman's  equation  such  that  there  is  a  value  assigned  to  every  state-action  pair instead of only to a  state.  Thus the Q-values satisfy

<!-- formula-not-decoded -->

where  c denotes  the  mean  of c. The  solution  to  this  equation  can  be  obtained by  updating  the  Q-values  iteratively;  an  approach  known  as  the vaz'ue  iteration method.  In the learning problem the values for  the mean of c and for the transition probabilities are  unknown.  However,  the observable quantity

<!-- formula-not-decoded -->

where St and Ut are the state of the system and the action taken at time t, respec› tively, is an unbiased estimate of the update used in value iteration.  The Q-Iearning algorithm is  a  relaxation  method that  uses  this  estimate iteratively to  update  the current  Q-values (see  below).

The Q-Iearning algorithm converges  mainly due to the contraction property  of the value iteration operator.

## 2.1 CONVERGENCE OF Q-LEARNING

Our proof is based on the observation that the Q-Iearning algorithm can be viewed as a  stochastic process  to which  techniques  of stochastic  approximation are generally applicable. Due  to  the  lack  of a  formulation  of stochastic  approximation for  the maximum norm, however,  we  need  to slightly extend  the standard results.  This is accomplished by the following theorem the proof of which can be found in Jaakkola et  al.  (1993).

Theorem 1 A  random iterative process ~n+I(X) = (l-ll:n(X))~n(x)+lin(x)Fn(x) converges to  zero  w.p.l  under the  following  assumptions:

- 1) The  state  space  is  finite.
- 2)  Ln ll:n(x) = 00, Ln ll:~(x) &lt; 00, Ln lin(x) = 00, Ln Ii~(x) &lt; 00, and E{lin(x)IPn} ~ E{ll:n(x)IPn}  uniformly w.p.1.
- 3) II E{Fn(x)IPn} Ilw~ 'Y II ~n IlwI where'Y E (0,1).
- 4)  Var{Fn(x)IPn} ~ C(1+ II ~n Ilw)2, where C is some  constant.

Here Pn = {~n,  ~n-I, .. ·' Fn- I, ... , ll:n-I,· .. , lin-I, ...  }  stands for the past  at  step n. Fn(x),  ll:n(x)  and lin(x)  are  allowed  to  depend  on  the  past  insofar  as  the  above conditions  remain  valid. The  notation II . Ilw refers  to  some  weighted  maximum norm.

In  applying  the  theorem,  the ~n process  will  generally  represent  the  difference between  a  stochastic  process  of interest  and some optimal value  (e.g.,  the  optimal value function).  The formulation of the theorem therefore  requires knowledge to be available about the optimal solution to the learning problem before it can be applied to any algorithm whose convergence is  to be  verified.  In  the case  of Q-Iearning the required  knowledge  is  available through  the  theory  of DP  and  Bellman's equation in  particular.

The  convergence  of  the  Q-Iearning  algorithm  now  follows  easily  by  relating  the algorithm to the converging stochastic process  defined  by Theorem  1.1

Theorem 2 The  Q-learning  algorithm  given  by

<!-- formula-not-decoded -->

converges  to  the  optimal Q*(s, u)  values  if

- 1)  The  state  and  action  spaces  are  finite.
- 2)  Lt ll:t(s, u) =  00 and Lt ll:;(s, u) &lt; 00 uniformly w.p.1.
- 3)  Var{cs(u)} is  bounded.

1 We  note that  the theorem  is  more  powerful  than is  needed  to prove  the convergence of Q-learning.  Its generality,  however,  allows  it  to be applied  to other  algorithms  as  well (see  the following  section  on TD(&gt;.)).

- 3) If, =  1, all policies  lead  to  a  cost free  terminal state  w.p.1.

Proof. By subtracting Q*(s, u) from both sides of the learning rule and by defining Llt(s, u) = Qt(s, u) -Q*(s, u) together with

<!-- formula-not-decoded -->

the Q-learning algorithm can be seen  to have the form of the process  in Theorem 1 with !3t(s, u) = at(s, u).

To verify  that Ft(s, u) has the required  properties  we  begin by showing that it is  a contraction mapping with respect  to some maximum norm.  This is done by relating Ft to the DP value iteration operator for  the same Markov chain.  More specifically,

<!-- formula-not-decoded -->

where we have used the notation Va(j) =  maXv IQt(j, v)-Q*(j, v)1 and T is  the DP value iteration operator for  the case  where  the costs  associated  with each state are zero.  If, &lt; 1 the contraction property of E{ F t (i, u)} can be obtained by bounding I: j Pij(U)Va(j) by  maxj Va(j) and then including the, factor.  When the future costs  are  not  discounted  (, =  1)  but the chain is  absorbing and all policies lead to the  terminal state  w.p.1  there  still exists  a  weighted  maximum norm with  respect to which T is  a  contraction mapping (see  e.g.  Bertsekas &amp; Tsitsiklis,  1989) thereby forcing  the  contraction  of E{Ft(i, u)}. The  variance  of Ft(s, u) given  the  past  is within the bounds of Theorem 1 as  it depends  on Qt(s, u) at most linearly and the variance of cs(u) is  bounded.

Note that the  proof covers  both the on-line  and batch versions. o

## 3 THE TD(-\)  ALGORITHM

The TD(A)  (Sutton,  1988)  is  also  a  DP-based learning algorithm that is  naturally defined in a Markov environment.  Unlike Q-learning, however, TD does not involve decision-making tasks but rather  predictions  about  the future  costs  of an evolving system.  TD(A) converges to the same predictions as a version ofQ-learning in which there is only one action available at each state, but the algorithms are derived from slightly  different  grounds  and their  behavioral differences  are not well  understood.

The algorithm is  based on the estimates

<!-- formula-not-decoded -->

where ~(n)(i) are n step  look-ahead predictions.  The expected  values of the ~&gt;"(i) are  strictly  better  estimates  of  the  correct  predictions  than  the lit (i)s  are  (see

Jaakkola et al.,  1993)  and the update equation of the algorithm

<!-- formula-not-decoded -->

can  be  written  in  a  practical  recursive  form  as  is  seen  below.  The convergence  of the algorithm is  mainly due to the statistical properties  of the V? (i) estimates.

## 3.1 CONVERGENCE OF TDP)

As  we  are  interested  in strong forms  of convergence  we  need  to  impose some  new constraints,  but  due  to  the  generality  of the  approach  we  can  dispense  with  some others. Specifically,  the  learning  rate  parameters an are  replaced  by a n( i)  which satisfy Ln an(i) = 00  and Ln a~(i) &lt;  00  uniformly  w.p.1. These  parameters allow  asynchronous  updating and they can,  in  general,  be  random variables.  The convergence  of the  algorithm is  guaranteed  by  the  following  theorem  which  is  an application of Theorem 1.

Theorem 3 For any finite  absorbing Markov chain,  for any distribution  of  starting states  with  no  inaccessible  states,  and for  any  distributions  of the  costs  with  finite variances  the  TD(A)  algorithm  given  by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lt at(i) = 00 and  Ln a;(i) &lt;  00 uniformly  w.p.i  and  within  sequences at(i)/maXtESat(i) ----;.  1 uniformly w.p.i.

converges  to  the  optimal predictions  w.p.i  provided" A E [0,1] with ,A &lt; 1.

Proof for  (1):  We  use  here  a  slightly different  form for  the  learning rule  (cf.  the previous section).

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where V n "( i; k) is  an  estimate  calculated  at  the ph occurrence  of  state  i  in  a sequence  and  for  mathematical  convenience  we have  made  the  transformation an(i) ----;. E{m(i)}an(i), where m(i) is  the  number  of times  state i was  visited during the sequence.

To apply Theorem 1 we  subtract V* (i), the optimal predictions, from both sides of the  learning equation.  By  identifying an(i) := an(i)m(i)/E{m(i)}, f3n(i) := an(i), and Fn(i) := Gn(i) -V*(i)m(i)/E{m(i)} we  need  to  show  that  these  satisfy  the conditions  of Theorem  1. For an(i) and f3n(i) this  is  obvious. We  begin  here  by showing that Fn(i) indeed is  a  contraction mapping.  To this end,

<!-- formula-not-decoded -->

which can be  bounded  above by  using the relation

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where O(x) = 0 if x &lt; 0 and 1 otherwise.  Here we  have also used  the fact  that VnA(i) is  a  contraction  mapping independent  of possible  discounting. As Lk P {m(  i) ~ k} = E{ m( i)} we  finally  get

<!-- formula-not-decoded -->

The variance of Fn (i) can be seen  to be bounded by

<!-- formula-not-decoded -->

For any absorbing Markov chain the convergence  to the terminal state is geometric and thus for  every finite k,  E{mk}::; C(k), implying that the  variance of Fn(i) is within the bounds of Theorem 1.  As Theorem 1 is  now applicable we  can conclude that the batch version  of TD(&gt;.)  converges  to the optimal predictions w.p.l. 0

Proof for  (2) The  proof for  the  on-line  version  is  achieved  by  showing  that  the effect  of the on-line updating vanishes  in  the limit thereby forcing  the two versions to  be  equal  asymptotically. We  view  the  on-line  version  as  a  batch  algorithm in which  the  updates are  made after each  complete sequence  but  are  made in such  a manner so as to be equal to those  made on-line.

Define GLYPH(cmap:df00)~ (i) = G n (i) + GLYPH(cmap:df00)~ (i)  to  be  a  new  batch estimate taking into account  the on-line updating within sequences.  Here Gn (i) is the batch estimate with the desired properties  (see  the  proof for  (1» and GLYPH(cmap:df00)~ (i) is  the  difference  between  the two.  We take  the  new  batch  learning  parameters  to  be  the  maxima over  a  sequence,  that is an(i) = maxtES at(i). As  all  the at(i) satisfy  the  required  conditions  uniformly w.p.1  these new  learning parameters satisfy them as  well.

To analyze  the new  batch algorithm we  divide  it into three  parallel processes:  the batch TD(  &gt;.)  with an (i) as  learning rate parameters, the difference between this and the  new  batch estimate,  and  the  change  in  the  value function  due  to the  updates made  on-line. Under  the  conditions  of the  TD(&gt;.)  convergence  theorem  rigorous

upper  bounds  can  be  derived  for  the  latter  two  processes  (see  Jaakkola,  et  al., 1993).  These results  enable  us  to write

<!-- formula-not-decoded -->

where C~ and C~ go  to  zero  with  w.p.I. This  implies  that  for  any  c &gt; 0  and II Vn -V* II~ c there  exists I &lt; 1 such that

<!-- formula-not-decoded -->

for n large  enough. This  is  the  required  contraction  property  of Theorem  1. In addition, it can readily be checked that the variance of the new estimate falls under the conditions of Theorem 1.

Theorem 1 now guarantees that for any c the value function in the on-line algorithm converges w.p.1 into some t-bounded region of V* and therefore the algorithm itself converges  to V* w.p.I. 0

## 4 CONCLUSIONS

In  this  paper  we  have  extended  results  from  stochastic  approximation  theory  to cover  asynchronous  relaxation  processes  which  have  a  contraction  property  with respect to some maximum norm (Theorem 1).  This new class of converging iterative processes  is  shown  to include  both the Q-Iearning and TD(A)  algorithms in either their on-line or batch versions.  We  note that the convergence  of the on-line version of TD(A) has not been shown previously.  We  also wish to emphasize the simplicity of our results.  The convergence  proofs for  Q-Iearning and TD(A)  utilize  only high› level statistical properties of the estimates used in these  algorithms and do not rely on  constructions  specific  to  the  algorithms. Our  approach  also  sheds  additional light on the similarities between  Q-Iearning and TD(A).

Although Theorem 1 is readily applicable to DP-based learning schemes, the theory of Dynamic Programming is  important only for  its characterization of the optimal solution and for  a contraction property needed  in applying the theorem.  The theo› rem can be  applied to iterative algorithms of different  types  as well.

Finally we note that Theorem 1 can be extended to cover processes that do not show the usual contraction property thereby  increasing its applicability to algorithms of possibly more practical importance.

## References

Bertsekas, D. P. (1987). Dynamic Programming:  Deterministic and Stochastic Mod› els. Englewood Cliffs,  NJ:  Prentice-Hall.

Bertsekas,  D. P., &amp; Tsitsiklis, J.  N.  (1989). Parallel and Distributed  Computation: Numerical Methods. Englewood Cliffs,  NJ:  Prentice-Hall.

Dayan,  P.  (1992).  The convergence  of TD(A) for  general  A. Machine  Learning,  8, 341-362.

Dayan,  P., &amp; Sejnowski,  T.  J.  (1993). TD(&gt;.)  converges  with  probability 1.  CNL, The Salk Institute, San Diego, CA.

Dvoretzky, A. (1956).  On stochastic approximation. Proceedings of  the Third Berke› ley  Symposium  on  Mathematical Statistics  and Probability. University of California Press.

Jaakkola, T., Jordan, M.  I., &amp; Singh, S.  P.  (1993).  On the convergence of stochastic iterative dynamic programming algorithms.  Submitted to Neural  Computation.

Peng J., &amp; Williams R. J. (1993).  TD(&gt;.) converges with probability 1.  Department of Computer Science  preprint,  Northeastern  University.

Robbins,  H., &amp; Monro,  S.  (1951). A  stochastic  approximation model. Annals  of Mathematical Statistics, 22,  400-407.

Sutton,  R.  S.  (1988).  Learning  to  predict  by  the  methods  of temporal  differences. Machine  Learning, 3,  9-44.

Tsitsiklis  J.  N.  (1993). Asynchronous  stochastic  approximation  and  Q-learning. Submitted to: Machine  Learning.

Watkins, C.J .C.H.  (1989). Learning from  delayed  rewards. PhD Thesis,  University of Cambridge, England.

Watkins, C.J .C.H, &amp; Dayan, P.  (1992).  Q-learning. Machine  Learning, 8,  279-292.