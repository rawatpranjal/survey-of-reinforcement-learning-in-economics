## Actor-Critic Algorithms

Vijay R. Konda John N. Tsitsiklis Laboratory for  Information and Decision Systems,

Massachusetts Institute of Technology, Cambridge, MA,  02139. konda@mit.edu,  jnt@mit.edu

## Abstract

We  propose  and  analyze  a  class  of  actor-critic  algorithms  for simulation-based  optimization  of a  Markov  decision  process  over a  parameterized family  of randomized  stationary policies. These are two-time-scale algorithms in which  the critic uses TD learning with a linear approximation architecture and the actor is  updated in  an  approximate  gradient  direction  based  on  information  pro› vided by the critic.  We show that the features for  the critic should span a subspace prescribed by the choice of parameterization of the actor.  We conclude by discussing convergence properties and some open problems.

## 1 Introduction

The vast  majority of Reinforcement  Learning  (RL)  [9J  and  Neuro-Dynamic  Pro› gramming (NDP)  [lJ  methods fall  into one of the following two categories:

- (a)  Actor-only methods work with a parameterized family of policies.  The gra› dient of the performance, with respect to the actor parameters, is  directly estimated by simulation, and the parameters are updated in  a  direction of improvement [4,  5,  8,  13J.  A possible drawback of such methods is that the gradient estimators  may  have  a  large  variance. Furthermore,  as  the  pol› icy  changes,  a  new  gradient  is  estimated independently of past estimates. Hence,  there is  no  "learning,"  in  the sense of accumulation and consolida› tion of older information.
- (b)  Critic-only methods rely exclusively on  value  function  approximation  and aim at learning an approximate solution to the Bellman equation, which will then  hopefully  prescribe a  near-optimal policy.  Such  methods are indirect in the sense that they do not try to optimize directly over a policy space.  A method of this type may succeed in constructing a "good" approximation of the value function,  yet lack reliable guarantees in terms of near-optimality of the resulting policy.

Actor-critic methods aim  at combining the strong points of actor-only and critic› only  methods. The  critic  uses  an  approximation  architecture  and  simulation  to learn  a  value function,  which  is  then used to update the actor's policy parameters

in  a  direction  of  performance  improvement. Such  methods,  as  long  as  they  are gradient-based,  may  have  desirable  convergence  properties,  in  contrast  to  critic› only  methods for  which  convergence is  guaranteed in  very  limited  settings.  They hold the promise of delivering faster convergence (due to variance reduction), when compared to actor-only methods.  On the other hand, theoretical understanding of actor-critic methods has been limited to the case of lookup table representations of policies  [6].

In this paper, we  propose some actor-critic algorithms and provide an overview of a convergence proof.  The algorithms are based on an important observation.  Since the number of parameters that the actor has to update is relatively small (compared to the  number of states),  the critic need  not  attempt to compute or approximate the exact value function,  which is  a high-dimensional object. In fact,  we  show that the critic should ideally compute a certain  "projection" of the value function onto a low-dimensional subspace spanned by a set of "basis functions,"  that are completely determined by  the  parameterization of the  actor. Finally,  as  the  analysis  in  [11] suggests for TD algorithms, our algorithms can be extended to the case of arbitrary state and action spaces as long as certain ergodicity assumptions are satisfied.

We  close  this  section  by  noting that ideas  similar to ours  have been  presented in the simultaneous and independent work of Sutton et al.  [10].

## 2 Markov decision processes and parameterized family of RSP's

Consider a Markov decision process with finite state space S, and finite action space A. Let 9 : S x A  -t ffi.  be a given cost function.  A randomized stationary policy (RSP) is  a mapping I-" that assigns to each state x a probability distribution over the action space A. We  consider  a  set  of randomized  stationary  policies  JPl = {1-"9; e E ffi. n }, parameterized in terms of a vector e. For each pair (x, u) E S x A, 1-"9 (x, u) denotes the probability of taking action u when the state x is  encountered, under the policy corresponding to e. Let PXy(u) denote the probability that the next state is y, given that the current state is x and the current action is u. Note that under any RSP, the sequence of states {Xn} and of state-action pairs {Xn' Un} of the Markov decision process form  Markov chains with state spaces Sand S x A, respectively.  We  make the following  assumptions about the family  of policies JPl.

- (AI)  For  all xES and u E A the  map e t-t 1-"9(X, u) is  twice  differentiable with  bounded  first,  second  derivatives. Furthermore,  there  exists  a ffi.n\_ valued function 'l/J9(X, u) such that \l1-"9(X, u) = 1-"9 (x, U)'l/J9(X, u) where the mapping e t-t 'l/J9(X, u) is  bounded and has first  bounded derivatives for any fixed x and u.
- (A2)  For each e E ffi. n , the Markov chains {Xn} and {Xn, Un} are irreducible and aperiodic, with stationary probabilities 7r9(X) and 'T}9(X, u) = 7r9 (x) 1-"9 (x, u), respectively, under the RSP 1-"9.

In reference to Assumption  (AI) , note that whenever 1-"9 (x, u) is  nonzero we  have

<!-- formula-not-decoded -->

Consider the average cost function&gt;. : ffi.n t-t  ffi.,  given by

<!-- formula-not-decoded -->

We  are interested in  minimizing  &gt;'(19)  over all  19. For each 19  E R n , let Ve : S t--7  R be the  "differential"  cost function,  defined as  solution of Poisson equation:

<!-- formula-not-decoded -->

Intuitively, Ve(x) can be viewed as the "disadvantage" of state x: it is the expected excess cost on top of the average cost incurred if we start at state x. It plays a role  similar to that played by the more familiar  value function  that arises  in total or discounted cost Markov decision problems.  Finally, for  every 19 E Rn, we define the q-function qe : S x A -+ R,  by

<!-- formula-not-decoded -->

We recall the following result, as stated in [8]. (Different versions of this result have been established in  [3,  4,  5].)

Theorem 1.

Since

<!-- formula-not-decoded -->

where 1/;b (x, u)  stands for  the i th  component of 1/;e .

In  [8],  the  quantity qe(x,u) in  the  above  formula  is  interpreted  as  the  expected excess  cost  incurred over a  certain renewal  period  of the Markov chain {Xn, Un}, under the RSP I-'e, and is then estimated by means of simulation, leading to actor› only  algorithms. Here,  we  provide  an  alternative interpretation  of the  formula  in Theorem 1, as an inner product, and thus derive a different set of algorithms, which readily generalize to the case of an infinite space as well.

For any 19  E Rn , we define the inner product (', .) e of two real valued functions q1 , q2 on S x A, viewed as vectors in RlsiIAI, by

<!-- formula-not-decoded -->

With this notation we  can rewrite the formula (1) as

<!-- formula-not-decoded -->

Let 11 ·lle  denote the norm induced by this inner product on RlsiIAI. For each 19  E Rn let we denote the span of the vectors {1/;b; 1 ::;  i  ::; n} in RISIIAI. (This is  same as the set of all functions f on S x A of the form f(x ,u) = 2::7=1 ai1/;~(x , U), for  some scalars a1,· . . ,an,)

Note that although the gradient of &gt;. depends on the q-function,  which  is  a  vector in  a  possibly  very  high  dimensional  space RlsiIAI, the dependence is  only through its inner products with vectors in we. Thus, instead of "learning"  the function qe, it would suffice to learn the projection of qe on the subspace We.

Indeed, let rIe  : RlsllAI t--7 We be the projection operator defined by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

it is  enough to compute the projection of qe onto we.

## 3 Actor-critic algorithms

We view actor critic-algorithms as stochastic gradient algorithms on the parameter space of the actor.  When the actor parameter vector  is 0, the job of the critic is to compute an approximation of the projection IIeqe of qe onto 'lie. The actor uses this approximation to update its policy  in  an approximate gradient direction.  The analysis in  [11,  12]  shows that this is  precisely  what TD algorithms try to do,  i.e., to  compute the projection of an  exact value function  onto a  subspace spanned by feature  vectors. This allows  us  to implement the critic  by  using  a  TD  algorithm. (Note, however, that other types of critics are possible, e.g., based on batch solution of least squares problems, as long as they aim at computing the same projection.)

We  note  some  minor  differences  with  the  common  usage  of TD.  In  our  context, we  need  the  projection  of  q-functions,  rather  than  value  functions. But this  is easily achieved by replacing the Markov chain {xt} in  [11,  12]  by the Markov chain {Xn, Un}. A further  difference  is  that  [11,  12]  assume that the control policy  and the feature  vectors  are fixed. In  our algorithms,  the  control  policy  as  well  as  the features  need  to change  as  the  actor  updates  its  parameters. As  shown  in  [6,  2], this need not pose any problems, as long as the actor parameters are updated on a slower time scale.

We  are  now  ready  to describe two  actor-critic algorithms, which  differ  only as  far as  the critic updates are concerned.  In both variants, the critic is  a  TD algorithm with a linearly parameterized approximation architecture for  the q-function, of the form

<!-- formula-not-decoded -->

where r = (rl, ... , rm) E ]Rm denotes  the  parameter  vector  of  the  critic. The features 4&gt;~, j  = 1, ... ,m, used by the critic are dependent on the actor parameter vector 0 and  are  chosen  such  that their  span  in ]RlsIIAI, denoted  by &lt;Pe, contains 'lI e. Note that the formula  (2)  still  holds  if IIe is  redefined  as  projection  onto &lt;Pe as long as &lt;Pe contains 'lie. The most straightforward choice would be to let m = n and 4&gt;~ = 't/J~ for  each i.  Nevertheless,  we  allow the possibility that m &gt; nand &lt;Pe properly contains 'lie, so  that  the critic uses  more features  than  that are actually necessary.  This added flexibility  may turn out to be useful  in a number of ways:

1. It is  possible  for  certain  values  of 0, the  features 't/Je are  either  close  to zero or are almost linearly dependent.  For these values  of 0, the operator IIe becomes ill-conditioned and the algorithms can become unstable.  This might be avoided by using richer set of features 't/J~.
2.  For  the  second  algorithm that we  propose  (TD(a)  a &lt; 1)  critic  can  only compute approximate - rather than exact - projection.  The use of additional features can result in  a reduction of the approximation error.

Along with the parameter vector r, the critic stores some auxiliary parameters: these are a  (scalar)  estimate A,  of the average cost,  and an  m-vector z which  represents Sutton's eligibility trace [1, 9].  The actor and critic updates take place in the course of a simulation of a single sample path of the controlled Markov chain.  Let rk, Zk, Ak be the parameters of the critic, and let Ok be the parameter vectpr of the actor, at time k. Let (Xk, Uk) be the state-action pair at that time.  Let Xk+l be the new state, obtained after action Uk is applied.  A new action Uk+l is generated according to the RSP corresponding to the actor parameter vector Ok. The critic carries out an update similar to the average cost temporal-difference method of [12]:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(Here, 1'k is  a  positive  stepsize  parameter.) The  two  variants  of  the  critic  use different  ways of updating Zk:

TD(J)  Critic: Let x* be a  state in S.

<!-- formula-not-decoded -->

TD(a)  Critic, 0 ~ a &lt; 1:

<!-- formula-not-decoded -->

Actor: Finally, the actor updates its parameter vector by letting

<!-- formula-not-decoded -->

Here, 13k is  a  positive stepsize and r(rk) &gt; 0 is  a  normalization factor  satisfying:

- (A3)  f(•) is  Lipschitz continuous.

(A4)  There exists C &gt; 0 such that

<!-- formula-not-decoded -->

The above presented algorithms are only two out of many variations.  For instance, one could also consider  "episodic"  problems in which one starts from a given initial state  and  runs  the  process  until  a  random  termination  time  (at  which  time  the process  is  reinitialized  at x*), with the objective of minimizing  the expected cost until  termination. In this  setting,  the  average  cost  estimate Ak is  unnecessary and is  removed from  the critic update formula.  If the critic parameter rk were to be  reinitialized  each  time  that x* is  entered,  one  would  obtain  a  method  closely related to Williams' REINFORCE algorithm [13].  Such a method does not involve any  value  function  learning,  because  the  observations  during  one  episode  do  not affect the critic parameter r during another episode.  In  contrast, in  our approach, the  observations  from  all  past  episodes  affect  current  critic  parameter r, and  in this  sense  critic  is  "learning".  This can  be advantageous because,  as  long  as (J is slowly changing, the observations from  recent episodes carry useful  information on the q-function under the current policy.

## 4 Convergence of actor-critic algorithms

Since  our  actor-critic  algorithms  are  gradient-based,  one  cannot  expect  to  prove convergence to  a  globally  optimal  policy  (within  the given  class  of RSP's).  The best that one could hope for is the convergence of '\l A((J) to zero; in practical terms, this  will  usually  translate  to  convergence  to  a  local  minimum  of A((J). Actually, because the T D(a) critic will generally converge to an approximation of the desired projection of the value function, the corresponding convergence result is  necessarily weaker, only guaranteeing that '\l A((h) becomes small (infinitely often).  Let us now introduce some further assumptions.

- (A5)  For each 0 E ~n, we  define an m  x m matrix G(O) by

<!-- formula-not-decoded -->

We  assume  that G(O) is  uniformly  positive  definite,  that  is,  there  exists some fl &gt; 0 such that for  all r E ~m and 0 E ~n

<!-- formula-not-decoded -->

- (A6)  We assume that the stepsize sequences bk}, {th} are positive, nonincreas› ing,  and satisfy

<!-- formula-not-decoded -->

where 15 k stands for  either /h or 'Yk. We also assume that

<!-- formula-not-decoded -->

Note that the last assumption requires that the actor parameters be updated at a time scale slower than that of critic.

Theorem 2. In  an  actor-critic  algorithm  with  a  TD(l)  critic,

<!-- formula-not-decoded -->

Furthermore,  if {Od  is  bounded  w.p. 1 then

<!-- formula-not-decoded -->

Theorem 3. For  every f &gt; 0, there  exists a: sufficiently  close to 1, such  that liminfk IIV'A(Ok)11 ::; f w.p. 1.

Note that the theoretical guarantees appear to be stronger in the case of the TD(l) critic.  However,  we  expect that TD(a:)  will  perform  better in  practice because of much smaller variance for  the parameter rk. (Similar issues  arise when considering actor-only algorithms.  The experiments reported in  [7]  indicate that introducing a forgetting factor a:  &lt; 1 can result in much faster convergence, with very little loss of performance.)  We  now provide an overview of the proofs of these theorems.  Since 13k/'Yk --+ 0,  the  size  of the  actor  updates  becomes  negligible  compared to the size of the  critic  updates. Therefore  the  actor  looks  stationary,  as  far  as  the  critic  is concerned. Thus,  the analysis  in  [1]  for  the TD(l)  critic  and the  analysis  in  [12] for  the TD(a:)  critic  (with  a: &lt; 1) can be used,  with appropriate modifications,  to conclude that the critic's approximation of IIokqok will  be  "asymptotically correct". If r(O) denotes  the value to which  the critic  converges when  the  actor  parameters are fixed  at 0, then the update for  the actor can be rewritten as

<!-- formula-not-decoded -->

where ek is  an error that becomes asymptotically negligible.  At this point, standard proof techniques for  stochastic approximation algorithms can be used to complete the proof.

## 5 Conclusions

The  key  observation  in  this  paper  is  that  in  actor-critic  methods,  the  actor  pa› rameterization and the critic parameterization need not, and should not be chosen

independently.  Rather,  an appropriate approximation architecture for  the critic is directly prescribed by the parameterization used in  actor.

Capitalizing on the above observation, we have presented a class of actor-critic algo› rithms, aimed at combining the advantages of actor-only and critic-only methods.  In contrast to existing actor-critic methods, our algorithms apply to high-dimensional problems (they do not rely on lookup table representations), and are mathematically sound in the sense that they possess certain convergence properties.

Acknowledgments: This  research  was  partially  supported  by  the  NSF  under grant ECS-9873451, and by the AFOSR under grant F49620-99-1-0320.

## References

- [1] D.  P.  Bertsekas  and  J.  N.  Tsitsiklis. Neurodynamic  Programming. Athena Scientific,  Belmont, MA,  1996.
- [2] V.  S.  Borkar. Stochastic  approximation  with  two  time  scales. Systems  and Control Letters, 29:291-294, 1996.
- [3] X. R.  Cao and H.  F.  Chen.  Perturbation realization, potentials,  and sensitiv› ity  analysis  of Markov processes. IEEE  Transactions  on  Automatic  Control, 42:1382-1393,1997.
- [4] P.  W.  Glynn.  Stochastic approximation for  monte carlo optimization.  In Pro› ceedings  of the 1986 Winter Simulation  Conference, pages 285-289, 1986.
- [5] T. Jaakola, S. P.  Singh,  and M.  1.  Jordan.  Reinforcement learning algorithms for  partially observable Markov decision  problems.  In Advances in Neural  In› formation  Processing  Systems, volume  7,  pages  345- 352,  San  Francisco,  CA, 1995.  Morgan Kaufman.
- [6] V.  R.  Konda and V. S.  Borkar.  Actor-critic like learning algorithms for Markov decision processes. SIAM Journal on Control and Optimization, 38(1) :94-123, 1999.
- [7]  P.  Marbach. Simulation  based  optimization  of Markov  reward processes. PhD thesis,  Massachusetts Institute of Technology,  1998.
- [8] P.  Marbach  and  J.  N.  Tsitsiklis. Simulation-based  optimization  of Markov reward processes.  Submitted to IEEE Transactions on Automatic Control.
- [9]  R. Sutton and A.  Barto. Reinforcement Learning:  An Introduction. MIT Press, Cambridge, MA,  1995.
- [10] R.  S.  Sutton, D.  McAllester, S.  Singh, and Y. Mansour.  Policy gradient meth› ods for reinforcement learning with function approximation. In this proceedings.
- [11] J.  N.  Tsitsiklis  and  B.  Van  Roy. An  analysis  of  temporal-difference  learn› ing  with  function  approximation. IEEE  Transactions  on  Automatic  Control, 42(5):674-690, 1997.
- [12] J.  N.  Tsitsiklis  and  B.  Van  Roy. Average  cost  temporal-difference  learning. Automatica, 35(11):1799-1808, 1999.
- [13] R.  Williams.  Simple statistical gradient following  algorithms for  connectionist reinforcement learning. Machine  Learning, 8:229-256, 1992.