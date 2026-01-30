## A Natural Policy Gradient

## Sham Kakade

Gatsby Computational Neuroscience Unit 17 Queen Square, London, UK WC1N 3AR http: //www.gatsby.ucl.ac.uk sham@gatsby.ucl.ac.uk

## Abstract

We provide a natural gradient method that represents the steepest descent direction based on the underlying structure of the param› eter space.  Although gradient methods cannot make large changes in  the  values  of the  parameters,  we  show  that the  natural gradi› ent is moving toward choosing a greedy optimal action rather than just a  better action.  These greedy optimal actions are those that would  be  chosen  under  one  improvement  step  of policy  iteration with  approximate, compatible value  functions,  as  defined  by  Sut› ton et al. [9].  We then show drastic performance improvements in simple MDPs and in the more challenging MDP of Tetris.

## 1 Introduction

There has been a growing interest in direct policy-gradient methods for approximate planning in  large  Markov decision  problems  (MDPs). Such  methods  seek  to find a  good policy 7r among some restricted class  of policies, by following  the gradient of  the  future  reward. Unfortunately,  the  standard  gradient  descent  rule  is  non› covariant.  Crudely speaking, the rule !:l.()i = oJ]  f / a()i is  dimensionally inconsistent since  the  left  hand  side  has  units  of ()i and the right  hand side  has  units  of  l/()i (and all ()i  do  not necessarily have the same dimensions).

In  this  paper,  we  present  a  covariant  gradient  by  defining  a  metric  based  on  the underlying  structure  of  the  policy. We  make  the  connection  to  policy  iteration by showing that the natural gradient is  moving toward choosing a  greedy optimal action. We  then  analyze  the  performance  of the  natural  gradient  in  both  simple and complicated MDPs.  Consistent with  Amari's findings  [1],  our  work  suggests that the plateau phenomenon might not be as severe using this method.

## 2 A  Natural Gradient

A finite  MDP is  a tuple (S, So, A, R, P) where: S is  finite  set of states, So is  a  start state, A is  a finite  set of actions, R is  a  reward function R : S x A --+  [0 , Rmax], and P is  the transition model.  The agent's decision  making procedure is  characterized by a  stochastic policy 7r(a;  s) , which  is  the  probability of taking action a in  state s (a semi-colon is  used to distinguish the random variables from the parameters of the distribution).  We  make the assumption that every policy 7r is  ergodic,  ie has a well-defined stationary distribution p7f. Under this assumption, the average  reward (or  undiscounted reward)  is 1](  7r) == 2:: s ,a p7f (s )7r(a; S  )R(s, a), the state-action value is  Q7f(S, a) == E7f{2:::oR(st,at) -1](7r)lso = s,ao = a} and the value  function  is J7f(s) == E7f(a';s) {Q7f(s, a')}, where and St and at are the state and action at time t. We consider the more difficult case where the goal of the agent is to find a policy that maximizes the average reward over some restricted class of smoothly parameterized policies, fr = {7rO : 8 E ~m}, where tro represents the policy 7r(a; S, 8).

The exact gradient of the average reward (see  [8,  9])  is:

<!-- formula-not-decoded -->

where  we  abuse  notation  by  using  1](8)  instead  of 1](7ro). The  steepest  descent direction  of  1](8) is defined  as  the  vector d8 that  minimizes  1](8 + d8) under the  constraint  that  the  squared  length  Id812 is  held  to  a  small  constant. This squared  length  is  defined  with  respect  to  some  positive-definite  matrix G(8),  ie Id812 == 2::ij Gij (8)d8id8j = d8 T G(8)d8 (using  vector  notation).  The steepest  de› scent  direction  is  then given  by G- 1 \11](8) [1]. Standard gradient descent  follows the direction  \11](8)  which  is  the steepest descent  under the assumption that G(8) is  the identity matrix, I. However, this as  hoc choice of a  metric is  not necessarily appropriate.  As  suggested  by  Amari  [1],  it  is  better to define  a  metric  based  not on the choice of coordinates but rather on the manifold (ie the surface)  that these coordinates parameterize.  This metric defines  the natural gradient.

Though we slightly abuse notation by writing 1](8), the average reward is technically a  function  on  the  set  of  distributions {7rO :  8  E ~m}. To  each  state s, there corresponds a  probability manifold,  where the distribution 7r(a; S, 8)  is  a  point  on this manifold with coordinates 8.  The Fisher information matrix of this distribution 7r(a;  s,8) is

<!-- formula-not-decoded -->

and it  is  clearly  positive  definite. As  shown  by  Amari  (see [1]), the  Fisher  infor› mation matrix, up to a scale, is  an invariant metric on the space of the parameters of  probability  distributions. It  is  invariant  in  the  sense  that  it  defines  the same distance between two points regardless of the choice of coordinates (ie the param› eterization)  used,  unlike G = I.

Since the average reward is  defined on a  set of these distributions, the straightfor› ward choice we  make for  the metric is:

<!-- formula-not-decoded -->

where the expectation is  with respect to the stationary distribution of 7ro. Notice that  although each Fs is  independent  of the  parameters of the  MDP's transition model, the weighting by the stationary distribution introduces dependence on these parameters.  Intuitively, Fs (8)  measures  distance  on a  probability manifold  corre› sponding to state sand F(8) is  the  average  such  distance.  The steepest  descent direction this gives is:

<!-- formula-not-decoded -->

## 3 The Natural Gradient and Policy Iteration

We now compare policy improvement under the natural gradient to policy iteration. For an appropriate comparison, consider the case in which Q7r (s, a) is approximated by some compatible function approximator r(s,a;w) parameterized by w [9,  6].

## 3.1 Compatible Function Approximation

For vectors (), w E ~m, we  define:

<!-- formula-not-decoded -->

where [\7logn(a;s, ())]i = 8logn(a;s, ())!8()i. Let w minimize  the  squared  error f(W, n)  == L,s,a p7r (s )n(a;  s, ())(r  (s, a; w) \_Q7r (s, a))2. This function approximator is compatible with the policy in the sense that if we use the approximations f7r (s, a; w) in  lieu  of their  true  values  to  compute  the  gradient  (equation  1),  then  the  result would still be exact [9, 6]  (and is thus a sensible choice to use in actor-critic schemes).

Theorem 1. Let w minimize the  squared  error f(W, no).  Then

<!-- formula-not-decoded -->

Proof. Since w minimizes  the squared error,  it  satisfies  the condition 8f!8wi = 0, which implies:

<!-- formula-not-decoded -->

or equivalently:

<!-- formula-not-decoded -->

By definition  of 'ljJ7r, \7n(a;s,()) = n(a;s,())'ljJ7r(s,a) and  so  the  right  hand  side  is equal to \71}.  Also by definition of 'ljJ7r, F( ()) = L,s,a p7r (s )n(  a;  s, ()) 'ljJ7r (s, a )'ljJ7r (s, a) T. Substitution leads to:

<!-- formula-not-decoded -->

Solving for w gives w = F(()) - l\71}(()), and the result follows  from  the definition of the natural gradient. D

Thus, sensible actor-critic frameworks (those using f7r(s ,  a; w)) are forced to use the natural gradient as the weights of a linear function approximator. If the function ap› proximation is  accurate, then good actions (ie those with large state-action values) have feature vectors that have a large inner product with the natural gradient.

## 3.2 Greedy Policy Improvement

A  greedy  policy  improvement  step using  our function  approximator would  choose action a in  state s if a E  argmaxa,  f7r (s, a';  w). In  this  section,  we  show  that  the natural  gradient  tends  to  move  toward  this best action,  rather  than just  a  good action.

Let  us  first  consider  policies  in  the  exponential  family (n(a;s, ()) IX exp(()T¢sa) where ¢sa is some feature vector in ~m). The motivation for the exponential family is  because it has affine geometry (ie the flat geometry of a plane), so a translation of a  point by a tangent vector will keep the point on the manifold.  In general, crudely speaking, the probability manifold of 7r(a;  s, 0) could be curved, so  a translation of a  point  by  a  tangent  vector  would  not  necessarily  keep  the point on the manifold (such as on a sphere).  We consider the general (non-exponential)  case later.

We now show, for the exponential family, that a sufficiently large step in the natural gradient direction will  lead to a  policy that is  equivalent to a  policy found  after a greedy policy improvement step.

Theorem 2. For 7r(a;  s, 0) ex: exp(OT 1&gt;sa),  assume  that ~'TJ(O) is non-zero  and  that w minimizes the  approximation  error.  Let7roo (a;s) =lima-+oo7r(a;s, O+a~'TJ(O)). Then 7r  00 (a; s) 10 if and  only  if a E argmaxa, F' (s, a';  w).

Proof. By  the  previous  result, F'(s,a;w) = ~'TJ(O)T'lj;7r(s,a). By  definition  of 7r(a;  s, 0) , 'lj;7r (s, a) = 1&gt;sa -E7r(a';s,O) (1)sa'). Since E7r(a';s,O) (1)sa') is  not  a  function of a, it follows  that

<!-- formula-not-decoded -->

After  a  gradient  step, 7r(a; s, 0 + a~'TJ(O)) ex: exp(OT 1&gt;sa + a~'TJ(O)T 1&gt;sa). ~'TJ(O) 10,  it  is  clear  that  as  a -+ 00  the  term ~'TJ(O)T 1&gt;sa dominates, 7r  00 (a, s) = 0 if and only if a f{. argmaxa, ~ 'TJ(  0)  T  1&gt;sa' . Since and  so D

It is  in  this  sense that the natural gradient tends to move toward choosing the best action. It is  straightforward to  show  that  if the  standard  non-covariant  gradient rule  is  used  instead  then 7r oo (a;  s) will  select  only  a better action  (not  necessarily the best), ie it will  choose an action a such that F'(s,a;w) &gt; E7r(a';s){F'(s,a';w)}. Our use of the exponential family was only to demonstrate this point in the extreme case of an infinite learning rate.

Let us return to case of a general parameterized policy.  The following theorem shows that the natural gradient is  locally moving toward the best action, determined by the local linear approximator for  Q7r (s, a).

Theorem 3. Assume that w minimizes the  approximation  error and let the  update to  the  parameter be  0' = 0 + a~'TJ(O). Then

<!-- formula-not-decoded -->

Proof. The change in 0,  ,6.0, is a~'TJ(O), so  by theorem 1, ,6.0 = aw. To first  order,

<!-- formula-not-decoded -->

where we  have used the definition of 'lj; and f.

D

It is  interesting to note that choosing the greedy action will not in general improve the policy,  and many detailed studies have gone into understanding this failure  [3]. However,  with  the  overhead of a  line  search,  we  can  guarantee improvement  and move toward this greedy one step improvement.  Initial improvement is  guaranteed since F is  positive definite.

## 4 Metrics and Curvatures

Obviously,  our choice of F is  not  unique  and the question  arises  as  to whether or not  there  is  a  better  metric  to  use  than F. In  the  different  setting  of  parameter estimation, the Fisher information converges to the Hessian, so it is  asymptotically efficient  [1], ie attains  the  Cramer-Rao  bound. Our situation  is  more  similar  to the blind source separation case where a  metric is  chosen based on the underlying parameter space [1]  (of non-singular matrices) and is  not necessarily asymptotically efficient (ie does  not attain second  order convergence).  As  argued by  Mackay  [7], one strategy is to pull a metric out of the data-independent terms of the Hessian (if possible),  and in fact,  Mackay  [7]  arrives at the same result as Amari for  the blind source separation case.

Although the previous sections argued that our choice is appropriate, we would like to understand how F relates to the Hessian V 2 TJ(B), which, as shown in [5],  has the form:

<!-- formula-not-decoded -->

Unfortunately, all terms in this Hessian are data-dependent (ie are coupled to state› action values) .  It is  clear that F does  not capture any information from  these last two terms, due to their VQ7r  dependence.  The first  term might have some relation to F due  the  factor  of V 2 7f. However, the  Q  values  weight  this  curvature of our policy and our metric is  neglecting such weighting.

Similar to the blind source separation case, our metric clearly does  not necessarily converge to the Hessian and so it is  not necessarily asymptotically efficient (ie does not attain a  second order convergence rate).  However, in general, the Hessian will not be positive definite and so the curvature it provides could be of little use  until B is  close  to  a  local  maxima. Conjugate methods  would  be  expected  to be  more efficient  near a  local maximum.

## 5 Experiments

We first  look  at  the  performance  of the  natural  gradient  in  a  few  simple  MDPs before  examining  its  performance  in  the  more  challenging  MDP  of Tetris. It is straightforward to  estimate F in  an  online  manner,  since  the  derivatives  V log 7f must be computed anyway to estimate VTJ(B). If the update rule

<!-- formula-not-decoded -->

is  used  in  a  T-Iength  trajectory,  then fiT is  a  consistent  estimate  of F. In  our first  two  examples,  we  do  not  concern ourselves  with  sampling issues  and instead numerically  integrate  the  exact  derivative (Bt = Bo + J~ VTJ(BddB). In  all  of our simulations, the policies tend to become deterministic (V log 7f -+ 0)  and to prevent F from becoming singular, we add about 10- 3 1 at every step in all our simulations.

We simulated the natural policy gradient in a simple I-dimensional linear quadratic regulator  with  dynamics x(t + 1) = .7x(t) + u(t) + E(t) and  noise  distribution E ~ G(O,l). The  goal  is  to  apply  a  control  signal u to  keep  the  system  at x = 0,  (incurring  a  cost  of X(t)2 at  each  step). The  parameterized  policy  used was 7f(u; x, B) ex  exp(Blx 2 + B2X). Figure  lA shows  the performance improvement when the units of the parameters are scaled by a factor of 10  (see figure text).  No› tice that the time to obtain a score of about 22  is  about three orders of magnitude rl

Figure 1:  A)  The cost Vs.  10glo(time) for  an LQG  (with 20  time step trajectories). The policy used was 7f(u; x, ()) ex:  exp(()lslX2 + ()2S2X) where the rescaling constants, Sl and S2, are shown in the legend.  Under equivalent starting distributions (()lSl = ()2S2 = -.8) , the right-most three curves are generated using the standard gradient method and the  rest  use  the  natural gradient. B)  See  text. C  top)  The average reward vs.  time  (on a  10 7  scale)  of a  policy under standard gradient descent  using the sigmoidal policy parameterization (7f(I; s, ()i)  ex:  exp(()i)/(1 + exp(()i)),  with the initial  conditions  7f(i, 1) = .8  and 7f(j, 1) = .1.  C  bottom)  The average reward vs. time  (unscaled)  under  standard gradient  descent  (solid  line)  and natural gradient descent  (dashed line)  for  an early window of the above plot.  D)  Phase space plot for  the standard gradient case (the solid line)  and the natural gradient case (dashed line) .

<!-- image -->

faster.  Also  notice  that the curves under different  rescaling are not identical. This is because F  is  not  an  invariant metric due to the weighting by Ps.

The effects  of the weighting by p(s) are particularly clear in a  simple 2-state MDP (Figure  IB),  which  has  selfand  cross-transition  actions  and  rewards  as  shown. Increasing the chance of a  self-loop  at i decreases the stationary probability of j. Using  a  sigmoidal  policy  parameterization  (see  figure  text)  and  initial  conditions corresponding to p(i) = .8  and p(j) = .2, both self-loop action probabilities will ini› tially be increased under a gradient rule (since one step policy improvement chooses the  self-loop  for  each  state). Since  the  standard gradient  weights  the  learning to each parameter by p(s) (see  equation 1), the self-loop action at state i  is  increased faster than the self loop probability at j, which has the effect  of decreasing the ef› fective learning-rate to state j  even further.  This leads to an extremely fiat plateau with average reward 1 (shown in  Figure  lC top),  where the learning for  state j  is thwarted by its low stationary probability.  This problem is so severe that before the optimal policy is  reached p(j) drops as low as 10- 7  from its initial value of .2, which is  disastrous  for  sampling  methods. Figure  1  C  bottom  shows  the  performance of the natural gradient  (in  a  very early time window of Figure  lC top).  Not only  is the  time  to  the optimal policy  decreased  by  a  factor  of 10 7 ,  the stationary distri› bution of state i never  drops  below  .05. Note  though the standard gradient  does increase the average reward faster  at the start, but only to be seduced by sticking at  state  i. The  phase space  plot  in  Figure  ID  shows  the  uneven  learning to  the different  parameters,  which  is  at  the  heart  of the  problem. In  general,  if  a  table lookup Boltzmann policy is  used  (ie  7f( a;  s , ()) ex: exp( () sa)), it  is  straightforward to show that the natural gradient weights the components of ~'fJ uniformly (instead of using p(s)), thus evening evening out the learning to all parameters.

The game of Tetris provides a  challenging high dimensional problem.  As  shown in [3],  greedy  policy  iteration  methods  using  a  linear  function  approximator exhibit drastic  performance  degradation  after  providing  impressive  improvement  (see  [3] for  a  description of the game, methods, and results).  The upper curve in Figure2A replicates these results.  Tetris provides an interesting case to test gradient methods,

Figure 2:  A)  Points vs.  10g(Iterations).  The top curve duplicates the same results in  [3]  using the  same features  (which  were  simple functions  of the heights  of each column and the number of holes in the game).  We have no explanation for this per› formance degradation (nor does  [3]).  The lower curve shows the poor performance of the standard gradient rule.  B)  The curve on the right shows the natural policy gradient  method  (and uses  the biased  gradient method of [2]  though this  method alone gave poor performance).  We found  we  could obtain faster  improvement and higher asymptotes if the robustifying factor of 10- 3 I that we added to F was more carefully  controlled  (we  did  not  carefully  control  the  parameters). C)  Due  to the intensive computational power required of these simulations we ran the gradient in a smaller Tetris game (height of 10 rather than 20)  to demonstrate that the standard gradient updates (right curve) would eventually reach the same performance of the natural gradient (left curve).

<!-- image -->

which  are guaranteed not  to degrade the policy.  We  consider a  policy  compatible with the linear function approximator used in [3] (ie  7f(a;s, (}) ex: exp((}T¢sa) where ¢sa are the same feature vectors).  The features  used in  [3]  are the heights  of each column, the differences in height between adjacent columns, the maximum height, and the number  of 'holes'. The lower  curve  in  Figure  2A  shows  the  particularly poor performance of the standard gradient method.  In an attempt to speed learn› ing,  we  tried a variety of more sophisticated methods to no avail, such as conjugate methods,  weight  decay,  annealing,  the  variance  reduction  method  of [2],  the  Hes› sian in  equation 6, etc. Figure 2B  shows  a  drastic improvement using the natural gradient (note that the timescale is linear).  This performance is consistent with our theoretical  results  in  section  3,  which  showed  that  the  natural gradient  is  moving toward the solution of a greedy policy improvement step.  The performance is some› what slower than the greedy policy iteration (left curve in Figure 2B)  which is to be expected using smaller steps.  However, the policy does not degrade with a gradient method.  Figure 2 shows that the performance of the standard gradient rule  (right curve)  eventually  reaches  the  the  same  performance of the  natural gradient,  in  a scaled down version of the game  (see figure text).

## 6 Discussion

Although gradient methods cannot make large policy changes compared to greedy policy  iteration,  section  3  implies  that  these  two  methods  might  not  be  that dis› parate, since  a  natural gradient method is  moving toward the solution of a  policy improvement step.  With the overhead of a  line search, the methods are even more similar.  The benefit is  that performance improvement is  now guaranteed, unlike in a  greedy policy iteration step.

It is  interesting,  and unfortunate,  to note that the F does not asymptotically con› verge to the Hessian, so conjugate gradient methods might be more sensible asymp› totically. However,  far  from  the  converge  point,  the  Hessian  is  not  necessarily informative, and the natural gradient could  be  more efficient  (as  demonstrated in Tetris).  The intuition as to why the natural gradient could be efficient far from the maximum, is that it is  pushing the policy toward choosing greedy optimal actions. Often,  the region  (in  parameter space)  far  from from  the maximum is  where large performance changes could occur.  Sufficiently close to the maximum, little perfor› mance change occurs  (due  to the small gradient),  so  although conjugate methods might  converge faster  near the  maximum,  the corresponding performance  change might be negligible.  More experimental work is necessary to further understand the effectiveness of the natural gradient.

## Acknowledgments

We thank Emo Todorov and Peter Dayan for  many helpful discussions.  Funding is from the NSF and the Gatsby Charitable Foundation.

## References

- [I] S.  Amari. Natural  gradient  works  efficiently  in  learning. Neural  Computation, 10(2):251- 276,  1998.
- [2] J.  Baxter  and P.  Bartlett.  Direct  gradient-based reinforcement  learning. Technical report,  Australian National University, Research  School of Information Sciences and Engineering, July 1999.
- [3] D.  P.  Bertsekas and J.  N.  Tsitsiklis. Neuro-Dynamic Programming. Athena Scientific, 1996.
- [4] P.  Dayan and G.  Hinton.  Using em for  reinforcement learning. Neural  Computation, 9:271- 278,  1997.
- [5] S.  Kakade. Optimizing average  reward  using  discounted reward. COLT.  in press., 200l.
- [6] V.  Konda and J.  Tsitsiklis.  Actor-critic algorithms. Advances  in Neural  Information Processing  Systems, 12, 2000.
- [7] D.  MacKay.  Maximum likelihood and covariant  algorithms for  independent compo› nent analysis.  Technical report, University of Cambridge, 1996.
- [8] P.  Marbach  and J .  Tsitsiklis.  Simulation-based optimization of markov reward pro› cesses.  Technical report,  Massachusetts Institute of Technology, 1998.
- [9] R.  Sutton,  D.  McAllester,  S.  Singh,  and Y. Mansour. Policy  gradient  methods for reinforcement  learning with function  approximation. Neural  Information  Processing Systems, 13,  2000.
- [10] L.  Xu and M.  1.  Jordan.  On convergence properties of the EM algorithm for  gaussian mixtures. Neural  Computation, 8(1):129- 151,  1996.