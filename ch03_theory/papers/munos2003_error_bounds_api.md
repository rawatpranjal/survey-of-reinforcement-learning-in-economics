## R/!ml Munos

## Error  Bounds for Approximate Policy Iteration

P~EMI.MUNOS@POLYTECHNIQUE. FR

Centre  de Math~matiques  Appliqu~es.  Ecole  Polytechnique.  91128 Palaiseau, France. http://www.cmap.polytechnique.fr/~ munos/

## Abstract

In  Dynamic  Programming,  convergence of  algorithms  such as  Value  Iteration or  Policy Iteration  results -in  discounted problems- from a contraction property  of  the  back-up operator, guaranteeing convergence  to  its fixedpoint. When  approximation is considered, known  results in Approximate Policy  Iteration provide  bounds  on the  closeness  to  optimality of  the  approximate  value  function obtained  by  successive policy  improvement steps as  a  function of  the maximum  norm of  value  determination  errors during  policy evaluation steps. Unfortunately,  such results have limited  practical range since  most function approximators  (such  as  linear regression)  select the  best  fit in  a  given class  of parameterized  functions  by minimizing some (weighted) quadratic  norm.

In this paper, we provide error bounds for Approximate  Policy Iteration using quadratic norms, and illustrate those  results in  the  case  of  feature-based linear function approximation.

## 1. Introduction

We  consider a Markov Decision  Process (MDP)  (Puterman, 1994; Bertsekas  &amp;  Tsitsiklis, 1996;  Sutton  &amp; Barto, 1998)  evolving on  a  state space X with states. Its dynamics is  governed by the transition probability function P(i,a,j) which gives  the  probability that the  next  state is j  E X knowing that the  current state is  i  E X and the  chosen action  is a  E A, where A is the  (finite) set of  possible actions. A policy 7r is a  mapping  from  X to A.  We write P~  the N x  N-matrix whose  elements are P~(i,j) = p(i, zr(i),j). Let r(i,a,j) be the  reward received  when  a transition from state i, action  a,  to state j  occurs.  Write r ~  the  vector  whose  components are r~(i)  = ~.  P~(i,j)r(i, Tr(i),j). Here, we consider discounted,  in~nite  horizon problems.

The value function V~(i) for  a policy zr  is  the  expected sum  of  discounted  future rewards when  starting from state  i  and using policy ~r:

<!-- formula-not-decoded -->

where rt is the  reward received at  time  t  and 3,  e [0,1)  a discount factor. It  is  known that  V"  solves  the Bellman equation

<!-- formula-not-decoded -->

Thus V ~  (considered as  a  vector of  size N) is the fixed-point of  the back-up  operator  T ~ defined  by T ~= r ~  + 3'P ~.. Since P~ is  a stochastic matrix,  it possesses  eigenvalues with  module  less than  or  equal to  one,  thus (I3'P ~) is invertible, and we write V ~ = (I -  7P~)-lr ~.

The optimal  value  function V* is the expected  gain when using an  optimal policy 7r*: V* =  V ~"  = sup, V ~. We  are  interested in  problems with  large state  spaces (N is  very large,  possibly infinite), which prevents  us  from using  exact  resolution methods  such as Value  Iteration or  Policy Iteration with look-up  tables. Instead, we consider the  Approximate Policy Iteration algorithm (Bertsekas &amp; Tsitsiklis, 1996) defined iteratively by the  two  steps:

- ¯ Approximate  policy evaluation: for  a given  policy  zrk,  generate an approximation Vk of the  value function  V ~k
- ¯ Policy  improvement: generate  a  new  policy  ~rk+l greedy with respect  to Vk:

<!-- formula-not-decoded -->

These steps are  repeated until no more improvement of the  policies is  noticed (using  some  evaluation  criterion). Empirically,  the  value  functions  "~  rapidly improve  in  the  first iterations of this  algorithm~  then oscillations occur with no more  performance  increase. The behavior  in  the  transitional phase is  due to  the

relatively good approximation of  the  value  function ([[Vk -~  [ [ i s l ow) in comparison t o t he closeness t optimality  [IV~  -  V*  [[, which  produces  greedy policies (with  respect to  the  approximate Vk) that  are  better than the  current  policies. Then, once some  closeness to optimality is  reached, the  error  in  the  value approximation  prevents  the  policy  improvement  step  from being efficient: the  stationary  phase  is  attained.  Hence,  this algorithm does not converge (there  is  no stabilization to  some  policy)  but it is  very fast and from the  intuition above, we can  expect  to  quantify  the  closeness to  optimality  at  the  stationary phase as  a function  of the  value  approximation errors. And  indeed,  a known result (Bertsekas &amp; Tsitsiklis, 1996, chap. 6.2)  provides  bounds  on the  loss V*  -  V ~  of  using  policy  irk instead  of  using the  optimal one,  as a  function of  the maximum norm of  the  approximation errors Vk -V ~k  :

<!-- formula-not-decoded -->

However,  this result is difficult to  use in  many  approximation  architectures (exceptions include  (Gordon,  1995;  Guestrin  et  al., 2001))  since it is  very costly to  control the  maximum norm;  the  weighted quadratic norms are  more commonly  used. We  recall that a  distribution #  on  X  defines an inner-product (f,h), = ~1 #(i)f(i)h(i) and a  quadratic (semi-) norm I[h[[~ = (h,h)~/2. Of course, equivalency  between norms implies  that [[hi[ &lt; [[h[[c~  \_&lt; V~[[h[[ (where [[. I[  denotes the  norm  defined  by the  uniform distribution p -1).  But then,  the  bound  (1), rewritten  in  quadratic norm will include the factor v~, which  is  too large  for  being of  any use in  most cases.

Our  main  result, stated  in  Section  2 and proved  in  Appendix A, is  to  derive  analogous bounds  in quadratic norms: the  loss  [IV* -  V ~k  [[~  (for  any distribution #) is bounded  by a  function of  the  approximation error [[Vk -V ~ [[~  (for  some  distribution #k related to and the policies  ~rk and ~r*),  as  well as by the  Bellman residual (Baird, 1995) [[Vk -TTr~Vk[[~ (for another distribution ~k).

In  Section  3,  we apply  those  results to  the  featurebased  linear function approximation  (where  the  parameterized functions are  weighted  linear combinations of  basis functions -the features), which have been  considered in Temporal  Difference learning TD(A)  (Tsitsiklis &amp;  Van Roy, 1996) and Least-Squares Temporal  Difference: LSTD(0) (Bradtke &amp; Barto, 1996), LSTD(A)  (Boyan,  1999), and  LS-Q-learning (Lagoudakis &amp; Parr, 2001).

Both the  approximations  obtained  by minimizing the quadratic Bellman residual and by finding  the  TD  so- lution (the fixed-point of  a  combined  operator) are considered in  sections 3.2 and 3.3. Under the assumption  of uniform  stochasticity of  the  MDP (Hypothesis 2), bounds  on  [[V*-V~"[[oo are derived based  on  the  minimum possible approximation  error infa  ][Va -  V~[[p..  Proofs are  given  in  Appendix  B.

These  linear approximation  architectures combined with  policy  improvement still lack  theoretical analysis  but  have produced  very promising experimental results on large  scale  control  and reinforcement learning problems  (Lagoudakis  &amp; Parr, 2001); we hope that this paper will  help  better  understand their behavior.

## 2. Quadratic Norm Bounds

Consider the  Approximate  Policy  Iteration algorithm described in  the  introduction.  7rk represents  the  policy at  iteration k,  and Vk  the  approximation  of  the  value function  V ~k .  The main result of  this paper is  stated in  this  theorem.

Theorem  1 For any  distribution #  (considered as row vector)  on X,  define  the  stochastic  matrices

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Write  #k = #Qk and ~k = #Qk. Then  #k  and  ~k  are distributions on X,  and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Some  intuition about this result as  well  as  its proof may be  found in  Appendix A.

Notice that  this result is stronger  than  the  bound  in max-norm (1), since  from (3)  and using  the  fact H" [[~  &lt;  [[" [[~, we deduce that  limsupk\_\_~o o  Jig*  ~ suPk [[Vk--V ~  [[oo  for  any distribution y  ll,  \_ #, which  implies (1).

Moreover, it provides information about what parts  of the  state-space  are  responsible  (in  terms of local  approximation error Vk  --  V 7r"  ) for the loss V*(i) -  ~r~'  (i) at  any state i. This information indicates  the  areas  of the  state space where we should focus  our  efforts in

the  value  approximation  (e.g. by locally reallocating computational  resources,  such as in  variable  resolution discretization (Munos  &amp; Moore, 2002)).

In  the  next section  we  describe how  to  use this result to  derive error  bounds  on the  loss  V*  -  V ~k  in  the  case of  linear  approximation  architectures.

## 3. Approximate Policy Evaluation

## 3.1. Linear feature-based approximation

We  consider  a class of functions  V, = ~a linearly parameterized  by a parameter a  (vector  of  size  K, usually  much  smaller than N), where  ¯  is  the  set  of  basis functions, called features (a  N x K matrix  in  which each column  represents  a  feature).

We  assume that the  columns of  ~ are  linearly independent.  Such  linear architectures include  state aggregation methods,  CMACs,  polynomial  or  wavelet regression  techniques,  radial basis  function  networks with  fixed bases, and finite-element methods. They have  been used  in  incremental Temporal Difference TD(A)  (Tsitsiklis &amp; Van  Roy, 1996) or  Least-Squares TD  (LSTD)  (Bradtke  &amp; Barto, 1996),  (Boyan,  1999). These LSTD  methods which 'knakes  efficient use  of training samples collected in  any arbitrary manner" have  recently been  extended to  model-free LS-Qlearning (Lagoudakis &amp; Parr, 2001). They  have demonstrated  very  good efficiency in  reinforcement learning  and control  of large  scale  problems.

The space of  parameterized functions  is  written  [~p] (the span of  the  columns  of  ~). At iteration k,  the approximate policy  evaluation  step  selects a  "good" approximation Vak  (written Vk  for simplicity) of  the value  function V ~ , in  the sense  that some (semi-) norm  [[Vk  -  V~[[p~ be  minimized, as  much  as  possible.  Several  approaches  for  this  minimization problem are  possible  (Bertsekas &amp;  Tsitsiklis, 1996; Schoknecht, 2002; Judd,  1998):

- ¯ Find  the optimal  approximate  solution, which is  the  best  possible  approximation in  [~]:  Vk  is the  orthogonal projection  Hp~  V ~k  of  V nk  onto [~] with respect  to  the  norm  [[. I[p~.  This regression problem  is very  costly  since  V nk  is  unknown,  but estimations may  be obtained  by Monte-Carlo simulations.
- ¯ Find  the minimal quadratic residual (QR) solution, which  is the  function  Vk  that  minimizes the  quadratic  Bellman  residual  [[Va T~kV~[[p~. This problem  is easy to  solve  since  it  reduces to the  resolution  of a  linear  system of  size  K: Find

a  such that

<!-- formula-not-decoded -->

where Dp~  is the N x  N diagonal matrix  whose elements are Dp~(i,i) = pk(i). This  problem always  admits a solution  since A  is  invertible.

- Find  the Temporal  Difference (TD)  solution, which  is  the  fixed-point  of  the  conjugate operator Hp~T ~  the  back-up  operator followed  by  the projection onto [~]  w.r.t  [[. [[pk- i.e. Vk  satisfies Vk  = Hpk  T *~  Vk. Again, this problem reduces  to a  linear  system of  size  K: Find a such that

<!-- formula-not-decoded -->

Here, A is  not always invertible.

The matrix  A and vector  b  of  the  QR  and TD  solutions may  be estimated  from transition data  coming  from arbitrary  sources,  e.g.  incrementally (Boyan, 1999)  from the  observation of trajectories induced by a given policy  or  by random policies (Lagoudakis  8~ Parr, 2001), or  by archived  data  coming  from prior  knowledge.

Thus, one needs to  specify  the  distribution Pk used in the  minimization  problem,  which usually depends on the  policy  7rk.  A steady-state distribution ~,  which would weight more  the  states that  are  frequently  visited, would be desirable for  purely  value  determination. However,  the  policy  improvement  step  may  perform badly since,  from Lemma 3 (see  Appendix  A),  the gain  in  policy  improvement  depends on the  value  approximation  at  states  reached  by policy  ~rk+t as well as their successors (for  policy  7r~),  which  may  be poorly approximated  if they are  ill-represented in  ~.  A more uniform  distribution p~ would  give  weight  to  all  states thus  insuring  a  more secure  policy  improvement  step (Koller &amp; Parr, 2000;  Kakade  &amp; Langford,  2002). consider these possible choices for  p~:

- Steady-state distribution  fi~  (if a such exists). satisfies the property ~ = ~  P~.
- Constant distribution ~ (does not  depend  on rc~).
- Mixed distribution p~  = ~(IAP~)-~(1  (for  0 &lt; A  &lt; 1),  which starts from  an initial distribution ~,  then  transitions induced by zr  occur for a  period  of  time that is a  random  variable that follows  an  exponential law At(1  -  A).  Thus corresponds to  the  distribution of  a Markov  chain that starts from a  state sampled according to

and which, at  each iteration, either  follows policy 7r  with  probability A or  restarts to  a  new  state with probability 1  -  A. Notice that when  A  tends to  0 then pX k  tends to  the  constant distribution  -f, and when  A  tends  to  1,  p  X tends  to  the  steadystate  distribution.

- ¯ Convex  combination of  constant  and steady-state distributions: p~ = (1  -  ~)-f + (i~.

Now,  in  order  to  bound  the  approximation  error HVk  -V~ll~k and Bellman residual IIVk  --  T~Vkll,k (to  be used in  Theorem  1)  as  a  function  of  the  minimum possible  approximation error  infa  I  IVa  -  V~I  I  p~, we  need some  assumption about  the  representational power of the  approximation architecture.

Hypothesis 1  (Approximation hypothesis) For any policy 7r ,  there exists,  in  the class o] parameterized functions, an  e-approximation (in p~-norm) of  the value function V~: for  some  e &gt; O, for  all  policies  7r,

<!-- formula-not-decoded -->

where p~ may  depend  on the  policy  7r.

Next,  we study  the  cases  where the  approximate function Irk  is chosen to  be the  QR  solution (subsection 3.2)  and the  TD  solution  (subsection  3.3).

## 3.2. The  Quadratic Residual solution

Consider ~k the  parameter  that minimizes the  Bellman  residual in  quadratic  pk-norm  (solution to  (4)). Write Irk  = Va~  = ~liak  the  corresponding value  function:

<!-- formula-not-decoded -->

Since,  for all o~,  Va -T'~kV,~  = (I7P~)(V~  -V~k), we deduce  that

<!-- formula-not-decoded -->

where II1" IIIp~ is the  matrt~ norm induced by  I1"  I1~ (i.e.  IIIAIIla :=  supll~ll~=x IIAxllp)Now we  have bound  on the  residual Vk -  T~r~vk in pk--norm, but  in Theorem  1 we actually need such a  bound  in  #k--normA crude  (but somehow  unavailable) bound

<!-- formula-not-decoded -->

where  II\_~lloo  express the  mismatch  between  the PU rather unknown  distribution ~u~ = /zQk and the  distribution Pk used in  the  minimization problem. In order  to  bound  this ratio, we now  provide conditions  for which a  upper-bound for  #k and a lower-bound for Pk are  possible.

We  make the  following assumption  on  the  MDP.

Hypothesis 2  (Uniform  stochasticity) Let  -fi be some  distribution, for  example  a  uniform distribution. There  exists  a constant C, such that  for  all  policies  r, /or  all i, j e  X,

<!-- formula-not-decoded -->

Notice that  this  hypothesis can always be satisfied for -f(i) = 1IN by choosing  C = N.  However, we are  actually interested in  finding  a  constant  C &lt;&lt; N, which requires, intuitively, that  each state  possesses  many successors  with rather small  corresponding transition probabilities.

Remark  1 An interesting case  for  which this assumption  is satisfied is  when  the  MDP has continuous-space (thus  N = oo but  all ideas  in  previous analysis  remain valid). In such case,  if the  continuous problem  has transition  probability kernel P~r(x, B) (probability  that the  next  state belongs  to  the  subset  B C X when  the current state  is  x  6 X and  the  chosen  action  is  ~r(x)), then the hypothesis reads that  there exists  a measure  -f on  X (with -f(X) = 1)  such  that Pr(x,B) &lt; C-f(B) for all x  and all subset  B.  This  is true  as  long  as the  transition  probabilities  admit  a pdf representation: p~ (x, B) = fB p~(ylx)dy with bounded  density p~(.[x).

From  this assumption,  we derive  a  bound  for  #k:

Lemma  1 Assume Hypothesis ~.  Then  lz~ &lt; C-f.

Remark  2 An assumption on  the Markov process, other than  Hypothesis 2, that would guarantee  an upper-bound  for  #k is  that  the  matrix  P~ and the  resolvent (I  -  7P ~)  -~ (1 -  9') have bounded entrant  probabilities: there exists two  constants C~ &lt;&lt; N and 02 &lt;&lt; N such that  for  all 7r  and all j  6 X

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then,  a  bound  is  #~ &lt; CtC~Iz (we will not  prove this result here). An important  case  for  which this assumption is satisfied is  when the  MDP is built from a discretization of  a (continuous-time)  Markov  diffusion  process for  which  the  state-dynamics are governed by stochastic  differential equations  with non-degenerate diffusion coe~O~eients.

The distributions pX and  P~k previously defined, which mix the  steady-state distribution to  a  rather uniform  distribution ~,  can  be  lower-bounded when ,~  &lt; 1  or  6 &lt; 1,  which allows to  use inequalities (7) and (2)  to  derive  an error  bound  on the  loss  V*  -  ~k when  using  the  QR  solution:

Theorem 2 Assume  that Hypothesis 2  holds with some  distribution -fi and constant C.

- ¯ Assume  that  Hypothesis 1 holds with the  distributionp~ = ~(/-AP'~)-I(1-A) (withO &lt;  A &lt;  1), then

<!-- formula-not-decoded -->

- ¯ Assume  that  Hypothesis 1  holds with  the  distributionp~. = (1-6)~+5~=. (withO &lt;  6  &lt; 1) (where -~ is  the  steady-state  distribution for  nk),  then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark  3 Note that  if the  steady-state distribution ~ is itself lower-bounded by some constant-fi )  O, then the  bounds  on p~,  P~k  can be tightened  for  A and 6 close  to  1,  which  would  suppress the  terms 1  -  A and 1 -  6 in  the  denominators of  the  right  hand  side  of  the above  inequalities.

## 3.3. Temporal  Difference solution

Now,  we consider  that Vk  is  the  Temporal Difference solution, i.e. the  fixed-point  of the  combined operator Hp~  T ~. We  notice that Vk  solves  the  system (equivalent  to  (5)  because ¯  has full column  rank):

<!-- formula-not-decoded -->

which has a solution  if the  matrix  (I  -  3'Hp~  P~") invertible. The approximation  error ek  = Yk  --V TM solves  the  system

<!-- formula-not-decoded -->

where  ¢k is the optimal approximation  error.

## 3.3.1. WHICH  DISTRIBUTION?

If  Pk  is  the  steady-state distribution  fi~  for policy zrk, then  we have [[[Pn~l[[P-k = 1  (Tsitsiklis &amp;  Van Roy,

1996). Thus,  if  Hypothesis  1 is  satisfied for  the steadystate distribution, then  from (10),  we deduce a bound on the  approximation error

<!-- formula-not-decoded -->

Now, ifpk  is  different  from  ~,,.  then  IIIP"'III~. (which is  always &gt; 1 since  P~" is  a stochastic matrix) may  be greater than  1/7  and (11)  does  not  hold  any  more. Even if we assume that for all policies 7r~  the  matrices I  -  7Hp~  P~' are  invertible (thus, that  the  V~ are  well-defined), which means that the  eigenvalues of IIp~  P~  are  all different from 1/0', it seems difficult to  provide  bounds on the  approximation error e~  = (I -  7Hp~  ~ ) -~ because t hose e igenvalues may  be close  to  1/3':  we can easily build  simple examples  for which the  ratio  of  [[e~[[p~ (as well as the  Bellman  residual [[Vk Tn~Vkl[p~ = [[(I -7P~r~)ek[lp~) by e  is as  large  as  desired. Some  numerical  experiments showed that the  TD  solution provided  better policies  than  the  QR solution  although the  value functions were not  so  accurately approximated.  The reason  argued was that the  TD  solution ~preserved  the shape of  the  value function to  some  extent  rather  than trying  to  fit the  absolute  values",  thus  %he  improved policy  from  the  approximate  value function  is  "closer" to  the  improved policy  from the  corresponding  exact value  function" (Lagoudakis  &amp; Parr, 2001).  More  forreally, this  would  mean that  the  difference  between  the backed-up  errors  using zr~+~  and another policy  zr

<!-- formula-not-decoded -->

is  small for  7r = 7r~.+~, the  greedy policy  w.r.t.  V ~. Since r~+~ is  unknown,  d~ would  need to  be small  for any policy  7r.  We  have

<!-- formula-not-decoded -->

Thus, there  are  two  possibilities: either  e~ belongs to the  intersection (for all r) of the  kernels  of  (P~+t P~),  in  which case  d~ is zero, or  if this is not  the case,  d~k is  also  unstable  whenever  the  eigenvalues of II~P ~  are  close  to 1/7. The first case,  which would be ideal (since  then,  7rk+~ would  be equal  to  7r~+l) does not  hold in  general.  Indeed, if it was true, this would  mean that  ek is  collinear  to  the  unit  vector  1 := (1 1 ... 1) T,  say ek = ekl for  some  scalar  e~ (then,  V~ would be  equal  to V ~ up  to  an  additive constant) and we would have ek = (Z --  7Hp~ P'~)ek  = ck  (I 7Hp~)l. But,  by  definition, ek  is  orthogonal  to  [0] w.r.t, the  inner product  (.,.)~ whereas the  vector

(I-TIIp~)l  is  not (for  q,  &lt; 1)  in  general (the  exception being if  1 is  orthogonal to  [¢]  w.r.t. (., .)p~).  Thus, as  soon as  the  eigenvalues  of Hp~P ~  are close  to  1/% the  approximation  error  ek as well  as the  difference  in the  backed-up  errors  d~k becomes  large.

Thus, we  believe that  in  general,  the  TD solution is  less stable  and predictable  (as  long as we  do not control  the eigenvalues  of  IIp~  p-k)  than  the  QR  solution. However,  the  TD  solution may  be preferable  in  model-free Reinforcement Learning,  when unbiased  estimators of A  and b  in  (4)  and (5)  need to  be derived from observed data  (Munos, 2003).

## 3.3.2. STEADY-STATE  DISTRIBUTION

If  we  consider the  case of the  steady-state  distribution and assume  that it  is  bounded  from below (for  example if all  policies  induce an irreducible, aperiodic Markov chain  (Puterman,  1994)), we are  able  to  derive following  error bound on the  loss  V*  -  V ~ .

Theorem  3 Assume  that Hypothesis  2  holds  for  a  distribution-fi (for example  uniform)  and a  constant that  Hypothesis  1 holds with the  steady-state  distributions  -fi,, and  that  -fi~  is bounded  from  below  by-~#l(with  ~ a constant),  then

<!-- formula-not-decoded -->

## 4. Conclusion

The  main  contribution  of this paper is  the  error  bounds on IIV*  -  V ~k  II, derived as  a function of the  approximation errors I  IVk- V'~II~ and the  Bellman  residuals I  IVk  -  T'~  Vkll~,k. The  distributions  #k and ~k indicate the  states that are  responsible  for  the  approximation accuracy. An  application  of this result  to  linear  function  approximation  is  derived  and error bounds  that  do not  depend  on the  number of states are  given,  provided that  the  MDP satisfies some  uniform stochasticity assumption  (that leads to  an upper-bound  for  #k  and ~)  and that  the  distribution used in  the  minimization problem  is lower-bounded  (in  order  to  insure  some  reliability of the  value approximation  uniformly  over the state-space, which secures  policy  improvement steps). In  the  case  of  the  QR solution, this  was guaranteed by using  a  somehow uniform mixed distribution, whereas in  the  case  of  the  TD  solution, we assumed  that the steady-state distribution was already  bounded from below.

## A.  Proof of  Theorem  1.

Define the approximation error: ek  =  Vk -  V ~, the  gain  between iteration k  and k  + 1:  g~ = V ~+1  V ~,  the  loss  of  using policy  Irk  instead  of  the  optimal policy: lk ----V*  V 'rk, and the  Bellman residual of  the  approximate value  function: bk = Vk --  T ~  Vk. Those  ek, gk,  lk, and  bk are column  vectors  of  size  N. We  first state  and prove the  following  results:

Lemma  2 It is true that:

<!-- formula-not-decoded -->

Proof:  Indeed,

<!-- formula-not-decoded -->

where we used the  fact  that T"Va -T~+ ~  Vk &lt;\_ 0 since ~rk+l is  greedy  with respect to Vk.  []

Lemma  3 It is true that:

<!-- formula-not-decoded -->

Proof:  Indeed,

<!-- formula-not-decoded -->

Lemxna 4 It is true  that:

<!-- formula-not-decoded -->

Or equivalently

<!-- formula-not-decoded -->

Proof: From Lemma  3, we have

<!-- formula-not-decoded -->

and (13)  follows from Lemma  2.  Inequality (12) derived  by factorizing (I7P ~)  and by noticing  that (I -  7P~)ek ---Vk -TM -T'~(Vk -"~) --V}  T ~  V~ -bk is  the  Bellman  residual  of  the  approximate function  Vk, which terminates  the  proof.  []

Now,  from Lemma 4,  we derive  the  following  results:

Corollary 1 We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

or equivalently that

<!-- formula-not-decoded -->

Proof: Write  fk  =  7[P~+'(I -  7P~+') -t -P~"  (I 7P~)-t]bk. Then,  from Lemma  4, lk+l &lt; ~[P~'lk  + fk.  By taking  the  limit superior  component-wise

<!-- formula-not-decoded -->

And  the  result follows  since I-TP ~" is  invertible. The proof of the  other inequality is  similar.  []

Corollary  2 By defining the  stochastic matrices  Qk and Qk as  in  Theorem  1,  we have

<!-- formula-not-decoded -->

where Ibkl and [ekl are vectors  whose  components  are Ibk(OI and lek(OI.

Proof:  First, the  fact that Qk  and Q~  are  stochastic matrices is  a consequence  of the  properties  that  if  P1 and P2 are  stochastic matrices, then P1P2, P2-~ z,  and (1-7) ([-TP1) -1  are stochastic  matrices too  (the  third property  resulting from the  two first and the  rewritv"  9'tP t~ The result follows ing of (I  -  7P1) -1  as  z.,t&gt;0 lJ. when  taking  the  absolute--value  in  the  inequalities of Corollary 1.  []

Now  we axe  able  to  prove  Theorem  1:

The fact that #~ and ~k are  distributions (positive vectors whose components sum to  one)  results from Qk and Qk being  stochastic matrices. Let  us  prove (2). For any vector  h,  define  2  t he vector whose components  are  h~.  We  have,  from the  convexity  of  the square  function  and from Corollary  2,

<!-- formula-not-decoded -->

Thus  limsupk~ Ill,~ll.  \_&lt; 04-q~y~ limsuPk--~oo Ilbkll.,.. Inequality (3)  is  deduced  similarly.

Remark  4 Some intuition about  these bounds may be perceived  in  a specific  case:  assume that  the  policy  r~ were to  converge, say to  ~,  and write  ~r the  approximation of  V ~. Then from Corollary  I,

<!-- formula-not-decoded -->

The  right  hand  side  of  this  inequality  measures the  expected difference between the  backed-up  approximation errors  using  ~ and 7r'with  respect  to  the  discounted future  state-distribution induced  by the  optimal policy. Thus here, the states  responsible for  the  approximation accuracy  are the  states  reached  by the  optimal policy as well as their  successors (for  policy ~).

## B. Proofs of Section 3

## Proof  of  Lern\_ma 1

First, for  two stochastic matrices  /'1  and P2 satisfying  (S), for all i and j, we have (P~P2)(i,j) Ek Pt (i, k)P2 (k, j) &lt; C~(j) Ek Pt  (i, k)  =  C~(j) recursively,  for all k,  (P1)k(i,j) &lt; C-fi(j). Thus  also (1 -- 'T)  (/-- ")'P,) --1 (i,  j)  = (I -- "7)  Et&gt;\_0 "7 t  (p1)t (i, \_&lt; c-~(i).

We  deduce  that Q~ defined in  Theorem  1  satisfies Vk(i,j) &lt;\_ C-fi(j). Thus, pk(j) = (PQk)(J) )-~i  tz(i)Q~(i,J) &lt;  C~(j) Y]~i p(i)  =  Cfi(j).

## Proof of  Theorem  2

Let  us  state and prove the  two Lemmas:

Lemma  5 Lower bounds for  p~.  and p5 k.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 6 Upper  bound  for IIIP'~lll, ~ and IllP'~lllpt. We have

<!-- formula-not-decoded -->

Proof:  First consider  pX. From  Hypothesis 2,

<!-- formula-not-decoded -->

Moreover,-fi = ~7xpX~(IAP ,~) 5  ~-~P~k, thus Ilh[l~= Ph ~  &lt;  ~  ,~xh2 \_ r-:-x,-k,o -- ~\_--~llhll~gTherefore, for

7rk 2 (7 an h,  lip  hll,~ \_&lt; ,\_-:~llhllp~. Now, it is  a~o true that

<!-- formula-not-decoded -->

Thus IIIP~"III~ \_&lt; min(~\_--~, ½).

Now  consider P~k" For  any  vector h,

<!-- formula-not-decoded -->

(where  we used  the property of  the steady distribution ~  =  ~kP'~). Moreover, ~  =  11\_-~(p6k --5~k), thus  Ilhll~-= '  2 iP=~hll~, \_&lt; ,\_-:~(llhll~ -~llhll~-,). Thus C(llhll~ - ~llhll~-,) +  &amp;llhll~, \_&lt; CIIhll~, since C  \_&gt; ~k h 2 Thus IIIP  IIIpg &lt;  c.

## Proof of Theorem 2"

For any distribution #, putting together (2), (7) and (6), we have limsup~lll~ll. &lt; U-~  lim  suPk-~c° ~-~ll~lllZ - 7P=~lll,..e ¯

## References

Baird, L.  C.  (1995). Residual  algorithms  :  Reinforcement learning with  function approximation. Machine Learning  :  proceedings  of  the  Twelfth  International Conference,

Bertsekas, D. P., &amp; Tsitsiklis, J. (1996). Neuro-dynamic programming. Athena Scientific.

Boyan, J. (1999).  Least-squares  temporal difference  learning. Proceedings of  the  16th  International Conference on Machine Learning, 49-56.

Bradtke,  S., &amp; Baxto,  A. (1996).  Linear  least-squares algorithms  for temporal  difference learning. Journal  of Machine  Learning,  2~, 33-57.

Gordon, G. (1995). Stable function approximation  in  dynamic programming. Proceedings  of  the International Conference on  Machine Learning.

Guestrin, C., Koller, D., &amp; Parr, 1%. (2001). Max-norm projections for  factored  mdps. Proceedings of  the  International  Joint  Conference  on Artificial Intelligence.

Judd, K.  (1998). Numerical  methods  in  economics. MIT Press.

Kakade, S., &amp; Langford,  J. (2002).  Approximately optimal approximate reinforcement  learning. Proceedings of  the 19th  International Conference on Machine Learning.

Koller,  D.,  &amp;  Parr,  1%.  (2000).  Policy iteration for  factored mdps. Proceedings of  the  16th  conference  on Uncertainty in  Artificial  Intelligence.

Lagoudakis, M.,  &amp; Parr, 1%. (2001). Model free leastsquares  policy  iteration. Technical  Report  GS-~001-05, Department of  Computer Science, Duke University.

Munos,  1%. (2003). Experiments in  policy iteration with linear approximation. Submitted  to  the  European Conference  on  Machine Learning.

Now, from  Lemmas  1, 5, 6, and  by  using the fact that IllZ-7P~lll~. -&lt;  1  +  7111P~lll~., we deduce the  bound in  I[" []#, but  since  this is  true for  any distribution #, the  same bound holds in II-I1~. []

## Proof of Theorem 3

For  any  distribution ~u,  let #k  = ~)k  with  ~)k  defined in Theorem 1.  Analogously  to (7) we have Ile~ll~ \_&lt; I1~11oo lle~ll~-.. Similarly to  Lemma 1,  we have #k  &lt; OF,  thus I1~11~ &lt;  ~¢-Since  ~k  is the steady-state distribution, ]]e~l]~, &lt;  ~\_~,  thus Ilekl]~, &lt; ~  ~-:i' and  from  (3),

<!-- formula-not-decoded -->

and  since this bound holds  for any  distribution #, it also holds in  max-norm. []

Munos,  1%.,  &amp;  Moore, A.  (2002).  Variable  resolution discretization in  optimal  control. Machine Learning Journal,  49, 291-323.

Puterman, M. L. (1994). Markov decision processes, discrete stochastic dynamic  programming. A WileyInterscience Publication.

Schoknecht, 1%.  (2002).  Optimality  of  reinforcement  learning  algorithms  with  linear function  approximation. Proceedings  of  the  15th Neural Information  Processing Systems conference.

Sutton,  R. S., &amp;  Barto,  A. G. (1998).  1%einforeement learning: An introduction. Bradford Book.

Tsitsiklis, J., &amp; Van Roy, B.  (1996).  An  analysis of  temporal  difference learning with  function  approximation. Technical  report LIDS-P-~3~$, MIT.