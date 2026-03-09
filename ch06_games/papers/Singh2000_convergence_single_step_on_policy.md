## Convergence  Results  for  Single-Step  On-Policy Reinforcement-Learning  Algorithms

| SATINDER SINGH                                                                                 | baveja@cs.colorado.edu    |
|------------------------------------------------------------------------------------------------|---------------------------|
| Department oj Computer Setcnce UniVfuity of Colorado Boulder, CO 80909.0430 TOMMI JAAKKOLA     | to:mm..i@cse.ucsc.eclu    |
| Compute?' Science Department UniVf1'3ity of California Santa Cruz, CA 95064 MICHAEL L. LITTMAN | llllittlllan@cs.duke.eciu |
| Department oj Computer Setence Duke Univer�ity Durham, NC 27708-0129 CSABA SZEPBSV.ARI         | szepes@sol.cc.u-szeged.hu |

Abstract. An  irnportffilt  application  of  reinforcement  learning (RL)  is to  finite-state  control problems and one of  the most difficult  problems  in  learning for  control is  balancing  the  explo­ ration / exploitation tradeoff.  Existing theoretical results for RL give very little guidance on rea.­ sonable ways  to  perform  exploration. In this  paper,  we  examine  the convergence of  single-step on-policy RL algoritlrrns for control.  On-policy algoritlrrns cannot separate exploration from learn­ ing  and  therefore must confront the exploration problem directly.  We prove convergence results for several related on-policy algoritlrrns with both decaying exploration and persistent exploration. We also provide examples of exploration strategies that can be followed during learning that result in convergence to both optimal values and optimal policies.

Keywords: reinforcement-learning, on-policy, convergence, Markov decision processes

## 1. Introduction

Most  reinforcement-learning (RL) algorithms ( Kaelbling  et  al., 1996; Sutton &amp; Barto, 1997) for  solving  discrete  optimal control  problems  use evaluation or value functions to cache the results of experience.  This is useful because close approxima­ tions to optimal value functions lead directly  to good control policies ( Williams &amp; Baird, 1993; Singh &amp; Vee, 1994).  Different RL algorithms combine new experience with old value functions to produce new and statistically  improved value fundions in  different  ways. All such algorithms face a tradeoff between exploitation and ex-

ploration  (Thrun,  1992;  Kumar &amp; Varaiya, 1986;  Dayan &amp; Sejnowski,  1996), i.e., between choosing actions that are best according to the current state of knowledge, and actions that are not the current best but improve the state of knowledge and potentially  yield higher payoffs in the future.

Following  Sutton  and  Barto  (1997),  we  distinguish  between  two  types  of  RL algorithms:  on-policy  and off-policy.  Off-policy  algorithms may update  estimated value  functions on the  basis  of hypothetical  actions,  i.e.,  actions other than those actually  executed-in  this  sense  Q-learning  (Watkins &amp; Dayan,  1992)  is  an  off­ policy algorithm.  On-policy algorithms, on the other hand, update value functions strictly  on  the basis  of the experience  gained from executing  some  (possibly  non­ stationary) policy.  This distinction  is  important because  off-policy  algorithms can (at least conceptually) separate exploration from control while on-policy algorithms cannot. More  precisely,  in  the  case  of  on-policy  algorithms,  a  convergence  proof requires more details of the exploration to be specified than for off-policy algorithms, since  the update rule depends a great deal on the adions taken by the system.

On-policy algorithms may prove to be important for several reasons.  The analogue of the on-policy/off-policy  distinction for RL prediction problems is the trajectory­ based/trajectory-free  distinction.  Trajectory-based algorithms appear  superior  to trajectory-free  algorithms  for prediction  when parameterized  function  approxima­ tors are used (Tsitsiklis &amp; Van Roy, 1996).  These results  carry over empirically  to the control case as  well  (Boyan &amp; Moore,  1995; Sutton, 1996).  In addition,  multi­ step prediction algorithms such as TD(.l.)(Sutton, 1988) are more flexible and data efficient  than  single-step  algorithms  (TD(O)),  and  most  natural  multi-step  algo­ rithms for control are on-policy.  These observations suggest that on-policy  control algorithms are important and worthy of study.

In  this  paper,  we  examine  the  convergence  of  single-step  (value  updates  based on  the  value  of  the  "next"  timestep  only),  on-policy  RL  algorithms  for  control. We do not address either  fundion approximation or multi-step algorithms; this is the  subject of  our  ongoing research. Earlier  work  has  shown  that  there  are  off­ policy  RL algorithms that converge to optimal value functions (Watkins &amp; Dayan, 1992;  Dayan,  1992;  Jaakkola  et  aI.,  1994;  Tsitsiklis,  1994;  Gullapalli &amp; Barto, 1994; Littman &amp; Szepesvari, 1996); we prove convergence results for several related on-policy  algorithms. We also  provide examples  of  poliejes  that  can  be followed during  learning  that  result  in  convergence  to  both  optimal  values  and  optimal policies. These  results  generalize  naturally  to  off-policy  algorithms,  such  as  Q­ learning, showing the convergence of many RL algorithms to optimal policies.

## 2. Solving  Markov  Decision  Problems

Markov decision  processes ( MDPS ) are  widely  used  to model controlled  dynamical systems in control theory, operations research and artificial intelligence (Puterman, 1994; Bertsekas,  1995; Barto et al.,  1995).  Let S = 1, 2, ... , N denote the discrete set of states of the system,  and let A be the discrete set of actions available to the system.  The  probability  of  making a transition  from state s to  state s ' on action

a is  denoted P:�I and the  random payoff associated  with  that transition  is  denoted T ( S,  a ) . A policy  maps  each  state  to  a  probability  distribution  over  actions-this mapping  can  be  invariant  over  time  (stationary)  or  change  as  a  function  of  the interaction  history  (non-stationary). For  any  policy 1r, we  define  a  value  function V�(s) = E,U::::: O 'Y'T,lso = s}, which  is  the  expected  value  of the  infinite-horizon sum  of the discounted  payoffs  when  the  system  is  started  in  state s and the policy 11' is  followed  forever. Note  that T, and s, are  the  payoff and  state  respectively at  timestep t, and (Tt,  sd is  a  (non-stationary)  Markov  proc.ess  with  transition probabilities  given  by  the  rules  that T, is  distributed  as T( 8" a,) and the probability that St+l = S is P�t�. Here, at is  the  action  taken  by  the system  at  timestep t. The discount  factor,  0  S; l' &lt;  1, makes  payoffs  in  the  future  less  valuable  than  more immediate  payoffs.

The  solution  of  an  MDP  is  an  optimal  policy 1r* that  simultaneously  maximizes the  value V1\""  (s) of  every  state s E  s. It  is  known  that  a  stationary  deterministic optimal  policy  exists  for  every  MDP (d. Bertsekas'  textbook, 1995). Hereafter, unless  explicitly  noted, all policies  are assumed to  be stationary.  The value function associated  with 1T"* is  denoted V*. Often it is convenient to associate values not with states but with state-action pairs, called  Q values as in Watkins' (1989) Q-Iearning: Q�(s, a) = R(s,  a) + 'YE{V�(SI)}, and Q'(s,  a) = R(s, a) + 'YE{V'(SI)}, where S ' is  the  random  next  state  on  executing  action a in  state s, and R(s,  a) is  expected value  of T(S,  a ) . Clearly, ll"(s) = argmaxa Q'(s, a), and V'(s) = max. Q'(s, a). The  optimal Q  values  satisfy  the  recursive  Bellman  optimality  equations  (Bellman, 1957 ) , '18, a:

<!-- formula-not-decoded -->

In  reinforcement  learning,  the  quantities  that  define  the  MDP, P and R, are  not known  in  advance. A  RL algorithm  must  find  an  optimal  policy  by  interacting with  the  MDP  directly;  because  effective  learning  typically  requires  the  algorithm to  revisit  every  state  many  times,  we  assume  the  MDP  is  "communicating"  (every state  can  be reached  from  every  other  state).

## 2.1. Off-Policy  and  On-Policy  Algorithms

Most RL algorithms  for  solving  MDPs  are  iterative,  producing  a  sequence  of  esti­ mates  of  either  the  optimal  (Q-)value  function  or  the  optimal  policy  or  both  by repeatedly  combining  old  estimates  with  the  results  of  a  new  trial  to  produce  new estimates.

A RL algorithm can be decomposed into two components.  The learning policy is  a non-stationary policy  that maps  experience  (states  visited,  actions  chosen,  rewards received)  into a current  choice  of action.  The update rule is  how  the  algorithm  uses experience  to change  its  estimate  of  the  optimal  value  function.

In  an  off-poliey  algorithm,  the  update  rule  need  not  have  any  relation  to  the learning policy.  Q-Iearning  (Watkins, 1989 ) is an  off-policy algorithm that estimates

the  optimal  Q-value  function  as  follows:

<!-- formula-not-decoded -->

where Qt is  the  estimate  at  the  beginning  of  the tth timestep,  and Btl at ,  Ttl and at are  the  state,  action,  reward,  and  step size  (learning  rate)  at  timestep t. This is  an  off-line  algorithm  as  the  update  of Qt(St,at) depends  on  m a x, ( Qt(St+ l , b )) , which  relies  on comparing various  "hypothetical"  actions b. The eonvergence  of the Q-learning  algorithm  does  not  put  any  strong  requirements  on  the  learning  policy other  than  that  every  action  is  experieneed  in  every  state  infinitely  often. This can  be  accomplished,  for  example,  using  the  random-walk  learning  policy,  which chooses  actions  uniformly  at  random. Later,  we  describe  several  other  learning policies  that result  in  c.onvergence  when  combined  with the Q-Iearning update  rule.

The update rule for SARSA(O)(Rummery, 1994; Rummery &amp; Niranjan, 1994; John, 1994, 1995; Singh &amp; Sutton, 1995; Sutton,  1996 ) is  quite similar to Q-learning:

<!-- formula-not-decoded -->

The  mam  difference  is  that  Q-Iearning  makes  an  update  based  on  the  greedy  Q value of  the  successor  state, 8t+l, while SARSA(O) l  uses  the  Q  value  of  the  action at+l actually  chosen  by  the  learning  policy. This  makes  SARSA(O ) an  on-policy algorithm, and therefore its  conditions  for convergence  depend  a  great deal  on  the learning  policy.  In particular, because S AR S A ( O ) learns  the  value  of  its  own actions, the  Q  values  can  converge  to  optimality  in  the  limit  only  if  the  learning  policy chooses actions optimally in the limit.  Section 3 provides some positive convergence results  for  two significant  classes  of learning  policies.

Under  a greedy  learning  policy  (i.e.,  always  select  the  action  that  is  best  accord­ ing  to  the  current  estimate),  the  update  rules  for  Q-Iearning  and  SARSA ( O ) are identical. The  resulting RL algorithm  would  not  converge  to optimal  solutions,  in general, because the need for infinite exploration would not be satisfied.  This helps illustrate  the  tension  between adequate exploration  and exploitation  with regard to convergence  to optimality.

## 2.2. Learning  Policies

A learning  policy  selects  an  action  at  timestep t as  a  function  of  its  experience history. In  this  paper,  we  consider  several  learning  policies  that  make  decisions based  on  a  summary  of  history  consisting  of  the  current  timestep t, the  current state s, and the current estimate Q of the optimal Q-value function.  Such a learning policy  can  be  expressed  as P r (a l s , t, Q), the  probability  that  action a is  selected given  the  history.

We  divide  learning  policies  for  MDPs  into  two  broad  categories;  a decaying ezplo­ ration learning  policy  becomes  more  and  more  like  the  greedy  learning  policy  over

time,  a persistent exploration learning  policy  does  not.  The  advantage  of  decaying exploration  policies  is  that  the  actions  taken  by  the  system  may  converge  to  the optimal  ones  eventually,  but  with  the  price  that their  ability  to adapt  slows  down. In  contrast to this, persistent exploration learning policies  can retain their  adaptiv­ ity  forever,  but  with  the  price  that  the  adions  of  the  system  will  not  converge  to optimality in the standard sense.  We prove the convergence of  SARSA ( O ) to optimal policies  in  the  standard  sense  for  a  class  of decaying  exploration  learning  policies, and  to  optimal  policies  in  a  special  sense  defined  below  for  a  class  of  persistent exploration  learning  policies.

Consider  the  class  of decaying  exploration  learning  policies  characterized  by  the following  two  properties:

1. eaeh adion is  visited infinitely often in every state that is  visited infinitely  often,
2. in  the  limit,  the  learning  policy  is  greedy  with  resped  to the  Q-value  fundion with  probability  1;

we  label  learning  policies  satisfying  the  above  conditions  as GLIE, which  stands for  "greedy  in  the  limit  with  infinite  exploration." An  example  of  such  a  learning policy  is  certain  forms  of  Boltzmann  exploration:

<!-- formula-not-decoded -->

where P't(s) is  the  state-specific  exploration  coefficient  for  time t, which  controls the  rate  of  exploration  in  the  learning  policy. To  meet  condition 2 above,  we would  like  P't  to  be  infinite  in  the  limit,  while  to  meet  condition  1  above  we  would like f3 t to  not  approach  infinity  too  fast. In  Appendix  A,  we  show  that f3 t ( 8) = In Th,(s)ICt(s) satisfies  the  above  requirements ( where Th'(s) '" t is  the  number  of times  state s has been  visited  in  the t timesteps,  and Ct(s) is  defined  in  Appendix A ) . Another example of a G  LIE learning policy is  some forms of t;-greedy exploration ( Sutton, 1996) ' which  at  timestep t in  state s picks  a  random  exploration  action with probability &lt;t(s) and the greedy action with probability l-&lt;,(s). I n Appendix A,  we  show that if 't(8) = clnt(s) for  0 &lt; c &lt; 1, then  .-greedy exploration  is GLIE.

We  also  analyze  "restricted  rank-based  randomized  learning  policies" (RRR), a class  of  persistent  exploration  learning  policies  commonly  used  in  practice. An RRR learning  policy seleds  adions  probabilistically  according to the  ranks  of their Q  values,  choosing  the  greedy  action  with  the  highest  probability  and  the  action with  the  lowest  Q  value  with  the  lowest  probability.  Different  learning  policies  can be  specified  by  different  choices  of  the  function T : {I, ... , m} ---+ 1R that  maps action  ranks  to  probabilities. Here, m is  the  number  of  actions. For  consistency, we  require  that T( l) 2' T( 2) 2'  ...  2' T( m) and 2::::, T( i) = 1. At  timestep t, the RRR learning  policy  chooses  an  action  by  first  ranking  the  available  actions aceording  to  the  Q  values  assigned  by  the  current  Q-value  fundion  Qt  for  the current  state St. We use the  notation p( Q, s, a ) to be the  rank of action a in  state s

based on Q{s,.) (e.g.,  if p{Q,s,a) = 1 then a = argm ax, Q{s, b )) ,  with ties broken arbitrarily. Once  the  actions  are  ranked,  the ith ranking  action  is  chosen  with probability T{i); that  is,  action a is  chosen  with  probability T{p{Q,  .,a)). The RRR learning policy is  "restricted"  in that it does not directly  choose actions-it simply assigns probabilities to actions according to their ranks.  Therefore, an RRR learning policy has the form Pr (a l s , t, Q) = T(p(Q"  s, a )) .

To  illustrate  the  use  of  the T function,  we  specify  three  well-known  learning policies  as  RRR learning policies  by the appropriate definition of T. The random­ walk learning poliey chooses aetion a in  state s with probability l!m. To achieve this  behavior  with  the  RRR  learning  policy,  simply  define T{i) = 11m for  all i; actions  will  be  equally  likely  regardless  of their  rank.  The  greedy  learning  policy can be specified by T{l) = 1, T(i) = 0 when 1 &lt; i:S m; it deterministically  seleels the action with the highest Q value.  Similarly, .-greedy exploration can be specified by defining T( 1) = 1 -E + Elm,  T(  i) = Elm, 1 &lt; i :S m. This policy  takes  the greedy action with probability 1 -€ and a random action otherwise.  To satisfy the condition that T(l) 2: T (2 ) 2: ... 2: T(m), we require that 0 :S ' :S  1.

Another commonly used persistent exploration  learning  policy is Boltzmann ex­ ploration  with a  fixed exploration  parameter. Note  there  is  no  choice  of T that specifies  Boltzmann  exploration;  Boltzmann  exploration  is  not  an  RRR  learning policy as the probability of choosing an adion depends on the adual Q values and not only on the ranks of actions in Q(.) .

## 3. Results

Below we prove results on the convergence ofsARsA(O) under the two separate cases of  G  LIE and RRR learning policies.

## 3.1. Convergence of SARSA ( O ) under  GLIE  Learning Policies

To ensure the convergence  of SARSA ( O ) , we  require  a  lookup-table  representation for the Q values and infinite visits to every state-action pair, just as for Q-Iearning. Unlike  Q-Iearning,  however, SARSA ( O ) is  an  on-policy  algorithm  and,  in  order  to achieve its convergence to optimality,  we have to further assume that the learning policy becomes greedy in the limit.

To state these assumptions and the resulting convergence more formally, we note first that due to the dependence on the learning policy, SARSA ( O ) does not directly fall  under  the  previously  published  convergence  theorems  (Dayan &amp; Sejnowski, 1994; Jaakkola et al.,  1994; Tsitsikli5,  1994;  Szepesvari &amp; Littman,  1996).  Only a slight  extension  is needed,  however,  and this  is  presented  in the form of Lemma 1 below (extending Theorem 1 of Jaakkola et aI.,  1994, and Lemma 12 of Szepesvari &amp; Littman, 1996). For c.larity,  we will not present the lemma in full generality.

<!-- formula-not-decoded -->

converges to zero with probability  one  (w.p.1) if the  following properties hold:

1.  the  set  of possible  states X is finite.
2. 0 &lt;:: "t ( � ) &lt;:: 1, Lt "t ( � ) = 00, Lt ,, ;( � ) &lt;  00 w.p.l,  where  the  probability  is ot'er the  learning rates at.
3. IIE{F,(·)IPt}llw &lt;:: 1&lt; 11 .a.t llw  +Ct, where I&lt;  E [0, 1) and Ct converges to zero w.p.i.
4. Var{Ft(�)IP,} &lt;:: K(1 + II .a.t llw) ', where K is ,orne con,tant.

Here Pt is  an  increasing  sequence  of u-fields that  includes  the  past of the process. In  particular,  we assume that "t, .a.t, Ft-1 E Pt. The notation 11 · llw refers to some (fixed) weighted mazimum norm.

Let  us  first  clarify  how  this  lemma  relates  to  the  learning  algorithms  that  are the  focus  of this  paper.  The  sequence  of  visited  states St and  selected  actions at are captured by defining the learning rates at in the following way.  We can define �t = ( St, at) and  further  require  that  O:t ( .-e ) = 0 whenever 2 -# �t. With  these definitions,  the  iterative  process reduces  to

<!-- formula-not-decoded -->

which resembles more closely the updates of the on-line algorithms such as SARSA ( O ) ( Equation 3). Also, note that the lemma shows the convergence of.1. to zero rather than to some non-zero optimal values.  The intended meaning of.1. is Q t -Q*, i.e., the difference  between the current  Q values, Qt, and the target Q values, Q*, that are attained asymptotically.

The  extension  provided  by  our  formulation  of  the  lemma  is  the  fact  that  the contraction  property ( the third condition ) need not be strict;  strict  contraction is now required to hold only asymptotically.  This relaxation makes the theorem more widely applieable.

Proof: While  we  have  stated  that  the  lemma  extends  previous  results  such  as the Theorem 1 of Jaakkola et al. (1994) and Lemma 12 of  Szepesvari &amp; Littman (1996), the  proof  of  our  lemma is,  however,  already  almost fully  contained  in  the proofs of these results ( requiring only minor, largely notational changes ) .  We, thus, refrain from repeating that proof here. ·

We can now use Lemma 1 to show the convergence of SARSA ( O ) .

THEOREM  1  In  finite state-action MDPS,  the Qt  values computed  by  the SARSA ( O ) rule  (see  Equation 3) converge  to  Q*  and  the  learning  policy 1r' t cont'erges  to  an optimal  policy".'  if  the learning  policy  is  GLIE,  the  conditions  on  the  immediate reward8 and state transitions listed in Section 2 hold and if the following additional conditions are satisfied:

1. The Q values are stored in  a lookup table.
2. The learning rales salisfyO:S at(s,a):s 1, Ltat(s,a) = 00 and Ltar(s,a) &lt; 00 and at(s,a) = 0 unless (s,a) = ( s t, a ,) .
3. Var{r(s,a)} &lt;  00.

Proof: The correspondence  to Lemma 1 follows from associating X with the set of state-action  pairs (s,a), at(�) with at(s,a) and Ll.t(s,a) with Qt(s,a)-Q'(s,a). It follows that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

where FtQ would be the corresponding Ft in Lemma 1 if the algorithm under consid­ eration were Q-leaming.  Further, we define F,( s, a ) = C,( s,  a) = 0 if (s,  a) # (St,  at) and denote the O"-field generated by the random variables {Stl  atl  at! rt-b"" 81, aI, aI, Qo} by Pt. Note that Qt,  Qt-l,  . ..  , Qo are Pt-measurable and, thus, both �t and Ft-1 are Prmeasurable, satisfying the measurability conditions of Lemma 1.

It is  well-known  that  for Q-leaming II E { F , Q ( - , . )  I  P,}II :S 1 ' II LI., II for  all t, where II  .  II is  the  maximum  norm. In  other  words,  the  expeded  update  operator  is  a contraction  mapping.  The only  difference  between  the  current F, and F,Q for  Q­ learning is the presence  of Ct. Therefore,

<!-- formula-not-decoded -->

Identifying c,  = IIE{C,{-,·)  I P,} II in  Lemma 1, we  are  left  with  showing  that Ct converges  to  zero  w.p.I. This,  however,  follows ( 1) from our assumption  of a GLIE policy (i.e., that non-greedy actions are chosen with vanishing probabilities), ( 2) the  assumption  of  finiteness  of the MDP, and (3) the  fact  that Qt(s,  a) stays bounded during  learning. To  verify  the  boundedness  property,  we  note  that  the SARSA(O) Q values can be upper bounded by the Q values of a Q-learning process that updates exactly the same state-action pairs in the same order as the SARSA ( O ) process. Similarly,  the SARSA ( O ) Q  values  are lower bounded  by  the  Q  values of  a  Q-Iearning  process  that  uses  a  min  instead  of  a  max  in  the  update  rule (d. Equation 2) and updates exactly the same state-action pairs in the same order as

the SARSA ( O ) process.  Both the lower-bounding and the upper-bounding Q-learning processes  are convergent and have bounded Q  values.

The  condition  on  the  variance  of Ft follows  from  the  similar  property  of Ft Q .

•

Note that if a GLIE learning policy is used with the Q-Iearning update rule, one gets convergence to both the optimal Q-value function and an optimal policy.  This begins to address a significant outstanding question in the theory of reinforeement learning:  How do  you a learn  a policy  that achieves  high  reward in  the  limit and during  learning? Previous  convergence  results  for  Q-Iearning  guarantee  that  the optimal  Q-value  function  is  reached  in  the  limit;  this  is  important  because  the longer the  learning  process  goes  on,  the  closer  to optimal  the  greedy  policy  with respect  to  the  learned  Q-value  function  will  be.  However,  this  provides  no  useful guidance for selecting actions during learning.  Our results, in contrast, show that it is possible to follow a policy during learning that approaches optimality over time.

The properties of G  LIE policies imply that for any RL algorithm that converges to the optimal value function and whose estimates stay bounded (e.g., Q-learning, and ARTDP of Barto et al., 1995), using GLIE learning policies will ensure a c.oncurrent convergence to an optimal policy.  However, to get an implementable RL algorithm, one still has to specify a suitable learning policy that guarantees that every action is attempted in every state infinitely  often (i.e., 2.:, "'t(s, a) = (0 ) . In the Appendix, we prove that, if the probability of choosing any particular action in any given state sums up to infinity, then the above condition is indeed satisfied.  To illustrate  this, we  derive  two learning strategies that are  GLIE.

## 3.2. Convergence  of SARSA ( 0 ) under RRR Learning  Policies

This section proves two separate results concerning a class of persistent exploration learning  policies: (1)  the SARSA ( O ) update rule  combined  with  an  RRR learning policy converges to a well-defined Q-value function and policy, and (2) the resulting policy is optimal, in a sense we will  define.

As mentioned earlier,  an RRR learning policy chooses actions probabilistically  by their ranking ac.eording to the eurrent Q-value function; a specific learning polic.y is specified by the function T, a probability distribution over action ranks.  A restricted policy Ji"  :  S --&gt; II(A, {l, . . . ,  m }) ranks actions in each state (recall that m denotes the  number  of  adions),  i.e.,  -rr(s)  is  a  bijedion  between A and  {1, . . .   ,m}. For convenience,  we use the notation -rr( s, a ) to denote the  assigned  rank of action a in state s, i.e., to denote -rr(s){a).  The mapping -1r represents a policy in the sense that an agent following restricted policy K from state s chooses action a with probability T( Ji"( s, a ll , the probability  of the rank assigned by Ji" to action a in  state s.

Consider what happens when the SARSA ( O ) update rule is used to learn the value of  a  fixed  restricted  policy K. Standard  convergence  results  for  Q-Iearning  can easily  be used to show that the Qt values will converge to the  Q-value function of K. Specifically, Qt will converge to Qtr, defined as  the unique solution to

<!-- formula-not-decoded -->

When  an  RRR  learning  policy  is  followed,  the  situation  becomes  a  bit  more complex.  Upon entering state s, the probability that the learning policy will choose, for example, the rank 1 action is fixed at T(l); however, the identity of that action changes as a function of the current Q-value function estimate Qth . ) . The natural extension  of  Equation  6  to  an  RRR  learning  policy  would  be  for  the  target  of convergence  of Qt in SARSA ( O ) to  be

<!-- formula-not-decoded -->

Recall that p(Q,  s/,a') represents the rank af adian a' according to the Q values Q of state S'. The only change between Equation 6 and Equation 7 is  that the  latter uses  an  assignment  of  ranks  that  is  based  upon  the  reeursively  defined  Q-value function Q 1 whereas the former uses a fixed assignment of ranks.  Using the theory of generalized  MDPs(Szepesvari &amp; Littman, 1996), we can show that this difference is  not  important from the  perspedive  of  proving  the  existence  and  uniqueness  of the solution of Equation 7.

Define

<!-- formula-not-decoded -->

now Equation 7 can be rewritten

<!-- formula-not-decoded -->

As long as ® satisfies the non-expansion property that

<!-- formula-not-decoded -->

for all Q-value functions Q and Q' and all states s, then Equation 9 has a solution and it is unique  (Szepesvari &amp; Littman, 1996).  The non-expansion property of I8l can be verified by the following argument.

- Consider a family of operators 18l:  Q(  s,  a) = ith largest  value of Q(  s,  a) for each 1 ::; i ::; m. These are all non-expansions.
- Define I8l:Q(s,a) = LiT(i) I8l�Q(s,a); it  is  a non-expansion as long as  every 18l: is  and T is a fixed probability distribution.
- It  is  dear  that I8l:Q(8,a) = 18l. Q(8,a) as  defined  in  Equation  8,  so I8l is  a non-expansion also.

Therefore, Q exists  and is  unique.  We  next  show that Q is,  in  fact,  the  target of convergence  for SARSA(O).

THEOREM  2 In finite state-action MDPS, the Q, values computed by the SARSA(O) rule (see Equation 3) converge to Q  and the learning policy 1ft converges to a re­ stricted optimal policy it if the learning policy is RRR, the conditions on the im­ mediate  rewards and state transitions listed in Section 2 hold,  and if the following additional conditions are satisfied:

1. Pr(at+1 = a | Qt, St+1) = T(p(Qt, St+1, 01+1)).
2. The Q values are stored in a lookup table.
3. The learning rates satisfy 0 &lt;:; ", , (8,a) &lt;:;  1, L,,,,,(8,a) = 00, L,"';(8,a) &lt; 00, and ",(s,a) = 0 unless (s,a) = (s"a,).
4. Var{r(s,a)} &lt; 00.

Proof: The result  readily  follows  from  Lemma  1 (or  Theorem 1 Jaakkola  et aI., 1994) and the proof  follows  nearly  identical  lines  as  that  of  Theorem 3.1. First, we  associate X (of  Lemma 1) with the  set  of  state-action  pairs (s,  a),  at{;l) with ",(s, a), but here  we  set Ll.,(s, a )  = Q,(s, a ) -Q(s,  a). Again,  it  follows  that

<!-- formula-not-decoded -->

where now

<!-- formula-not-decoded -->

Further,  we  define F,{s,a) = C,{s,a) = 0 if (8,a) # (s"a,) and denote the (J'­ field generated  by the random  variables {St, at,  at, Pt-I, ... , Sl, a!, al, Q o } by Pt. Note  that Qt, Qt-I, ... , Qo are  Pt-measurable  and,  thus,  both �t and  Ft-l  are Prmeasurable, satisfying the measurability  conditions of Lemma l .

Substituting  the right-hand  side of Equation 7 for Q( s" a,) in  the  definition  of F, together  with the  properties  of sampling Pt,  St+l and at+l  yields  that

<!-- formula-not-decoded -->

where in the first equation we have exploited the fact that E{1'tISt,  at} = R(st,atl, in the second equation that Pr(St+1 I St, at) = P:",,+, and that P r (at +1 = a  I  Q t, St + d = T(p(Qt,  't+1,  all (condition  1),  whereas  the  inequality  comes  from  the  properties of  rank-based  averaging  (see  Lemma 7 and  Theorems 9 and 10 of  Szepesvari &amp; Littman ' s (1996) technical report.  Finally, it is not hard to prove that the varianc.e of Ft given the past Pt satisfies condition 4 and, therefore, we do not include it here  .

•

We have shown that SARSA ( O ) with an RRR learning policy converges to Q. Next, we show that Q is,  in a sense,  an optimal  Q-value function.

An optimal restricted policy is one that has the highest expected total discounted reward  of all restricted  policies. The greedy restricted policy for a Q-value function Q  is ft(s,  a) = p(Q, 5, a); it  assigns  each  action  the  rank  of  its  corresponding  Q value.  Note that this is the policy dictated by the RRR learning policy for a fixed Q-value function Q.

The  greedy restricted  policy  for  Q'  (the optimal Q-value function of the MDP ) is  not  an  optimal  restricted  policy  in  general, so  the  Q-learning  rule  in  Equation 2 does  not find an optimal restricted policy.  However 1 the next theorem shows that the greedy restricted policy for Q (Equation 7) is an optimal restricted policy.  This Q function is very similar to Q* 1 except that actions are weighted according to the greedy restricted policy instead of the standard greedy policy.

THEOREM 3 The greedy restricted policy with respect to Q is an optimal restricted policy.

Proof: We  construct  an  alternate MDP so  that  every  restricted policy  in  the original MDP is  in  one-to-one  correspondence  with  (and  has  the  same  value  as) a  deterministic  stationary  policy  in  the  alternate MDP. Note  that,  as  a  result of  the  equivalence  of  value  functions,  the  optimal  policy  of the  alternate MDP will eorrespond to an optimal restricted policy of the original MDP (the restricted policy that achieves the best values for each of the states) and, thus, the theorem will foll ow if  we show that the optimal policy in the alternate MDP corresponds to the greedy restricted  policy  with respect to Q.

The alternate MDP is defined by  (5,.iI, H, P, ")' ). Its  action space is.il = II(A, {l, .. . , m}), i.e.,  it  is  the set of all bijections from A to {I,  . . .   ,  m}.  The rewards are

<!-- formula-not-decoded -->

and the transition  probabilities  are given by p�! = 2: a EA T(p.(a))p$aJ!. Here, J1. is an  element  of A. One  can  readily  check  that the  value of a restricted  policy 1r is just  the  value  of Jt in  the  alternate MDP.

The value of the greedy restricted  policy with respect to Q in the original MDP is

<!-- formula-not-decoded -->

Substituting the definition of Q from Equation 7 into Equation 10 results in

<!-- formula-not-decoded -->

Using Equation 10 once again, we find that iT satisfies  the recurrence  equation

<!-- formula-not-decoded -->

Meanwhile,  the optimum value of the alternate MDP satisfies

<!-- formula-not-decoded -->

The highest  value permutation is the one that assigns the highest probabilities to the actions  with the  highest  Q  values  and the  lowest  probabilities  to  the  actions with the lowest Q values.  Therefore, the recurrenee  in  Equation  12 is the same as that in Equation 11, so, by  uniqueness, V* = V. This means the greedy restricted policy  with respect to Q is the optimal restricted policy. ·

As a corollary of Theorem 3.2, given a communicating MDP and a RL algorithm that follows an RRR learning policy specified by T where T(  i) &gt; 0 for all 1 ::; i ::; m, SARSA(O) converges to an optimal restricted policy.

We  conjecture  that  the  same  result  does  not  hold for  persistent  Boltzmann  ex­ ploration  beeause  related  synchronous  algorithms  do not have  a unique  target  of convergence  (Littman,  1996).

## 4. Conclusion

In this paper, we have provided convergence results for SARSA ( O ) under two different learning  policy  classes;  one  ensures  optimal  behavior  in  the  limit  and  the  other ensures  behavior  optimal  with  respect  to  constraints  imposed  by  the  exploration strategy. To the best of our knowledge, these constitute the first convergence results for  any on-policy  algorithm. However,  these  are  very  basic  results  because  they apply only  to the lookup-table  case,  and more  importantly  because  they  do  not seem to extend naturally to general multi-step on-policy  algorithms.

## Acknowledgments

We thank Rich Sutton for help and encouragement.

## Appendix  A

## G LIE  Learning  Policies

Here,  we  present  conditions  on  the  exploration  parameter  in the commonly  used Boltzmann exploration and E-greedy exploration strategies to ensure that both in­ finite  ex p lora t i o n and greedy  in the limit conditions are  satisfied.

In a eommunieating MDP, every  state gets visited infinitely  often as long as each action  is  chosen  infinitely  often  in  each  state  (this  is  a  consequence  of  the  Borel­ Cantelli  Lemma ( Breiman, 1992); all  we  have  to  ensure  is that in  each state  each action  gets  chosen  infinitely  often  in  the  limit. Consider  some state s. Let t$  ( i) represent  the  timestep  at  which  the i th visit  to  state s occurs. Consider  some action a. The probability  with  which action a is  exeeuted  at the i t � visit  to state s is  denoted Pr(als,t,(i)) ( i.e, Pr(a = a,ls, = s,t,(i) = t)).

We  would like  to  show  that if  the  snm of the  probabilities  with  which  action a is  ehosen is  infinite, i.e.,  2...:�1 Pr(als, t,(i)) =  00, then the number of times action a gets  executed  in  state s is  infinite  w.p.l. This  would  follow  directly  from  the Borel-Cantelli Lemma  if the  probabilities of  selecting  action a at  the  different i were  independent.  Unfortunately,  in  this  case the  random choice  of action at the i t h vi s i t to  state s affects  the probabilities  at the i + 1  I t visit to state s ( through the  evolution  of  the  Q-value  f unction).  However,  if there  exists  another  stochastic process  that  also sums  to  infinity,  lower  bounds  the  sequenee  of  probabilities  of selecting  action a, and satisfies  the  independence  conditions  required by the  Borel­ Cantelli  Lemma,  then  again  the  result  would  follow. We state  this  below  more formally.

OBSERVATION  1 Consider a stochastic  process {Pi}�' with 0 -::: Pi -::: 1 for all i. Let random variable Si be 1 with probability Pi and 0 with probability 1 -Pi. Further, let N(n) = 2...:�:::: 1 S i. If there exists  another  stochastic process {Ci}�l such that 0-::: Ci -::: Pi for all i,  �;:, Ci =  00, and all finite  subsets of {Ci} are independent, then limn\_= N(n) =  00 w.p.1.

Proof: Let Si be e qua l to 1 with probability Ci and 0 with probability  l-ci, then the  Borel-Cantelli  Lemma proves  that limn\_= N(n) =  00 w.p.!.,  where N(n) n · 2...:i:::: 1 Si· However, since Pi 2: Ci, the  result mnst also follow for N(n). ·

## A.I. Boltzmann Exploration

In Boltzmann exploration,

<!-- formula-not-decoded -->

where f3t(s) is the state-specific  exploration coefficient for time t. Let the number of  visits  to state s in  timestep t be denoted as n,ls) and assume  that .(s,  a) has a finite  range.  We know that L:�1 eli = 00; therefore,  to  meet the  conditions  of Observation A, we will  ensure  that for all actions a E  A, Pr(als, t.(i))  2: c/i ( with c :S  1 ) .  To do that we need for all a:

<!-- formula-not-decoded -->

where bme,x = argmax bEA Qt{s, b) above and m is the number of actions.  Further, let c = l / m .  Taken together, this means that we want I't(s) :S I n nt(s)/Ct(s) where Ct(s) = max..IQt(s, b=ax) -Qt(s,  a)l. Note that C,(s) is  bounded  because  the  Q values remain bounded.

It should also be clear that for every s, lim,\_oo n,ls) =  00, and therefore

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

this means that Boltzmann exploration with I't(s) = I n n,(s)/C,(s) will be greedy in the  limit.

## A.2. e-Greedy Exploration

In €-greedy exploration we pick a random exploration action with probability €t{s) and the greedy action with probability l- et(s). Let et(s) = c/nt(s) with 0 &lt; c &lt; 1. Then, Pr(als, t,(i)) 2: et(s)/m, where m is  the  number  of  actions. Therefore, Observation  A  combined  with  the  fact  that L:�1 eli = 00 implies  that  for  all s, L:�1 Pr(als, t,(i)) =  00. Further,  for  all s , limt----&gt;oo nt(s) =  00, and, therefore, lim t ----&gt; 00 Et(S) = 0, ensuring that the learning policy is greedy in the limit.  Therefore, if et(s) = c/nt(s) then e-greedy exploration is  GLIE for 0 &lt; c &lt; 1.

## Notes

1. The name is  a reference to the fact  that it is a sing l e-step a l gorithm that makes updates on the ba.sis of a sta.te, Action, Reward, state, Action 5tuple.

## References

- Andrew G. Barto,  S.  J. Bradtke, and  Satinder Singh  (1995). Learning  to  act  using �al-time dynamic programming. Artificial Intelliyence, 72(1  ):81-138.
- Richard Belhnan (1957). Dynamic  Programming. Princeton University Press, Princeton, NJ.
- Dimitri P.  Bertsekas (1995). Dynamic  Programming  and  Optimal  Control. Athena Scientific, Belmont, Massachusetts.  Volumes 1 and 2.
- Justin A. Boyan and Andrew W. Moore.(1995).  Generalization in  reinforcement learning:  Safely approximating the value function. In G. Tesauro, D.  S. Touretzky,  and  T.  K.  Leen,  editors, Advances  in  Neural  Information  Processing  Systems 7, pages  369-376,  Cambridge, MA.  The MIT Press.
- Leo Breiman.(1992). Probability. S oc i ety for Industrial and Applied Mathematics, Philadelphia, Pennsylvania.
- Peter Dayan (1992).  The convergence of TD('\) for general.\. Machine  Learning, 8(3):341-362. Peter Dayan and Terrence J.  Sejnowski (1994). TD(.\) converges with probability 1. Machine Learning, 14(3).
- Peter Dayan and Terrence J. Sejnowski (1996).  Exploration bonuses and dual control. Machine Learning, 25:5-22.
- Vijaykumar  Gullapalli and  Andrew G. Barto (1994). Convergence of  indirect  adaptive asyn­ chronous value  iteration  algorithms. In J.  D. C owan , G. Tesa ur o , and J. Alspector,  editors, Advances  in  Neural  Information  Processing  Systems 6, pages 695-70:2, San Mateo, CA. Morgan Kaufmann.
- Tommi Jaakkola, Michael 1. Jordan, and Satinder Singh (1994). On the convergence of stochast i c iterative dynamic programming algorithms. Neural  Computation, 6(6):1185-1201, November.
- George H. John (1995).  When the best move isn't optimal: Q-Iearning with exploration.  Unpublished manuscript,  available  through URL :ftp://starry.stan:ford.edu/pub/gj ohn/papers/rein-nips .ps. Leslie Pack Kaelbling, Michael L. Littman, and Andrew W. Moore (1996). Reinforcementlearning: A su rv ey . Journal  of Artificial  Intelliyence  Research, 4:237-285.
- George H.  John (1994). When  the  best move  isn't  optimal:  Q-Iearning with exploration. In Proceedings  of  the  Twelfth  National  Conference  on  Artificial  Intelliyence, page 1464, Seattle, WA.
- P. R. Kumar and P.  P.  Varaiya. (1986). Stochastic Systems; Estimation,  identification,  and Adaptitte  Control. Prentice Hall, Englewood Cliffs, NJ.
- Michael  L.  Littman and  Csaba  Szepesvari  (1996). A  generalized reinforcement-learning mod­ el:  Convergence and applications. In  Lorenza Saitta,  editor, Proceedings of  the Thirteenth International  Conference  on  Machine  Learning, pages 310-318.
- Michael Lederman Littman (1996). Alyorithms f o r  Sequential Decision Making. PhD  thesis, Department of Computer Science, Brown University, February. Also Technical Report CS-9609.
- Martin L. Putennan (1994). Markov  Decision  Processes-Discrete  Stochastic  Dynamic  Program­ ming. John Wiley &amp; Sons, Inc., New York, NY.
- G. A. R UIIlIll ery  (1994). Problem  solving  with  reinforcement  learning. PhD  thesis,  Cambridge University Engineering Department.
- G. A. RUIIlIllery and M. Niranjan (1994).  On-line Q-Iearning using connectionist systems.  Tech­ nical Report CUED/F-INFENG/TR 166, Cam b r i dge University Engineering D ep ar t ment .
- Satinder P. Singh and Richard S. Sutton (1996). Reinforcement learning with replacing eligibility traces. Machine  Learniny, 22(1/2/3):123-158.
- Satinder Pal Singh and Richard C. Yee (1994).  An upper bound on the loss from approximate optimal-value functions. Machine  Learning, 16:227.

- Rich  Sutton and  Andy Barto (1997)\_ An  Introduction  to  Reinforcement  Learning. The  MIT Press, forthcoming.
- Richard  S. Sutton (1988).  Learning to  predict by  the  method of temporal differences. Machine Learning, 3(1):9-44.
- Richard S.  Sutton (1996). Generalization in reinforcement learning:  Successful  examples using sparse coarse coding.  In D. S. Touretzky, M.  C.  Mozer,  and M. E. Hasselmo, editors, Advance., in  Neural  I nformation  Proce.,.,ing  Sy.,t�ms 8, Cambridge, MA. The MIT  Press.
- Csaba  Szepesvari  and  Michael L. Littman  (1996). Generalized Markov  decision  processes: Dynamic-programming  and  reinforcement-learning algorithms. Technical  Report  CS-96-11, Brown University, Providence, RI.
- Sebastian B.  T h ru n  ( 1992). The role of  exploration in  learning control. In David A. White and Donald  A.  Sofge, editors, Handbook  of Intelligent  Control:  Ne'Ural,  Fuzzy,  and  Adaptitte Approache.,. Van Nostrand Reinhold, New York, NY.
- John N.  Tsitsiklis  ( 1994). Asynchronous stochastic approximation and  Q-Iearning. Machine Learning, 16(3):185-20:2, September 1994.
- John N. Tsitsiklis and Ben jamin Van Roy (1996).  An analysis of temporal-diff erence learning with function approximation. Technical Report LIDS-P-2322, Massachusetts Institute of  Technology, March. Available  through  URL http://;reb.m.it. edu/bvr/v'Nll/td.ps. To  appear in IEEE Tran.,aetion.,  on  Automatic  Control.
- Christopher J.  C.  H.  Watkins ( 1989). Learning  from  Delayed  Reward.,. PhD  thesis,  King's College,  Cambridge, UK.
- Christopher J. C. H. Watkins and Peter Dayan (199:2).  Q-Iearning. Machine  Learning, 8(3)::279292.
- Ronald J. Williams  and  Leemon C. Baird, III (1993). T i g ht  performance bounds  on  greedy policies  based on imperfect value functions. Technical Report NU-CCS-93-14,  Northeastern University, College of Computer Science, Boston, MA,  November.