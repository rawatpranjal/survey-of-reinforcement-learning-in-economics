## Technical  Note

## Q,-Learning

CHRISTOPHER  J.C.H. WATKINS

25b  Framfield  Road,  Highbury, London N5  IUU, England

## PETER  DAYAN

Centre for  Cognitive Science,  University  of Edinburgh,  2  Buccleuch  Place,  Edinburgh EH8  9EH,  Scotland

Abstract. Q-learning (Watkins, 1989) is a simple way for agents to learn how to act optimally in controlled Markovian domains. It amounts to an incremental  method for dynamic programming which imposes  limited computational demands. It works by successively improving its evaluations of the quality of particular actions at particular  states.

This paper presents and proves in detail a convergence theorem for Q,-learning based on that outlined in Watkins (1989).  We show that Q-learning converges to the optimum action-values with probability  1 so long as all actions are  repeatedly  sampled  in all states  and the  action-values are  represented  discretely.  We also sketch extensions to the cases of non-discounted, but absorbing, Markov environments, and where many Q values can be changed each  iteration,  rather  than just  one.

Keywords.  Q -learning,  reinforcement learning, temporal  differences,  asynchronous dynamic programming

## 1.  Introduction

Q-learning (Watkins, 1989) is a form of model-free  reinforcement learning. It can also be viewed as a method of asynchronous dynamic programming  (DP). It provides agents with the capability of learning to act optimally  in Markovian domains by experiencing the consequences  of actions,  without requiring  them  to build  maps  of  the  domains.

Learning  proceeds  similarly  to Sutton's  (1984;  1988)  method  of temporal  differences (TD):  an agent tries an action at a particular  state,  and evaluates its consequences  in terms of  the  immediate  reward  or  penalty  it  receives and its  estimate  of the  value of the  state to which it is taken.  By trying all actions  in all states  repeatedly,  it learns  which are  best overall, judged by long-term  discounted  reward. Q-learning is a primitive (Watkins,  1989) form of learning, but, as such, it can operate as the basis of far more sophisticated  devices. Examples  of its use include  Barto and Singh  (1990),  Sutton (1990),  Chapman  and Kaelbling  (1991),  Mahadevan  and  Connell  (1991),  and  Lin  (1992),  who developed  it independently.  There  are  also  various  industrial  applications.

This paper presents the proof outlined by Watkins (1989) that Q-learning converges.  Section 2 describes the problem, the method, and the notation, section 3 gives  an  overview of the proof,  and section  4 discusses  two extensions.  Formal  details  are left  as far as possible to the appendix.  Watkins (1989)  should be consulted  for a more extensive  discussion of Q-learning, including its relationship  with dynamic programming and TD. See also Werbos (1977).

## 2.  The task  for  Q-learning

Consider  a computational agent moving around some discrete,  finite world, choosing one from  a finite collection  of actions  at  every time  step.  The  world constitutes a  controlled Markov process  with the agent as a controller.  At step n, the agent is equipped to register the state xn (€ X) of the world, an can choose  its action an (€  2)1 accordingly. The  agent receives  a probabilistic  reward rn, whose mean value ( RXn (an) depends  only  on the state and action, and the state of the world changes probabilistically to yn according to the law:

<!-- formula-not-decoded -->

The task facing the agent is that of determining an optimal policy, one that maximizes total discounted expected reward. By discounted reward, we mean that rewards received s steps hence are worth less than  rewards received now, by a factor of ys (0  &lt; y &lt;  1). Under a  policy T, the  value of  state x is

because the  agent expects  to receive (Rx(r(x)) immediately  for performing the  action I recommends,  and  then  moves  to  a  state  that  is  'worth' V*(y) to  it,  with  probability Pxy [&gt;(*)]. The theory of DP (Bellman &amp; Dreyfus,  1962; Ross,  1983) assures us that there is  at  least  one  optimal  stationary  policy T*  which  is  such that

<!-- formula-not-decoded -->

is as well as an agent can do from  state x. Although this might look circular, it is actually well defined,  and  DP provides a number of methods for calculating V* and one T*(  assuming that (Rx(a) and Pxy[a]  are  known. The task facing  a Q,  learner  is that of determining a f* without initially knowing these values. There are traditional methods (e.g., Sato, Abe &amp; Takeda,  1988)  for learning (Rx(a) and Pxy[a] while concurrently  performing DP, but any  assumption of certainty  equivalence,  i.e.,  calculating  actions as if the  current model were accurate, costs dearly in the early stages of learning (Barto &amp; Singh,  1990). Watkins (1989)  classes  ^-learning  as incremental dynamic programming,  because of the step-bystep  manner in  which it determines  the  optimal  policy.

For  a  policy T,  define  Q, values  (or  action-values)  as:

<!-- formula-not-decoded -->

In  other  words,  the  3, value  is the  expected  discounted  reward for  executing  action a at state x and  following  policy T  thereafter.  The  object  in  Q-learning  is  to  estimate  the  5,

values for an optimal policy. For convenience,  define these as Q*(x,  a) = Q**(x,  a),  Vx,  a. It is straightforward to show that V*(x) =  maxa Q*(x, a) and that if a* is an action at which the maximum is attained,  then an optimal  policy  can be  formed as r*(x) =  a*. Herein lies  the  utility  of the  Q,  values-if an  agent  can  learn  them,  it can  easily  decide  what it is optimal to do. Although there may be more than one optimal policy or a*, the  5*  values are  unique.

In Q-learning,  the agent's experience  consists of a sequence of distinct stages or episodes. In the  n th episode,  the  agent:

-  observes  its current  state x n ,
-  selects  and performs an action a n ,
-  observes  the subsequent  state y n ,
-  receives  an immediate  payoff r n , and
-  adjusts its  Q n-1 values using a learning  factor  an,  according  to:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

is the best the agent thinks it can do from state y. Of course,  in the early stages of learning, the Q, values may not accurately reflect the policy they implicitly define (the maximizing actions  in equation 2).  The  initial  Q,  values, Q, 0 (X,  a), for all states and  actions are  assumed  given.

Note  that  this  description  assumes  a  look-up  table  representation  for  the Q, n (x, a). Watkins (1989) shows that ^-learning may not converge correctly for other representations.

The most important condition implicit in the convergence  theorem  given below is that the sequence  of episodes  that forms the basis of learning must include an infinite number of episodes  for each  starting state and  action.  This may be considered  a strong  condition on the  way states and actions  are  selected-however,  under  the  stochastic  conditions  of the theorem,  no method  could be guaranteed to find  an optimal policy under weaker  conditions.  Note,  however,  that the  episodes  need  not  form  a continuous  sequence-that is the y  of one  episode need  not  be  the x of  the  next  episode.

The following theorem  defines a set of conditions  under which Qn (x,  a) -Q*(x, a) as n -o.  Define n i (x, a) as the index of the i th time  that action a is tried  in state x.

## Theorem

Given bounded  rewards \rn\ &lt;  (R, learning  rates  0  &lt; an  &lt; 1,  and

<!-- formula-not-decoded -->

then Qn(x,  a) -G*(x, a) as n -o, Vx, a,  with probability 1.

## 3.  The  convergence  proof

The key to the convergence proof is an artificial controlled Markov process called the actionreplay process ARP, which is constructed from the episode sequence and the learning rate sequence an.

A formal  description  of the ARP  is given in the appendix, but the easiest  way to think of it is in terms of a card game. Imagine each episode  (xt, at,  y t ,  r t ,  a t ,) written on a card. All  the  cards  together  form  an  infinite  deck,  with the  first  episode-card  next-to-bottom and stretching infinitely  upwards, in order.  The bottom card  (numbered 0) has written on it  the  agent's  initial  values Q0(x,  a) for all  pairs of x and a. A state of the  ARP, (x,  n), consists of a card number (or level) n, together with a state x from  the real  process.  The actions  permitted  in the  ARP  are  the  same  as  those  permitted  in the  real  process.

The next state of the ARP,  given current state (x, n) and action a, is determined as follows. First, all the cards for episodes later than n are eliminated, leaving just a finite deck.  Cards are then removed one at a time from top of this deck and examined until one is found whose starting  state  and action  match x and a, say at episode t. Then  a biased  coin  is flipped, with probability a, of coming out heads,  and 1 a, of tails. If the coin turns up heads, the episode recorded on this card is replayed, a process described below; if the coin turns up tails, this card  too is thrown away and the search  continues for another card matching x and a. If the bottom card is reached,  the game stops in a special,  absorbing,  state, and just  provides  the  reward  written on this  card  for x,  a, namely  Qo(x, a).

Replaying the episode on card t consists of emitting the reward, rt , written on the card, and then moving to the next state  (yt, t  1}  in the ARP, where y t  is the state to which the real process went on that episode.  Card t itself is thrown away. The next state transition of  the  ARP  will  be  taken based  on just  the  remaining  deck.

The above completely specifies how state transitions and rewards are determined in the ARP. Define P^tiyim) [a] and (Rjn)(a) as the transition-probability matrices and  expected rewards of the  ARP. Also define:

<!-- formula-not-decoded -->

as the probabilities  that, for each x,  n and a, executing action a at state (x,n) in the ARP leads  to  state y of  the  real  process  at  some  lower  level  in  the  deck.

As defined above,  the ARP  is as much a controlled  Markov process  as is the real  process.  One can therefore consider  sequences of states  and controls,  and  also  optimal  discounted C* values for the ARP.2 Note that during such a sequence, episode cards are only removed from the deck, and are never replaced.  Therefore, after a finite number of actions, the bottom  card  will always be  reached.

## 3.1.  Lemmas

Two lemmas form the heart of the proof.  One shows that, effectively by construction,  the optimal  Q,  value for ARP  state (x, n) and action  a  is just Qn(x, a).  The next shows that for  almost all possible  decks, P^[a] converge  to Pxy[a] and  Rx(n)(a)  converge to Sx(a) as n -o. Informal statements of the lemmas and outlines of their proofs are given below; consult  the  appendix  for  the  formal  statements.

## Lemma  A

Qn(x, a) are  the optimal  action  values  for ARP  states (x, n) and ARP  actions a.

The ARP was directly constructed to have this property. The proof proceeds by  backwards induction,  following the  ARP  down through  the  stack  of past  episodes.

## Lemma B

Lemma B concerns the convergence  of the ARP  to the real  process.  The first  two steps are preparatory; the next two specify  the form of the convergence and provide foundations for  proving  that  it  occurs.

## B.1

Consider  a discounted,  bounded-reward,  finite  Markov process.  From  any starting  state x, the difference between the value of that state under the finite  sequence of s actions and its value under  that  same  sequence  followed by any other  actions  tends  to 0 as s o.

This follows  from  the presence of the discount factor  which weighs the (s  + l)th  state by ys  0 as s -o.

## B.2

Given any level /,  there  exists  another  yet higher  level, h, such that the  probability can be  made arbitrarily  small of  straying below l after  taking  5 actions  in the  ARP, starting from  above h.

The probability, starting at level h of the ARP  of straying below any fixed level / tends to 0 as h -o.  Therefore there is some  sufficiently  high level for which s actions can be safely accommodated,  with an arbitrarily high probability of leaving the ARP  above /.

## B.3

With probability  1, the probabilities P$[a] and expected  rewards (RJ (n) (a) in the ARP  converge and tend to the transition  matrices  and expected  rewards  in the real process as the level n increases  to  infinity.  This,  together  with  B.2,  makes  it  appropriate  to  consider P^\d\ rather  than the ARP transition matrices P$^,fy, m)[a], i.e.,  essentially  ignoring the level  at  which the  ARP  enters  state y.

The ARP effectively  estimates the mean rewards and transitions of the real process over all the episodes.  Since its raw data are unbiased, the conditions  on the sums and sums of  squares  of  the  learning  rates  c ni ( X , a )  ensure  the  convergence  with  probability  one.

## B.4

Consider executing a series of s actions in the ARP  and in the real process. If the probabilities P^[a] and  expected  rewards  ( R (n) (a  ) at  appropriate  levels  of the  ARP  for  each of the actions, are close  to P xy [a] and ( Rx(a),Va, x,  y, respectively,  then the value of the series  of actions  in  the  ARP  will be  close  to  its  value in the  real  process.

The discrepancy in the action values over a finite number s of actions between the values of two approximately equal  Markov processes  grows at most quadratically  with s. So, if the transition probabilities  and rewards are close,  then the values of the actions must be close  too.

## 3.2.  The theorem

Putting these together,  the ARP tends towards the real process, and so its optimal  Q, values do too. But  3 n (a, x) are the optimal  Q values for the n th level of the ARP  (by Lemma A), and  so tend  to Q*(x,  a).

Assume,  without loss  of generality,  that Q 0 ( X , a) &lt; R/(l -y) and that  (R &gt;  1.

Given e  &gt;  0,  choose s such  that

<!-- formula-not-decoded -->

By  B.3,  with probability  1, it  is  possible  to  choose  /  sufficiently  large  such  that  for n  &gt;  l, and Va, x,  y,

<!-- formula-not-decoded -->

By B.2,  choose h sufficiently  large  such that for n  &gt;  h, the  probability,  after  taking s  actions,  of  ending  up  at a  level  lower  than  l  is  less  than  min{(e(l -y)/6s(A), (e/3s(s  + 1)&lt;R)}.  This  means  that

<!-- formula-not-decoded -->

where the primes  on P'(n)  and (R' ( n } indicate that these are conditional  on the level in the ARP  after  the  sth  step  being  greater  than  /.

Then,  for n &gt; h, by B.4,  compare  the value QARp((x, n),  a 1 , . . . , as) of taking actions a1, ..., as at state x in the ARP, with  Q,(x, a1,  ..., as) of taking  them  in the real  process:3

<!-- formula-not-decoded -->

Where,  in equation 4,  the first term  counts the cost of conditions for B.2 not holding, as the cost of straying below / is bounded by 2s&lt;R /(1 -  7). The second term is the cost, from  B.4,  of  the  incorrect  rewards  and  transition  probabilities.

However, by B.1, the effect  of taking only s actions makes a difference  of less than  e/6 for  both the ARP  and the real process.  Also since equation 4 applies to any set of actions, it applies perforce to a set of actions optimal for either the ARP or the real process.  Therefore

<!-- formula-not-decoded -->

So,  with  probability  1, Qn(x,  a) -Q*(x,  a)  as n -o as required.

## 4.  Discussions and conclusions

For  the  sake  of  clarity,  the  theorem  proved  above  was somewhat restricted.  Two  particular  extensions  to the version  of ^-learning  described  above  have been used in practice.  One is the non-discounted  case  (7  =  1), but for a Markov process  with absorbing goal  states, and the other is to the case where many of the  5 values  are updated  in each iteration rather than just one (Barto, Bradtke &amp; Singh, 1991). The convergence result holds for both of these,  and this section  sketches the modifications to the proof that are necessary.

A process  with absorbing  goal states has one or more states which are bound in the end to trap the agent.  This ultimate certainty of being trapped plays the role that 7  &lt;  1  played in  the  earlier  proof,  in  ensuring  that the  value  of  state x under  any policy r, Vw(x), is bounded,  and that lemma B.1 holds,  i.e.,  that the difference  between considering  infinite and finite (s) numbers of actions  tends  to 0 as s -o.

Since the process  would always get trapped were it allowed to run, for every state x there is  some  number of actions u(x) such that no matter  what they are,  there  is a probability p(x) &gt;  0 of  having  reached  one of the  goal  states  after  executing  those  actions.  Take

u*  = maxx{u(x)}, and p* =  minx{p(x)} &gt; 0  (since  there  is only  a finite number of states).  Then  a  crude  upper  bound  for V*(x) is

since in each « * steps the agent earns a reward of less than u *R,  and has probability less than  (1 -  p*) of not  having been trapped. Similarly, the  effect  of measuring the  reward after  only  o u* steps  is less  than (1 p*)*u*® -0 as j -o, and so an equivalent of  lemma  B.1  does  hold.

Changing more than one Q value on each iteration requires a minor modification to the action  replay  process ARP  such that an action  can be taken at any level  at which it was executed  in the  real  process-i.e.,  more  than one  action  can  be  taken at  each  level.  As long  as the  stochastic  convergence  conditions  in  equation  3 are  still  satisfied, the  proof requires  no non-trivial modification. The Qn(x, a)  values are still optimal  for the modified ARP,  and this  still tends  to the real process in the original manner.  Intuitively, the proof relies on the ARP  estimating rewards  and transition  functions  based  on  many  episodes, and  this  is just  speeded  up  by changing more  than one  Q value per  iteration.

Although the paper has so far presented an apparent dichotomy between  9-learning and methods based on certainty equivalence, such as Sato, Abe and Takeda (1988),  in fact there is more of a continuum. If the  agent can  remember  the  details  of its learning  episodes, then,  after  altering  the learning  rates,  it can use  each  of them more  than  once  (which is equivalent to putting cards that were thrown away, back in, lower down on the ARP  stack). This biases the Q-learning process towards the particular sample of the rewards and transitions that it has experienced. In the limit of re-presenting  'old'  cards infinitely  often, this reuse amounts to the certainty  equivalence step of calculating  the optimal actions for the observed  sample of the Markovian environment rather than the actual environment itself.

The theorem above only proves the convergence of a restricted version of Watkins' (1989) comprehensive  Q-learning  algorithm, since it does not permit updates based on the rewards from more than one iteration.  This addition was pioneered  by Sutton (1984;  1988)  in his TD(X)  algorithm,  in which a reward  from  a step taken r iterations previously is weighted by Xr, where X &lt; 1. Unfortunately, the theorem  does not extend trivially to this case,  and alternative  proof  methods  such as those  in  Kushner and  Clark  (1978)  may be  required.

This paper has presented  the proof outlined by Watkins (1989) that  ^.-learning converges with probability one under reasonable conditions on the learning rates and the Markovian environment.  Such  a  guarantee  has  previously  eluded  most  methods  of reinforcement learning.

## Acknowledgments

We are very  grateful to Andy Barto,  Graeme  Mitchison,  Steve Nowlan, Satinder  Singh, Rich Sutton and three anonymous reviewers for their valuable comments  on  multifarious aspects of Q-learning  and  this paper.  Such  clarity  as  it possesses  owes to  Rich  Sutton's

tireless  efforts.  Support was from  Philips Research  Laboratories and SERC.  PD's current address  is CNL,  The  Salk Institute,  PO Box 85800,  San Diego,  CA 92186-5800,  USA.

## Notes

1.  In general,  the  set of available actions may differ  from  state  to state. Here we assume it does not, to  simplify the  notation.  The  theorem  we present  can  straightfowardly  be  extended  to  the  general  case.
2.  The discount  factor  for the ARP  will  be taken to be y,  the  same  as  for  the  real  process.
3.  The bars  over the  £  indicate  that the  sum  is  over only  a  finite  number of actions,  with  0  terminal reward.

## Appendix

## The action-replay  process

The definition  of the ARP  is  contingent on a particular  sequence  of episodes  observed in the real process. The state space of the ARP is {(x, n)}, for x a state of the real  process and n &gt; 1, together  with one,  special,  absorbing state,  and  the action  space  is  {a}  for a an  action  from  the  real  process.

The  stochastic  reward  and state transition consequent  on  performing action a at  state {x,  n) is given as  follows.  For convenience,  define ni  =  n i ( x , a),  as the  index of the  ith time  action a was tried  at  state x. Define

<!-- formula-not-decoded -->

such that ni* is the last time before episode n that x,  a was exeucted  in the real  process. If  i*  =  0, the  reward  is  set as  Q0(x,  a),  and the ARP  absorbs.  Otherwise, let

<!-- formula-not-decoded -->

be  the  index  of the  episode  that  is replayed or  taken,  chosen  probabilistically  from  the collection  of existing  samples  from  the real process.  If ie  = 0,  then  the  reward is set at Qo(x,  a) and the ARP  absorbs,  as above,  Otherwise,  taking ie provides  reward rnie, and causes  a  state  transition  to (ynie,  ne  1) which  is  at  level nie  1. This  last  point is crucial,  taking an action  in the ARP  always causes a state transition to a lower level-so it ultimately terminates. The discount factor in the ARP  is 7, the same as in the real process.

## Lemma  A:  Qn  are  optimal  for  the  ARP

Qn(x,  a) are the optimal  action  values for ARP  states  {x, n) and ARP actions a.  That is

<!-- formula-not-decoded -->

## Proof

By induction. From the construction  of the ARP, Q 0 ( X ,  a) is the optimal-indeed the only possible-action  value  of (x, 0), a. Therefore,

<!-- formula-not-decoded -->

Hence  the theorem  holds  for n  = 0.

Suppose  that  the values  of £n-1, as produced  by the  ^-learning  rule,  are  the  optimal action  values  for  the  ARP  at  level n  - 1, that  is

<!-- formula-not-decoded -->

This implies that the V n - 1 ( x ) are the optimal values V* for the ARP at the n  1th level, that  is

<!-- formula-not-decoded -->

Now consider  the cases in trying to perform action a in (x, n). If x,  a  =  xn,  an, then this is the same  as performing a in (x, n  1), and £ n ( x , a)  = Qn-1(x,a).  Therefore,

<!-- formula-not-decoded -->

Otherwise,  performing an in  {xn, n}

-  with  probability  1 an is exactly the same  as performing an in (xn,  n  1),  or
-  with probability an yields  immediate  reward rn and new state (yn,  n  1).

Therefore  the  optimal  action  value in  the  ARP  of {xn,  n},  an is

<!-- formula-not-decoded -->

from  the  induction  hypothesis  and  the  $n  interation  formula in  equation 1.

Hence, Q,n(x,  a)  = Q*ARP({x, n),  a), Va, x, as  required.

## Lemma  B

B.1 Discounting infinite  sequences

Consider  a  discounted,  bounded-reward,  finite  Markov  process  with  transition  matrix Pxy[a]. From any starting  state x, the difference between the value of that state under any set  of s actions  and  under those same s actions  followed  by any arbitrary policy tends to 0 as s -o.

## Proof

Ignoring  the  value of the s  + 1th  state  incurs  a  penalty of

But if all rewards are bounded by R, | V*(x)\  &lt; R/(l  -  7), and so

<!-- formula-not-decoded -->

B.2  The  probability of straying  below level l is executing s actions can be make arbitrarily small

Given any level  l,  there  exists another  yet higher  level, h, such that the  probability can be made  arbitrarily  small  of straying below  / after taking s actions  in the  ARP, starting from  above h.

## Proof

Define ih as the largest i such that n'(x, a) ^  n, and il as the smallest  such that ni (x,  a) &gt;  l. Then,  defining xno =  1, the probability  of straying below  l  starting  from (x, n),  n  &gt;  l executing  action a is:

<!-- formula-not-decoded -->

where,  as before, ni  =  n i (x, a). But  njl;/(l -«'')  &lt;  exp(E)t,;  «'«)  0 as n and hence ih -o. Furthermore, since the state and action spaces are finite,  given n , there exists  some level n1 such that starting above there  from  any (x, a)  leads  to a level above / with probability at least  1 17.  This argument iterates  for the second  action with n1 as the new lower limit. n can be chosen appropriately to set the overall probability of  straying below  / less  than  any arbitrary  e  &gt; 0.

B. 3  Rewards  and  transition probabilities  converge  with probabability  1

With probability  1, the probabilities P%\a] and expected rewards (RJ n) (a) in the ARP  converge and tend to the transition matrices  and expected  rewards in the real process as the level n increases  to  infinity.

## Proof

A standard theorem  in stochastic convergence  (e.g.,  theorem  2.3.1  of Kushner &amp;  Clark, 1978)  states  that  if Xn are  updated  according to

where 0  &lt; 8B &lt; 1, £,"i 3n = o, £?!, p% &lt; o,  and £n, are bounded random  variables with  mean  E,  then

<!-- formula-not-decoded -->

If  R(x,n) (a)  is the expected  immediate  reward for performing action a from  state x at level n in  the  ARP, then &amp;(x,n) (a)  satisfies

where  the R and  the a satisfy  the  conditions  of  the  theorem  with  E  = 6x(a), and remembering  that n i is the  i th occasion  on which action a was tried  at state x. Therefore ®(x,n) (a) -Rx((a) as n o, with probbility  one.  Also,  since there is only a finite number  of  states  and  actions,  the  convergence  is  uniform.

Similarly,  define

<!-- formula-not-decoded -->

as a (random variable) indicator  function  of the n th transition, mean value Pxy (a). Then, with P^}\a] as  the probability  of ending up at  state y based  on  a transition  from  state x using  action a at  level n in  the ARP,

and  so, by the theorem, P^\a} -Pxy[a] (the  transition  matrix  in the real  process) as n -o, with  probability  one.

Since, in addition,  all observations  from  the real process are independent,  and, by B.2, the probability  of straying below  a fixed  level k can be made arbitrarily  small, the transition probabilities  and expected rewards for a single step conditional on ending up at a level greater than k also  converge  to Pxy[a] and R x (a) as n -o.

B.4  Close rewards and  transitions  imply  close  values

Let P!y[a], for i  = 1 ...  s  be the transition  matrices of s Markov chains,  and  RJ(a) be the reward functions. Consider  the s-step chain formed from  the concatenation of thesei.e.,  starting  from  state  x 1 ,  move to state x 2 according  to /^[fl]],  then state X3 , according to Px2x3[a2],  and  so on,  with commensurate  rewards.  Given  n  &gt;  0,  if P i [ a ] are within n/R  of Pxy[a], Va, x,  y, and  Rx(a)  ... Rx(a) are within n of  Rx(a),  Va, x, then the value of the s actions in the concatenated  chain is within n s(s + l)/2  of their value in the real  process.

## Proof

Define:

<!-- formula-not-decoded -->

as the expected  reward in the real process for executing two actions, a 1 and  a 2 at state x, and

<!-- formula-not-decoded -->

as  the  equivalent  in the  concatenated  chain  for exactly  the  same  actions.

Then,  since / Rx(a)  &amp;x(a)\ &lt; n and Pxy[a] Pxy[a] &lt;  n / R ,  Va, i, x, y,

<!-- formula-not-decoded -->

Similarly,  for s actions,

<!-- formula-not-decoded -->

This  applies  to the ARP  if the  rewards  and transition  matrices at the successively  lower levels  are  sufficiently  close  to  those  in the  real  process-the main body  of the  theorem quantifies  the  cost  of this  condition  failing.

## References

- Barto,  A.G., Bradtke,  S.J. &amp; Singh,  S.P. (1991). Real-time  learning  and  control  using asynchronous  dynamic programming. (COINS  technical  report  91-57).  Amherst:  University of  Massachusetts.
- Barto,  A.G. &amp; Singh, S.P. (1990). On the computational economics  of reinforcement learning. In D.S. Touretzky, J. Elman, T.J. Sejnowski &amp; G.E. Hinton, (Eds.), Proceedings of the 1990 Connectionst Models  Summer School. San Mateo,  CA:  Morgan Kaufmann.
- Bellman,  R.E. &amp;  Dreyfus, S.E. (1962). Applied  dynamic programming. RAND  Corporation.
- Chapman,  D.  &amp; Kaelbling, L.P. (1991).  Input generalization  in delayed  reinforcement learning: An algorithm and performance  comparisons. Proceedings of the 1991 International Joint Conference  on Artificial  Intelligence (pp. 726-731).
- Kushner, H. &amp; Clark, D. (1978). Stochastic approximation methods for  constrained and unconstrained  systems. Berlin,  Germany:  Springer-Verlag.
- Lin, L. (1992).  Self-improving reactive agents based on reinforcement learning, planning and teaching. Machine Learning,  8.
- Mahadevan &amp; Connell  (1991).  Automatic programming of behavior-based robots  using reinforcement learning. Proceedings  of  the  1991  National  Conference  on AI (pp. 768-773).
- Ross, S. (1983). Introduction  to  stochastic  dynamic  programming. New York,  Academic  Press.
- Sato,  M.,  Abe, K. &amp;Takeda, H. (1988). Learning control of finite Markov chains with explicit trade-off between estimation  and  control. IEEE  Transactions on  Systems,  Man  and  Cybernetics,  18, pp. 677-684.
- Sutton, R.S.  (1984). Temporal  credit assignment in reinforcement  learning. PhD Thesis,  University of Massachusetts, Amherst,  MA.
- Sutton, R.S.  (1988). Learning to predict  by the methods of temporal  difference. Machine Learning,  3, pp.  9-44.
- Sutton. R.S. (1990). Integrated architectures  for learning,  planning, and reacting based on approximating dynamic programming. Proceedings of  the  Seventh International  Conference  on Machine  Learning. San  Mateo,  CA: Morgan Kaufmann.
- Watkins,  C.J.C.H.  (1989). Learning  from  delayed  rewards. PhD  Thesis,  University of Cambridge,  England. Werbos, P.J. (1977). Advanced forecasting methods for global crisis warning and models of intelligence. General Systems  Yearbook, 22, pp. 25-38.