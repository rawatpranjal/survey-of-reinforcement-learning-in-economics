## Learning  to  Predict  by the  Methods of  Temporal Differences

## RICHARD S.  SUTTON

(RICH@GTE.COM)

(;TE  Laboratories Incorporated, ~0 Sylvan  Road,  Waltham,  MA  0225~,  U.S.A.

(Received:

April 22,  1987)

(Revised:

February  4.  1988)

Keywords: Incremental learning, prediction, connectionism, credit assignment, evaluation flmctions

Abstract. This  article introduces a class of  incremental learning procedures specialized for prediction  that is, for using  past experience with  an  incompletely known system to predict its  future  behavior. Whereas conventional prediction-learning methods assign credit by means of  the  difference between  predicted and actual outcomes,  tile new methods assign credit by means of  the  difference between temporally successive predictions. Although  such temporal-difference method~ have  been  used  in Samuel's checker player, Holland's bucket brigade, and  the  author's Adaptive  Heuristic Critic, they  have  remained  poorly  understood. Here we prove  their convergence and  optimality for  special cases and  relate them to  supervised-learning methods.  For most  real-world prediction problems, telnporal-differenee methods  require less memory  and less peak  computation than  conventional methods and they  produce  more accurate predictions. We  argue  that  most problems  to  which supervised learning is  currently applied  are  really prediction problems  of  the  sort  to  which temporaldifference methods  can  be applied to  advantage.

## 1.  I n t r o d u c t i o n

This  article concerns  the  woblem of learning to  predict, that. is, of  using  past experience with  an incompletely known  system  to  predict its future behavior. For examt)le,  through  experience one might learn  to predict  for  particular chess  positions whether  they  will lend  to  a  win.  for  particular cloud  formations whether there  will  be rain. or fbr  particular economic conditions how  nmch the stock  market will  rise  or fall.  Learning  to predict  is  one of  the most basic  and prevalent kinds  of  learning. Most pattern recognition problems,  for examt)le, can be treated as prediction problems  in  which the  classifier nmst predict the  correct classifications. Learning-to-predict problems  also  arise in heuristic search, e.g., in  learning an  evahmtion  function that  predicts tile utility of  searching particular parts  of  tile search  space, or  in  learning the  underlying model of  a problem  domain. An  important advantage  of  prediction learning is that. its  training examples  can be taken  directly from the  temporal  sequence of  ordinary sensory  input: no special supervisor or  teacher is  required.

In  this article, we introduce and  provide tilt first formal  results in  the  theory of  temporal-difference {TD) methods, a  class of  incremental learning procedures specialized for  prediction problems.  Whereas conventional prediction-learning methods are  driven  by the  error between  predicted and actual outcomes.  TD methods are  similarly driven  by the  error  or difference between temporally successive predictions; with  them,  learning occurs  whenever  there is  a change in prediction over  time.  For example,  suppose a weatherman attempts  to predict on each  day  of  the  week whether  it will rain  on the  following Saturday. The conventional approach  is to  compare  each  prediction to  the  actual outcome whether  or  not  it does  rain  on Saturday. A TD  approach, on the  other  hand, is  to  compare each  day's  prediction with that  made  on the  following (lay.  If a 50%  chance  of  rain  is  predicted on Monday, and a 75%  chance  on ~wsday, then  a TD  method increases predictions for  days  similar to  Monday, whereas a conventional method might either increase or  decrease them depending  on Saturday's actual outcome.

We  will  show  that  TD  methods have two kinds  of  advantages  over  conventional prediction-learning methods. First, they  are  more incremental and therefore  asier to  compute.  For  example,  the  TD  method for  predicting Saturday's weather  can  update  each  day's prediction on  the  following day,  whereas the  conventional method must wait  until Saturday, and then  make the  changes for  all days  of  the  week. The conventional method would have  to  do more computing  at  one  time  than  the  TD  method and  would require more storage duriltg the  week.  The second  advantage  of  TD  methods is  that  they  tend  to  make more efficient use of  their experience: they  converge  faster and produce  better  predictions. We  argue  that  the  predictions of  TD  methods are  both  more accurate and easier to  compute than  those  of  conventional methods.

The earliest and best-known  use of  a TD  method was in  Samuel's  (1959) celebrated checker-playing program.  For  each  pair of  successiw, game positions, the  program used the  difference between the  evaluations assigned  to  the  two positions to  modify the  earlier one's  evaluation. Similar methods have been used in  Holland's (1986)  bucket  brigade, in the  author's Adaptive  Heuristic Critic (Sutton, 1984;  Barto, Sutton  &amp; Anderson,  1983), and in  learning systems studied by Witten  (1977), Booker (1982), and Hampson  (1983).  TD  methods have also  been proposed  as  models of  classical conditioning (Sutton  &amp;  Barto, 1981a,  1987:  Gelperin, Hopfield &amp; Tank,  1985;  Moore et  al, 1986;  t(lopf~ 1987).

Nevertheless, TD  methods  have  remained  poorly understood. Although  they have performed  well,  there  has been no theoretical understanding of  how  or why they  worked. One reason  is  that  they  were never  studied independently, but  only  as  parts of  larger and more complex systems.  Within  these  systems. TD  methods were used to  improve  evaluation fimctions by better predicting goal-related events such  as  rewards, penalties, or  checker game outcomes.  Here we advocate viewing  TD  methods  in  a simpler way as  methods  for  efficiently learning to  predict arbitrary events, not  lust goal-related ones.  This  simplification allows us  to  evaluate them in  isolation and has  enabled us  to  obtain formal results. In this paper,  we prove  the  convergence  and optimality of  TD  methods for  important special cases, and we formally relate them to  conventional supervised-learning procedures.

Another  simplification we make in  this paper  is to  focus on  numerical prediction processes rather than on rule-based or symbolic  prediction (e.g., Dietterich &amp;  Miehalski,  1986). The approach, taken  here  is  much like that used in  connectionism and in  Sanmel's  original work our  predictions are  base(t on  numerical features combined using  adjustable parameters  or "weights." This and other ret)resentational assumptions  are  detailed in  Se('tion 2.

Given the current  interest in learning procedures  for multi-layer connectionist networks (e.g., Rumelhart,  Hinton,  &amp;  Williams, 1985: Ackley, Hinton. &amp;  S@~owski,  1985; Barto, 1985; Anderson,  1986; Williams,  1986: ttalnpson &amp;  Volper.  1987),  we  note  that  here  we are  c(meerned  with  a different, set  of issues. The work with  multi-layer networks  focuses on learning input-output mat)pings of  more comt)lex flm('tional forms.  Most of  that work remains  within the  supervised-learning paradigm,  whereas  here  we are  interested in  extending and going  beyond it.  We  consider mostly  mappings of  very  simple  flmctional forms,  because  the  differences between supervised learning methods and TD methods are  clearest in  these  cases.  Nevertheless. the TD methods presented here  can [)e  directly extende(t to  multi-layer networks  (see  Seetiou 6.2).

The next  section introduces a specific ('lass of  temporal-difference t)roeedures by contrasting them with  convent ional, supervised-h,arning at)proaches, focusing on ('omputational issues.  Section  3 deveh)ps  an extended  ~'xample  that illustrates the  i)otential I)erfl)rmance a.dvantage~ of  TD  methods. Section  4 contains the  convergeilee and optimality theorems  and discusses TD  methods as gradient descent.  Section  5 discusses how  to  extend  TD  t)rocedures, and Se('th)n (i  relates them to  other  research.

## 2.  Temporal-difference and supervised-learning approaches to  prediction

Historically. t he most hnportant learlfing para(tigm has  t)een that  of supervi,~e,d tear,ring. In this  fl'amework  the  h'arner is  asked to  associate pairs  of items.  When  later presente&lt;t with  .iust the  first item  of  a t)air, the  learner is suI&gt;l)osed ~o recall the  second.  This  I)ara(tigm has t)een used in  l)attern classili('ation, ('oncet)t acquisition, learning from examt)les, system  identification. and associative memory.  For exanli)le, in  pattern ('lassifi(.aiion and concept a('(tuisitiol~, the  first item  i,~ an instance of  some pattern or  concept, and tim second  item  is  the  mtme  of  that  concet)t. In  system  identification, the  learner lnust reproduce the  input-out put  t)ehavior of  some tlnkliown system.  Here.  t h(, first item  of  each  pair is  an input  and the  second  is  the  correst)onding output.

Any prediction t)roblem can be cast  in  the  supervised-learning paradigm  by (aking  1he  tirst item  to  be  the  data  based  on  whi('h a t)rediction must be  made, and the  second  item  to  t)e the  actual outcome,  what the  I)redietion should  have been.  For  example,  to  tu'e(tict Saturday's weather, one  can  form  a  pair from  the measur(,meilts tak(,n on Monday  and the  a('tual (~bs(,rve(t weather  on Saturday. another pair from  the  measurements  taken  on  Tuesday  and  Satur(lay's weather, and so on. Although this  pairwise at)proach ignores the  ~equential structure of  the  l)robleln, it  is  easy  to  understand and analyze  and it  has been widely used. In this  pat)er, we  refer to tiffs as the ,sup~'rvi.~ed-h~arning approach t.o

prediction learning, and we refer to  learning methods  that  take  this approach as supervised-learninff methods. We  argue  that such  methods are  inadequate, and that  TD  methods  are  far  preferable.

## 2.1  Single-step and multi-step prediction

To clarify this claim, we distinguish two kinds  of  prediction-learning problems.  In single-step prediction problems, all information about  the  correctness of  each  prediction is  revealed at  once. In  multi-step prediction problems,  correctness is  not  revealed until more than  one  step  after the  prediction is  made, but partial information relevant to its  correctness is  revealed at  each  step. For  example,  the  weather  prediction problem  mentioned  above  is  a multi-step prediction problem  because  inconclusive  vidence  relevant to  the  correctness of  Monday's  prediction becomes available in  the  form  of  new observations on Tuesday,  Wednesday,  Thursday  and Friday. On  the  other hand,  if each  day's weather  were  to  be  predicted on the  basis of  the  previous day's observations that is, on  Monday predict Tuesday's weather, on  Tuesday  predict Wednesday's weather, etc. one would  have a single-step prediction problem,  assuming  no filrther observations were made between  the  time  of  each  day's prediction and its confirmation or  refiltation on the  following day.

In  this paper, we will be  concerned only  with  multi-step prediction problems. In single-step problems,  data  naturally comes in  observation-outcome pairs; these problems  are  ideally suited to  the  pairwise supervised-learning approach. Temporal-difference methods  cannot  be  distinguished from  supervised-learning methods  in  this case; thus  tile former  improve  over  conventional methods  only on multi-step problems. However. we argue  that  these  predominate  in  realworld  applications. For  example,  predictions about  next  year's economic  performance  are  not  confirmed or  disconfirmed all at  once,  but  rather bit  by bit as  the  economic situation is  observed  through  the  year.  The likely outcome of  elections is  updated  with  each  new poll, and the  likely outcome of  a chess game is  updated  with  each  move. When  a baseball batter predicts whether  a pitch will be a strike, he updates  his  prediction continuously during  the  bali's flight.

In fact, many  problems  that  are  classically cast  as single-step prediction problems  are  more naturally viewed  as  nnflti-step problems.  Perceptual learning  problems,  such  as  vision or  speech  recognition, are  classically treated as supervised learning, using  a training set  of  isolated, correctly-classified input patterns. When  humans hear  or  see  things, on the  other hand,  they  receive a stream  of  input over  time  and  constantly update  their hypotheses abo~t  what they  are  seeing or  hearing. People  are  faced not  with  a single-step problem  of unrelated pattern  class pairs, but  rather with  a series of  related patterns, all providing information about  the  same classification. To disregard this structure  seems improvident.

## 2.2  Computational issues

In  this subsection, we introduce a particular TD  procedure  by formally relating it  to  a classical supervised-learning procedure, the  Widrow-Hoff rule.

We  show that  the  two procedures produce  exactly the  same weight  changes, but that  tile TD  procedure  can be implemented  incrementally and therefore requires far  less  computational power. In the  following subsection, this  TD procedure  will  be used also  as a conceptual bridge  to  a larger family  of  TD procedures that  produce  different weight  changes  than  any  supervised-learning method. First, we detail the  representational assumptions  that  will  be used throughout the  paper.

We  consider multi-step prediction problems in  which experience comes in observation-outcome sequences of  the  form x  l, x2,  x 3 , . . . , xm,  z, where each xt is  a vector of  observations available at  time  t in  the  sequence, and z is  the outcome of  the  sequence. Many  such  sequences will  normally  be experienced. The components of  each xt are  assumed to  be real-valued measurcxnents  or features, and z is  assumed to  be a real-valued scalar. For each  observationoutcome sequence,  the  learner produces  a corresponding sequence  of  predictions P1,  P2,  P3  . . . . , Pro,  each  of  which is  an estimate of  z.  In  general, each Pt can  be a function of  all  preceding observation vectors up through  time  t,  but, for simplicity, here  we assume that it is  a  flmction only  of zt. 1 The predictions are  also  based  on a vector of  modifiable parameters or  weights, w. Pt's functional dependence  on xt and w will sometimes  be  denoted  explicitly by writing it  as P(xt, w).

All  learifing procedures will  be expressed as  rules for  updating  w.  For  the moment we assume that  w is updated  only  once  for each  complete  observationoutcome sequence  and thus  does not change during  a sequence. For each observation, an.increment to  w, denoted Awt, is  determined. After  a  complete sequence  has  been  processed, w  is  changed  by (tile sum of)  all the  sequenee's in('renleIlt s:

<!-- formula-not-decoded -->

Later,  we  will  consider more incremental cases  in  which w  is  updated  after each  observation, and also  less  incremental cases  in  which it  is  updated  only after accumulating Awt's over  a training set consisting of  several sequences.

The supervised-learning approach  treats each  sequence  of  observations and its outcome as a sequence  of  observation-outcome pairs; that  is, as  the  pairs (xl, z), (x2, z)  ..... (x~,  z).  The increment due to  time  t  depends  on the  error between Pt and z,  and on how  changing  w  will  affect Pt. The prototypieal supervised-learning update  procedure is

<!-- formula-not-decoded -->

where a is  a  positive parameter affecting tile rate of  learning, and the  gradient, V~Pt, is  the  vector of  partial derivatives of Pt with  respect to  each  component of  w.

For  example,  consider the  special case  in  which Pt is  a linear function of zt and w, that  is, in  which Pt =  wTxt =  ~i w(i)xt(i), where w(i) and xt(i) are

1The other  eases  can be reduced  to  this one by reorganizing the  observations in  such a way that  each  xt  includes some or all of  the  earlier observations. Cases  in  which  predictions should  depend on t  can also  be reduced  to  this one by including t  as  a component of  x~.

the  i th  components  of  w and  ¢t, respectively. 2 In  this case  we have  V,~Pt  = xt, and (2)  reduces to  the  well  known Widrow-Hoff  rule  (Widrow &amp;  Hoff,  1960):

<!-- formula-not-decoded -->

This  linear learning method is  also known as  the  "delta rule," the  ADALINE, and the  LMS  filter. It  is  widely  used  in  connectionism, pattern recognition, signal processing, and adaptive control. The basic  idea  is  that  the  difference z  - wTxt represents the  scalar error between  the  prediction, wTxt, and what it  should  have been,  z.  This is  multiplied by the  observation vector xt to determine the  weight  changes  because xt indicates how changing  each  weight will affect the  error. For  example,  if the  error is  positive and xt(i) is  positive, then wi(t) will  be increased, increasing wTxt and reducing  the  error.  The Widrow-Hoff rule  is  simple,  effective, and robust.  Its  theory  is  also  better developed than  that of  any  other learning method (e.g., see  Widrow &amp; Stearns, 1985).

Another instance of  the  prototypical supervised-learning procedure  is  the "generalized  elta  rule," or backpropagation procedure, of  Rumelhart  et  al. (1985). In this ease,  Pt is  computed by a multi-layer connectionist network and is  a nonlinear flmction of  xt  and w.  Nevertheless, the  update  rule  used is  still exactly (2), just  as  in  the  Widrow-Hoff  rule, the  only  difference being that  a more complicated process  is  used  to  compute the  gradient V,,Pt.

In  any  case, note  that all Awt in  (2)  depend  critically on z,  and  thus  cannot be determined  until the  end of  the  sequence  when  z becomes known.  Thus, all observations and predictions made during  a sequence  must be remembered until its end,  when all the  Awt's  are  computed.  In  other words,  (2)  cannot  be computed incrementally.

There is, however,  a TD  procedure  that  produces  exactly the  same result as  {2), and yet  which can be computed incrementally. The key  is  to  represent the  error z  - Pt  as  a sum of  changes  in  predictions, that  is, as

<!-- formula-not-decoded -->

Using  this, equations (1)  and (2)  can  be combined as

<!-- formula-not-decoded -->

a.

2wT iS  the  transpose of the  column vector  w.  Unless  otherwise noted,  all vectors  are column vectors.

In  other words,  converting back  to  a rule to  be used  with  (1):

<!-- formula-not-decoded -->

Unlike  (2), this equation can  be  computed incrementally, because  each Awt depends only  on a pair  of  successive predictions and on the  sum  of  all  past values  for V~,Pt. This  saves  substantially on memory, because  it  is  no longer necessary to  individually remember all past  values of VwPt. Equation  (3)  also makes nmch  milder  demands on the  computational speed  of  the  device  that implements  it; although  it  requires slightly more arithmetic operations overall (the  additional ones are  those  needed  to  accumulate t ~k=l VwPk), they  can be distributed over  time  more evenly.  Whereas (3)  computes one increment to  w  on each  time  step, (2)  must wait  until a sequence  is  completed  and then compute all  of  the  increments due to  that  sequence.  If  M is  the  maximum possible length  of  a sequence, then  under  many circumstances (3)  will  require only  1/Mth of  the  memory  and speed  required by (2). 3

For  reasons that  will  be made clear shortly, we refer to  the  procedure given by (3)  as the  TD(1) procedure.  In addition, we will  refer to  a procedure  as linear if  its predictions Pt are  a linear function of  the  observation vectors zt and the  vector of  memory  parameters w, that  is, if Pt  =  wTxt. We  have  just proven:

Theorem 1 On  multi-step prediction problems,  the  linear  TD(1) procedure produces  the  same per-sequence weight  changes  as  the Widrow-Hoff  procedure. Next,  we  introduce a family  of  TD  procedures that  produce  weight  changes different from those  of  any supervised-learning procedure.

## 2.3  The TD(A)  family  of  learning procedures

The hallmark  of  temporal-difference methods is  their sensitivity to  changes in  successive predictions rather than  to  overall error between  predictions and the  final outcome.  In  response to  an increase (decrease) in  prediction fl'om Pt to Pt+l, an increment Awt is  determined  that  increases (decreases) the  predictions for  some or  all of  the  preceding observation vectors Xl,... ,zt. The procedure given  by (3)  is  the  special case  in  which all of  those  predictions are altered to  an  equal  extent. In  this article we also  consider a class of  TD  procedures  that  make greater alterations to  more recent predictions. In  particular, we consider an exponential weighting with  recency, in  which alterations to  the predictions of  observation vectors occurring k steps  in  the  past  are  weighted according to  A  k for 0 &lt; A &lt;  1:

<!-- formula-not-decoded -->

3Strictly speaking,  there  are  other  incremental procedures  for  implementing  the combination  of (1) and  (2),  but only tile TD rule (3) is  appropriate for  updating  w  on  a per-observation basis.

Note  that for  ~ --  1  tills is  equivalent to  (3), the  TD  implementation of  the  prototypical supervised-learning method.  Accordingly, we call this new procedure TD(A)  and we will refer to  the  procedure given  by (3)  as  TD(1).

Alterations of  past  predictions can  be weighted  in  ways other than  the  exponential form given  above,  and this may be appropriate for  particular applications. However, an important advantage  to  the  exponential form is  that  it can be computed incrementally. Given that et is  the  value  of  the  sum in  (4) for  t, we can  incrementally coml)ute et+l, using  only  current information, as

<!-- formula-not-decoded -->

For  ~ &lt;  1,  TD(A)  produces weight  changes  different fl'om those made by any supervised-learning method. The difference is  greatest in  the  case  of  TD(0) (where  A = 0),  ill which the  weight  increment is  determined only  by its effect on the  prediction associated with  the  most recent observation:

<!-- formula-not-decoded -->

Note  that this procedure is formally very  similar to  the  prototypical supervisedlearning procedure  (2).  The two equations are  identical except  that the  actual outcome  z  in  (2)  is  replaced by the  next  prediction Pt+a in  the  equation above. Tile  two methods  use  the  same learning mechanism, but  with  different errors. Because  of  these relationships and  TD(0)'s overall simplicity, it is an  important focus  here.

## 3.  Examples of  faster learning with TD  methods

In  this section we begin  to  address the  claim  that  TD  methods  make more efficient use  of  their experience than  do  supervised-learning methods,  that they converge  more rapidly and  make more accurate predictions along  the  way. TD methods  have  this advantage whenever  the  data  sequences have  a certain statistical structure that  is  ubiquitous in  prediction problems. This structure naturally arises whenever the  data  sequences  are  generated by a dynamical system, that  is, by a system  that has  a  state which  evolves and is  partially revealed over  time.  Almost  any  real system  is  a  dynamical  system, including the weather, national economies, and chess  games. In  this section, we develop  two illustrative examples:  a game-playing example  to  help  develop  intuitions, and a random-walk  example  as  a simple  demonstration with  experimental results.

## 3.1  A  game-playing  example

It seems  counter-intuitive that TD  methods  might  learn more efficiently than supervised-learning methods.  In  learning to  predict an outcome,  how can  one

Figure  1. A  game-playing example showing the  inefficiency of  supervised-learning methods.  Each circle represents a position or  class of  positions from  a  twoperson  board  game. The "bad"  position is  known from  long  experience to lead  90%  of  the  time to  a  loss and only  10%  of  the  time  to  a  win.  The first game in  which the  "novel" position occurs evolves as  shown by  the  dashed arrows.  What  evaluation should  the  novel  position receive as  a result of this experience?  Whereas TD  methods correctly conclude  that  it  should be considered another bad state, supervised-learning methods associate it fully with  winning, the  only  outcome  that has  followed  it.

<!-- image -->

do better than by knowing and using  the actual  outcome as a performance standard?  How  can using  a biased  and potentially inaccurate subsequent  prediction possibly be a better use of  the  experience?  The following example is meant to  provide  an intuitive understanding of  how  this is  possible.

Suppose there  is  a game position that  you have  learned is  bad for  you,  that has  resulted most  of  the  time  in  a  loss and only  rarely in  a  win  for your  side.  For example,  this position might  be a backgammon  race  in  which you are  behind, or a disadvantageous configuration of  cards  in  blackjack. Figure  1 represents a simple  case  of  such a position as a single  "bad" state that  has led  90%  of the  time  to  a loss and only  10%  of  the  time  to  a win. Now  suppose  you play a game  that  reaches  a novel  position (one  that  you have never  seen  before), that  then  progresses to  reach  the  bad state, and that  finally ends  nevertheless in a victory for  you.  That is,  over  several moves it  follows  the  path  shown by dashed lines  in Figure  1. As  a result of  this  experience, your opinion  of tile bad state would presumably  improve,  but  what of  the novel state?  What value  would you associate with it as  a result of  this  experience?

A  supervised-learning method would form a pair  from the  novel  state and the  win that  followed  it, and would conclude  that. the  novel  state is  likely to lead  to a win.  A  TD  method, on the other  hand, would form a pair  from the novel  state and the bad state  that immediatehj followed  it,  and would conclude  that  the  novel  state is  also  a bad one,  that  it  is  likely to  lead  to  a

<!-- image -->

Figure 2.  A generator of  bounded  random walks.  This  Markov process generated the data  sequences in  the  example.  All  walks  begin  in  state D.  From states B, C, D, E, and F, the  walk has  a 50%  chance  of  moving either to  the right or  to  the  left. If either edge  state, A or  G,  is  entered, then  the  walk terminates.

loss. Assuming we have  properly classified the "bad"  state, the  TD  method's conclusion is  the  correct one;  the  novel  state led  to  a position that  you know usually leads  to  defeat; what happened  after that  is  irrelevant. Although  both methods  should  converge to  the  same evaluations with  infinite experience, the TD  method learns a better evaluation from  this limited experience.

The TD  method's  prediction would also  be better had the  game been lost after reaching  the bad state, as is  more likely. In this  case,  a supervisedlearning method would tend  to  associate the  novel  position fully with  losing, whereas  a TD  method would tend  to  associate it with  the  bad position's 90% chance  of  losing, again  a  presumably  more accurate assessment. In  either case, by  adjusting its evaluation of  the  novel  state towards  the  bad  state's evaluation, rather than  towards  the  actual outcome,  the  TD  method makes better use  of the  experience. The bad state's evaluation is  a better performance  standard because  it  is  uncorrupted by random factors that  subsequently influence the final outcome. It  is  by eliminating this source  of  noise  that  TD  methods can outperform supervised-learning procedures.

In  this example,  we have ignored  the  possibility that  the  bad state's previously  learned evaluation is  in  error. Such errors will  inevitably exist and will affect the  efficiency of  TD  methods in  ways that  cannot  easily be evaluated in  an example of  this sort.  The example  does  not  prove  TD  methods will  be better on balance, but  it  does  demonstrate that  a subsequent  prediction can easily be a better performance  standard than  the  actual outcome.

This  game-playing example can  also  be used  to  show how TD  methods can fail. Suppose the  bad state is  usually followed  by defeats except when  it  is preceded  by the  novel  state, in  which case  it  always leads  to a victory. In this odd case, TD  methods  could  not  perform  better and might  perform  worse than  supervised-learning methods. Although  there  are  several techniques for eliminating or  minimizing  this sort  of  problem,  it  remains  a greater difficulty for  TD  methods than  it  does  for  supervised-learning methods. TD  methods try  to  take  advantage  of  the  information provided  by the  temporal  sequence of  states, whereas  supervised-learning methods ignore it. It  is  possible for  this information to  be misleading, but  more often it  should  be  helpful.

Finally, note that  although  this  example involved  learning an evaluation function, nothing  about  it  was specific to  evaluation functions. The methods can equally well  be used  to  predict outcomes  unrelated to  the  player's goals, such  as  the  number of  pieces left at  the  end of  the  game. If  TD  methods are more efficient than  supervised-learning methods in  learning evaluation functions, then  they  should  also  be more efficient in general  prediction-learning problems.

## 3.2  A  random-walk  example

The game-playing  example is  too  complex to  analyze  in  great  detail. Previous  experiments with TD  methods have also  used complex domains (e.g., Samuel,  1959;  Sutton, 1984;  Barto  et  al., 1983;  Anderson,  1986,  1987).  Which aspects of  these  domains can be simplified or  eliminated, and which aspects are  essential in  order  for  TD  methods to  be effective? In  this paper,  we propose that  the  only  required characteristic s  that  the  system  predicted be a dynamical  one,  that it  have  a state which can  be observed  evolving over  time. If this is  true, then  TD  methods  should  learn more efficiently than  supervisedlearning methods  even  on  very  simple  prediction problems, and this is  what we illustrate in  this subsection. Our example is  one of  the  simplest of  dynamical systems, that  which generates bounded random walks.

A  bounded random walk is  a state sequence  generated by taking  random steps  to  the  right or  to  the  left until a boundary  is  reached. Figure  2 shows a system  that  generates such  state sequences.  Every  walk begins  in  the  center state D. At each  step the  walk  moves to  a  neighboring state, either to  the  right or  to  the  left with  equal  probability. If  either edge state (A or  G) is  entered, tile walk ternfinates. A  typical walk might be DCDEFG. Suppose we wish to  estimate the  probabilities of  a walk ending  in  the  rightmost state, G, given that  it is  in  each  of  the  other  states.

We  applied linear supervised-learning and TD  methods to  this problem in a straightforward way. A  waifs  outcome was defined to  be z =  0 for  a walk ending on the left at A  and z =  1 for  a walk ending on the right  at G. The learning methods estimated  the  expected  value  of  z;  for  this  choice  of z,  its expected  value  is  equal  to  the  probability of  a right-side termination. For each nonterminal state i,  there  was a corresponding observation vector xi; if  the  walk was  in state  i at time t then xt =  xi. Thus; if  the  walk DCDEFG occurred, then  tile learning procedure would be given  the  sequence XD, xc,  XD,XE, xr,  1. The vectors {xi}  were the  unit  basis  vectors of  length 5,  that  is, four  of  their components were 0 and the  fifth was  1 (e.g., XD = (0, 0,  1, 0,  0)T), with  the  one  appearing at, a different component for  each  state. Thus, if  the  state the  walk was in  at  time  t  has its 1 at  the  i th component of  its observation vector, then  the  prediction Pt =  'wTxt was simply  the  value of  the  ith  component of  w. We  use  this  particularly simple  case  to  make  this exainple as  clear as  possible. The theorems  we prove  later for  a more general class of  dynanfical systems  require only  that the  set of  observation vectors {xi  } be linearly independent.

Two computational experiments were  performed  using  observation-outcome sequences generated as  described above.  In  order to  obtain statistically reliable

E R R O R

<!-- image -->

A

Figure  3. Average  error on the  random-walk  problem  under  repeated presentations. All  data are  from  TD(A) with  different values of  A. The dependent  measure used  is the  RMS  error between  the  ideal predictions and  those found  by  the learning procedure  after being  repeatedly presented with  the  training set until convergence of  the  weight  vector. This  measure  was averaged  over 100 training sets  to  produce  the  data  shown.  The A =  1 data  point  is the  performance level attained by the  Widrow-Hoff  procedure.  For each data  point, the  standard error is  approximately ~ = 0.01, so  the  differences between  the  Widrow-Hoff  procedure and the  other  procedures are  highly significant.

results, 100 training sets, each  consisting of  10  sequences, were constructed for use by all  learning procedures.  For all  procedures, weight increments  were computed according  to TD(A), as given  by (4). Seven different values  were used  for A. These  were  A =  1,  resulting in  the  Widrow-Hoff  supervised-learning procedure, A =  0,  resulting in  linear TD(0),  and also  A =  0.1, 0.3, 0.5, 0.7,  and 0.9,  resulting in  a range  of  intermediate TD  procedures.

In the first experiment, the  weight  vector  was not updated after each sequence  as  indicated by (1).  Instead, the  Aw's were  accumulated over  sequences and only  used  to  update  the  weight  vector  after the  complete  presentation of a training set.  Each training set  was presented  repeatedly to each learning procedure  until the  procedure  no longer  produced any significant changes  in tile weight  vector. For  small  a,  the  weight  vector always  converged  in  this way, and always  to  the  same final value,  independent, of  its  initial value.  We  call this  the repeated  presentation8 training paradigm.

The true  probabilities of  right-side termination the ideal  predictions -for each of  the  nonterminal  states can be computed as described in  section 4.1. and 65.   for  states B, C, D, E  and F, respectively. As Theseare ~, ~, 1, 3

Figure 4. Average  error on random walk problem  after experiencing 10 sequences. All  data  are  from  TD(~) with  different values of  a and A. The dependent measure  is  the  RMS error between  the  ideal predictions and those  found by the  learning procedure  after a single  presentation of  a training set. This  measure  was averaged over  10O training sets. The )~  =  t  data  points represent performances of  the  Widrow-Hoff  supervised-learning procedure.

<!-- image -->

a measure of  the  performance  of  a learning procedure  on a training set,  we used  the  root  mean squared  (RMS)  error between  the  procedure's asymptotic predictions using  that  training set  and the  ideal  predictions. Averaging  over training sets, we found that  performance  improved  rapidly as A was reduced below 1 (the  supervised-learning method) and was best  at  ,~  = 0 (the  extreme TD  method),  as  shown in  Figure  3.

This result contradicts conventional wisdom.  It  is  well  known  that, under repeated presentations, the  Widrow-Hoff  procedure  minimizes  the  RMS error between  its predictions and the  actual outcomes  in  the  training set, (Widrow &amp; Stearns, 1985).  How  can  it be that  this optimal  method performed  worse  than all the  TD  methods for  A &lt;  1? The answer  is  that  the  Widrow-Hoff  procedure only  minimizes  error on the  training set; it  does  not  necessarily minimize  error for  future experience. In  the  following section, we prove  that  in  fact it  is  linear TD(0)  that  converges  to what can be considered  the optimal  estimates for matching  future  experience - those  consistent with the  maximum-likelihood estimate of  the  underlying Markov process.

The  second experiment  concerns  the question  of learning rate  when  the training set  is  presented just  once rather than repeatedly until  convergence. Although  it is  difficult to  prove  a  theorem  concerning learning rate, it is  easy  to perform  the  relevant computational experiment.  We  presented the  same data to the  learning procedures, again  for  several values  of  A. with the  following

Figure 5. Average  error at  best  ~ value  on random-walk  problem.  Each data  point represents the  average over  100 training sets of  the  error in  the  estimates found  by TD()~),  for  particular A and a values, after a single presentation of  a  training set. The ,~  value is  given  by the  horizontal coordinate. The a value  was selected from  those  shown in  Figure 4 to  yield the  lowest  error for  that  ), value.

<!-- image -->

procedural changes.  First, each training set  was presented once to  each  procedure.  Second,  weight  updates  were  performed  after each  sequence, as  in  (1), rather than  after each  complete  training set.  Third,  each  learning procedure was applied with  a range  of  values  for  the  learning-rate parameter  a.  Fourth, so  that there  was no bias  either toward  right-side or  left-side terminations, all components of  the  weight  vector were initially set  to  0.5.

The  results for  several representative values  of  ~  are  shown  in Figure  4. Not surprisingly, the  value  of  a had a significant effect on performance, with best  results obtained  with intermediate values.  For all  values, however,  the Widrow-Hoff  procedure, TD(1),  produced  the  worst  estimates. All  of  the  TD methods  with  A &lt;  1  performed  better both  in  absolute terms  and over  a wider range  of  a values than  did  the  supervised-learning method.

Figure  5 plots the  best  error level achieved  for  each  .~  value, that  is, using the a value  that  was best  for  that  I value.  As in  the  repeated-presentation experiment, all ~ values  less than  1  were  superior to  the  ,~  =  1 case.  However, in  this experiment  the  best  ~ value  was not  0,  but  somewhere near  0.3.

One reason  I = 0 is  not  optimal for  this problem  is  that  TD(0) is  relatively slow at propagating prediction levels back along  a sequence. For example, suppose  states D, E, and F  all  start with the  prediction value  0.5,  and the sequence  xD,  XE, XF, 1 is  experienced. TD(0) will  change  only  F's  prediction, whereas  the  other  procedures will  also  change E's  and D's  to  decreasing ex-

tents.  If  the  sequence  is  repeatedly presented, this  is  no handicap,  as the change  works back an additional step  with  each  presentation, but  for  a single presentation it  means slower  learning.

This handicap could be avoided  by working backwards through  the sequences.  For  example,  for the  sequence xu,  x¢,  xr~  1~  first F's  prediction could be updated  in  light of  the  1,  then  E's  prediction could  be  updated  toward  F's new  level, and so on.  In this  way  the  effect of  the 1 could  be propagated back to  the  beginning  of  the  sequence  with  only  a single presentation. The drawback to  this  technique is  that  it  loses the  implementation advantages  of TD  methods. Since  it  changes  the  last prediction in  a sequence  first, it  has no incremental implementation. However, when this  is  not  an issue, such as when learning is  done offine from an existing database, working  backward in this way should  produce  the  best  predictions.

## 4. Theory of temporal-difference methods

In this section, we provide  a theoretical foundation for  temporal-difference methods.  Such a foundation is  particularly needed  for  these  methods because most of  their learning is  done on the  basis  of  previously learned  quantities. "Bootstrapping" in  this  way may  be what makes TD  methods efficient, but it  can also  make them difficult to  analyze and to  have  confidence in.  In  fact, hitherto no TD  method has ever  been proved stable or convergent  to the correct predictions. 4  Tile  theory  developed  here  concerns  the  linear TD(0) procedure and a class of  tasks  typified by the  random walk example  discussed in  the  preceding section. Two  major  results are  presented: (1)  an asymptotic convergence theorem  for linear TD(0) when presented with  new data  sequences; and (2)  a theorem  that  linear TD(0) converges  under  repeated presentations to  the  optimal  (maximum  likelihood) estimates. Finally, we discuss how  TD methods can be viewed as  gradient-descent procedures.

## 4.1  Convergence of  linear TD(0)

The theory  presented here  is  for  data  sequences  generated  by absorbing Markov procea~e8 such as the  random-walk process  discussed in  the  preceding  section.  Such processes, in  which each next  state depends only  on the current state, are  among  the  formally simplest dynamical  systems.  They are defined by a set  of  terminal states T, a set  of  nonterminal states N, and a set of  transition probabilities pij  (i E N, j E N  t\_) T),  where  each  P~3 is  the  probability of  a  transition from  state i  to  state j,  given  that the  process is  in  state i. The  "absorbing" property means that  indefinite cycles among  the  nonterminal  states are  not  possible; all sequences  (except for  a set  of  zero  probability) eventually terminate.

Given an initial state ql,  an absorbing Markov  process  provides a way of generating a state sequence  ql, q2~.-., qm+l, where  qm+l E T. We  will assume the  initial state is  chosen  probabilistically from  among the  nonterminal states,

aWitten  (1977) presented a  sketch of  a  convergence proof for a  TD  procedure that predicted discounted costs in  a  Markov decision problem, but  many steps were  left out, and  it now appears that the  theorem  he  proposed is not  true.

each  with  probability lti. As in  the  random walk  example,  we do not  give  the learning algorithms direct knowledge  of  the  state sequence, but  only of  a  related observation-outcome sequence Xl, xu,... ,Xm,  z. Each numerical  observation vector xt is  chosen  dependent  only  on the  corresponding nonterminal state qt, and the  scalar outcome z is  chosen  dependent  only  on the  terminal state qm+l. In  what  follows, we assume  that there is a  specific observation vector xi corresponding to  each  nonterminal state i  such  that if qt  = i, then xt = xi.  For each  nonterminal state j,  we assume  outcomes  z are  selected from  an  arbitrary probability distribution with  expected value  2j.

The first step  toward  a  formal  understanding of  any  learning procedure is  to prove  that it converges asymptotically to  the  correct behavior with  experience. The desired behavior in  this case  is to  map each  nonterminal state's observation vector  xi to  the  true  expected  value  of  the  outcome z given  that  the  state sequence  is  starting in  i.  That is, we want the  predictions P(xi, w) to  equal E {z l  i},  Vi E  N.  Let us call these  the ideal predictions. Given complete knowledge  of  the  Markov process, they  can  be computed as  follows:

<!-- formula-not-decoded -->

For any matrix  M,  let  [M]o denote  its ij th component,  and,  for  ally vector v,  let  [vii denote  its i tl*  component. Let Q  denote  the  matrix  with  entries [Q],j =  pij  for i,j E N, and let  h denote  the  vector with  components [h]i = ~jeT  P~J23 for  i  E N. Then we can  write the  above  equation as

<!-- formula-not-decoded -->

The second  equality and the  existence of  the  limit and the  inverse are  assured by Theorem A.1. 5 This  theorem  can  be applied here  because  the  elements of Qk  are  the  probabilities of  going  from one nonterminal state to  another in  k steps; for  an absorbing Markov process, these  probabilities must all converge to  0 as  k ~  c~.

If the  set  of  observation vectors {  xi  I i  E N  }  is  linearly independent, and if is chosen  small enough,  then  it is known that  the  predictions of  the  WidrowHoff  rule converge  in  expected value  to  the  ideal predictions (e.g., see  Widrow &amp;  Stearns, 1985). We  now prove  the  same result for  linear TD(0):

Theorem  2 For any absorbing Markov chain, for  any  distribution of  starting probabilities pi,  for  any outcome distributions with  finite expected  values  2j, and for  any linearly independent  set  of  observation vectors  {  xi I  i ~  N  }, there  exists an ~ &gt;  0 such that, for  all  positive c~ &lt;  e and for  any initial weight  vector, the  predictions of  linear  TD(O)  (with weight  updates  after each sequence)  converge  in  expected value  to  the  ideal predictions (5). That is,  if

~To simplify presentation of  the  proofs, some of  the  more straightforward but  potentially distracting steps have  been  placed in  the  Appendix  as  separate theorems. These  are  referred to  in  the  text as  Theorems A.I,  A.2,  and A.3.

Wn denotes  the weight  vector  after n  sequences  have been experienced, then lim~\_~o¢ E  {xT~n} =  E{z  Ii}  =  [(I  Q)lh]~,  vi  e N.

PROOF: Linear  TD(0) updates  w~  after each sequence  as  follows, where m denotes  the  number of  observation vectors in  the  sequence:

<!-- formula-not-decoded -->

where  Xq~ is the  observation vector corresponding to  the  state qt entered at  time t  within  the  sequence.  This  equation groups  the  weight  increments according to  their time  of occurrence within the  sequence.  Each increment corresponds to a  particular state transition, and so  we can  alternatively group  them according to  the  source  and destination states of  the  transitions:

<!-- formula-not-decoded -->

where ~/ij denotes  the  number of  times  the  transition i ~ 3 occurs  in the sequence.  (For  j E T,  all but  one  of  the rh3 is  0.)

Since  the  random processes generating state transitions and outcomes are independent of  each  other, we can  take  the  expected value  of  each  term  above, yielding

<!-- formula-not-decoded -->

where d~ is  the  expected number of  times  the  Markov chain  is  in  state i  in  one sequence, so  that d, pij is  the  expected value  of rhj. For  an absorbing Markov chain  (e,g., see  Kemeny  &amp;  Snell, 1976,  p.  46):

<!-- formula-not-decoded -->

where [d]{ = d,  and [#]{ = #{,  i  6 N.  Each di  is  strictly positive, because  any state for which d{ = 0 has  no  probability of  being  visited and can  be  discarded.

Let wn  denote  the  expected  value  of wn. Then, since  the  dependence  of E{w,~+l I  wn}  on w,~ is  linear, we can  write

<!-- formula-not-decoded -->

an iterative update  formula  in @n  that  depends only  on initial conditions. Now  we rearrange terms  and  convert to  matrix and  vector notation, letting D denote  the  diagonal matrix  with  diagonal entries [D]i~  = di and X  denote  the matrix  with  columns  xi:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assuming for  the  inoment  that limn-~o¢ (I  - aXTXD(1 - Q))n = 0,  then, by theorem  A.  1,  the  sequence  {X  Tw~ }  converges to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which is  the  desired result. Note that  D -1 must exist because  D  is  diagonal with  all positive diagonal entries, and (XTX)  -1 must exist by Theorem A.2.

It thus  remains to  show that limn-~o~ (I-aXTXD(I-  Q))n = 0.  We  do  this by first showing  that D(I-Q) is  positive definite, and  then  that XTXD  (IQ) has  a  full set  of  eigenvalues all of  whose  real parts are  positive. This  will enable us  to  show that a can  be  chosen  such  that all eigenvalues of I-c~XTXD(I-Q) are  less than  1  in  modulus,  which  assures us  that its powers  converge.

We  show that D(I  - Q) is  positive definite 6 by applying the  following lemma (see Varga, 1962,  p.  23,  for  a proof):

Lemma If  A  is  a real,  symmetric,  and strictly diagonally  dominant matrix with  positive diagonal  entries, then  A  is  positive definite.

We  cannot  apply  this lemma directly to D(I - Q) because  it is  not  symmetric. However, by Theorem A.3,  any matrix  A  is  positive definite exactly when the symmetric  matrix A + A T is  positive definite, so  we can prove  that D(I - Q) is  positive definite by applying  the  lemma  to  S = D(I - Q) + (D(I  - Q))T. S is  clearly real  and symmetric;  it remains  to  show that  it  has positive diagonal entries and is  strictly diagonally dominant.

First, we note  that

<!-- formula-not-decoded -->

We  will  use  this  fact several times  in  the  following.

S's  diagonal  entries are  positive, because [S]ii= [D(I Q)]~i  +  [(D(I  Q))T]ii  =  2[D(IQ)]ii  =  2 d i [ I Q]ii =  2di(1pii) &gt; 0,  i  E N. Furthermore, S's  off-diagonal entries are  nonpositive, because,  for i ~  j,  [S]ij  =  [D(IQ)]o + [(D(/-  Q))T]i j = d~[I  - Q]ij  + d j [ I   - Q]ji  =  -dipi3 - djpji  &lt;\_ O.

S is  strictly diagonally dominant if  and only  if  ][S]ii I  \_ &gt;   Z j ¢ i   I[S]OI, for  all i,  with  strict inequality holding  for  at  least one i.  However,  since  [S]ii&gt; 0 and [S]O &lt;  0,  we need only  show that  [S]ii &gt;  -~j¢i[S]ij, in  other  words,  that ~j[S]o &gt; 0,  which can be directly shown:

<!-- formula-not-decoded -->

Furthermore, strict inequality must hold  for  at  least one i, because  #~ must be strictly positive for  at  least one i.  Therefore, S is  strictly diagonally dominant and the  lemma  applies, proving  that  S and D(I-Q) are  both  positive definite.

6A matrix  A is positive definite if and only  if yTAy &gt; 0  for all real vectors y  ¢ 0.

Next we show that XTXD(I - Q) has a full set  of  eigenvalues all of  whose real parts are  positive. First of  all, the  set  of  eigenvalues is  clearly full, because the  matrix  is  nonsingular, being  the  product  of  three  matrices, XrX,  D, and I - Q, that  we have already  established as  nonsingular. Let ,~  and y be any eigenvalue-eigenvector pair. Let y  = a +  bi and z = (XTX)-ly ¢  0 (i.e., y =  XTXz). Then

<!-- formula-not-decoded -->

where "*" denotes  the  conjugate-transpose. This  implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since  the  left side  and (Xz)*Xz must both be strictly positive, so must the real part  of  A.

Furthermore, y must also  be an eigenvector of I - aXTXD([ - Q), because (I  -  c ~ X T X D ( I -  Q))y =  y -  c~Ay =  (1  -  ~)y.  Thus, all  eigenvectors of I -  o~XTXD(I  --  Q) are  of  the  form 1  -  aA, where A has positive real  part. For  each A =  a  + bi, a &gt; 0,  if a is  chosen  0 &lt;  a &lt;  a--~,  then  1  - aA will  have modulus  7 less than  one:

<!-- formula-not-decoded -->

2a will  be different for  different ~; choose  e to  be the The criterial value smallest such vMue. Then, for  any positive a  &lt;  e, all eigenvalues 1 -  aA of I - c~XD(I -  Q.)X  T are  less than  one in  modulus. And  this  immediately implies (e.g., see  Varga,  1962,  p.  13)  that  lim~-~oo(I -o~XD(I - Q)xT) n =  O, completing  the  proof. ·

We  have just  shown  that  the  expected  values  of  the  predictions found by linear TD(0) converge  to the ideal  predictions for  data  sequences  generated by absorbing  Markov  processes.  Of course,  just  as with the Widrow-Hoff procedure, the  predictions themselves do not  converge; they  continue  to  vary around  their expected  values  according to  their most recent experience. In  the case  of  the  Widrow-Hoff  procedure, it  is  known  that  the  asymptotic variance of  the  predictions is  finite and can be made  arbitrarily small  by the  choice of  the  learning-rate parameter  c~. Furthermore, if  a  ]s  reduced  according  to 1  then  the  variance converges  to  zero  as an appropriate schedule, e.g., a =  ~,

7The modulus of  a  complex  number a  + bi is  x / ~ .

well.  We  conjecture that  these  stronger forms  of  convergence hold  for  linear TD(0) as  well, but  this remains  an open  question. Also  open is  the  question of convergence of  linear TD(A) for  0 &lt; A &lt;  1.  We  now know that both  TD(0) and TD(1) the  Widrow-Hoff  rule  -  converge in  the  mean to  the  ideal predictions; we conjecture that  the  h~termediate TD(A) procedures do as  well.

## 4.2  Optimality  and learning  rate

The result obtained  in the  previous  subsection assures  us that  both TD methods  and  supervised learning methods  converge  asymptotically to  the  ideal estimates for  data  sequences  generated by absorbing Markov processes. However, if both  kinds  of  procedures converge  to  the  same result, which gets  there faster? In  other words,  which kind  of  procedure makes the  better predictions from a finite rather than  an infinite amount of  experience? Despite  the  previously noted  empirical results showing  faster learning with  TD  methods,  this has not been proved  for  any general  case. In this  subsection we  present  a related formal  result that  helps  explain the  empirical result of  faster learning with  TD  methods. We  show that  the  predictions of  linear TD(0) are  optimal in  an important sense  for  repeatedly presented finite training sets.

In the  following, we first define  what we  mean  by optimal  predictions for finite training sets. Though optimal, these predictions are  extremely expensive to  compute,  and neither TD  nor  supervised learning methods compute them directly. However, TD  methods do have a special relationship with them. One common  training process  is  to  present  a finite amount of  data  over  and over  again  until the  learning process  converges  (e.g., see  Ackley,  Hinton,  &amp; Sejnowski,  1985; t-~.umelhart, Hinton,  &amp;  Williams,  1985). We  prove  that  linear  TD(0) converges  under this repeated  preseutations training paradigm to the  optimal predietions~ while  supervised-learning procedures converge  to  suboptimal  predictions. This  result also  helps  explain TD  methods'  empirically faster learning rates.  Since  they  are  stepping toward a better final result, it makes sense  that  they  would also  be better after the  first, step.

The word optimal can  be  misleading because  it suggests a universally agreed upon criterion for  the  best  way of  doing  something.  In  fact, there  are  many kinds  of  optimality, and choosing  anlong tlmm  is  often  a critical decision. Suppose that  one observes  a training set. consisting of a finite number  of observation-outcome sequences, and that  one knows tile sequences  t,o be generated  by,  an absorbing Markov process  as  described in  the  previous section. What  might one mean by the  %est'"  predictions given  such  a training set'?

If  the a priori distribution of  possible Markov  processes is  known,  then the  predictions that  are  optimal  in  the  mean  square  sense  can be calculated through  Bayes's rule. Unfl)rtunately, itis  very  difficult to  justify any a priori assumptions  about  possible Markov processes. In  order  to  avoid  making any such assumptions, mathematicians  have developed  another  kind of  optimal estirnate, known  as the maximum-likelihood estimate.  This is  the kind of optimality with  which we will  be concerned.  For  example,  suppose  one flips a coin  ten times  and gets  seven  heads.  What  is  the  best  estimate of the probability of  getting a head on tile next  toss?  In  one  sense, the  best  estimate depends  entirely on a  priori assmnptions about  how likely one  is to  rm~ into fair

and  biased coins, and  thus  cannot be  uniquely determined. On the  other hand, the  best answer  in the  maximum-likelihood sense  requires no  such  assumptions; it  is  simply  ~.  In  general, the  maximum-likelihood estimate of  the  process that produced  a  set of  data is  that process whose probability of  producing the data  is the  largest.

What is the  maximum-likelihood estimate for our  prediction problem?  If the observation vectors xi  for each  nonterminal state i  are  distinct, then  one  can enumerate  the  nonterminal states appearing in  the  training set  and  effectively know which  state the  process is  in  at  each  time.  Since  terminal states do not produce  observation vectors, but  only  outcomes, it is not  possible to  tell when two sequences end in  the  same terminal state; thus  we will assume that  all sequences terminate in  different states, s

Let  T and  N  denote the  sets of  terminal and  nonterminal states, respectively, as  observed in  the  training set.  Let  [~)]~j = ~j (i,j E N) be the  fraction of the  times  that  state i  was entered in  which a transition occurred  to  state j. Let  z)  be the  outcome  of  the  sequence  in  which  termination occurred at state j e T, and let [~t], = ~je~  15~jzj, i  e N.  (~  and ~t  are  the  maximumlikelihood estimates of  the  true process parameters Q and  h.  Finally, estimate the  expected value  of  the  outcome  z,  given  that the  process is  in  state i  E N, as

<!-- formula-not-decoded -->

That is, choose  the  estimate that  would be ideal if  in  fact the  maximumlikelihood estimate of  the  underlying process  were exactly correct. Let us call these  estimates the optimal predictions. Note that  even  though  ~)  is  an estimated quantity, it still corresponds to  some absorbing Markov chain. Thus, lim,~\_-~c¢ (~n  = 0 and  Theorem A.1 applies, assuring the  existence of  the  limit and inverse in  the  above  equation.

Although  the  procedure outlined above  serves well  as  a  definition of  optimal performance, note  that  it  itself would be impractical to implement. First of  all, it  relies heavily on the  observation vectors xi  being  distinct, and on the  assumption that  they  map one-to-one onto  states. Second,  the  procedure involves keeping  statistics on each pair of  states (e.g., the  :Siy) rather than on each  state or  component  of  the  observation vector. If  n is  the  number of states, then  this procedure requires O(n  2) memory whereas  the  other learning procedures require only O(n) memory. In  addition, the  right side of  (8)  must be  re-computed each  time  additional data  become available and  new estimates are  needed.  This  procedure may require as  much as  O(n  a)  computation per time  step as  compared  to O(n) for  the  supervised-learning and  TD  methods.

Consider the  ease in  which  the  observation vectors are  linearly independent, the  training set  is  repeatedly presented, and the  weights  are  updated  after each  complete  presentation of  the  training set. In  this case, the  Widrow-ttoff

aAlternatively, we may  assume that  there  is  only  one terminal state and that  the  distribution  of  a sequence's outcome depends  on its  penultimate state. This  does  not  change  any of  t h e   c o n c l u s i o n s of  the  analysis.

procedure  converges so as to  minimize  the  root  mean  squared  error between its predictions and the  actual outcomes  in  the  training set  (Widrow &amp;  Stearns, 1985). As illustrated earlier in  the  random-walk example,  linear TD(0) converges  to  a different set  of  predictions. We  now  show that  those  predictions are  in  fact the  optimal  predictions in  the  maximum-likelihood sense  discussed above.  That is, we prove  the  following theorem:

Theorem  3 For any training set  whose observation vectors  {  x~  [  i E  2V  } are  linearly independent, there  exists an e &gt; 0 such that, for  all positive a &lt; ( and for any initial weight  vector,  the  predictions of  linear  TD(O)  converge, under repeated presentations of  the  training set  with  weight  updates  after each complete  presentation, to the optimal  predictions (8). That is,  if wn  is  the value of  the  weight  vector  after  the training set  has been presented n  times, then lim,~  x~w~  =  [(/-  (~)-1~],, Vie 5;.

PROOF: The proof  of  Theorem 3 is  almost  the  same as that  of  Theorem 2, so  here  we only  highlight the  differences. Linear  TD(0) updates  wn  afl.er each presentation of  tile training set:

<!-- formula-not-decoded -->

where rn~ is  tile number of  observation vectors  in the s TM sequence  in the training set, Pt s  is the  t gh  prediction in  the  .s th  sequence, and P~  +L is  defined to  be the  outcome of  tile S  t h sequence.  Let ~]ij be the  number of  times  the transition i  ~  j appears  in  the  training set; then  the  sums can  be regrouped as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where  d~  is the number of times state i E  ~V  appears  in the training  set. The  rest  of the proof for  Theorem 2, starting  at (6),  carries  through with estimates substituting for actual values tilroughout..  The  only step in the proof  that  requires  additional support is  to  show  that  (7)  still holds,  i.e., that dT =  ~tT(i  \_ Q)-L, where [/~]i  s  tile mmLber  of  sequences  in  the  training set that  begin  in  state i  E  2).  Note that  ~ e ~ r/~j =  ~\_~ ddS~j  is  tile nmnber of  times  state j appears  in  tile training set  as  the  destination of  a t ransmon. Since  all occurrences of  state j must be  either as  the  destination of  a  transition or  as  the  beginning state of  a  sequence. (t3 =  [/;]3 + ~i dd)zj. Converting this to matrix  notation, we have d  T =  it T + drQ, which yields the  desired conclusion, dT = f~r ( I \_ ~2  )-1 after algebraic manipulations. ·

We  have just  shown  that  if  linear TD(0) is  repeatedly presented  with a finite training set, then  it converges  to  the  optimal  estimates. The WidrowHoff  rule, on the  other hand,  converges to  the  estimates that  minimize  error on

the  training set; as  we saw in  the  random-walk  example,  these are  in  general different from the  optimal  estimates. That TD(0) converges to  a better set  of estimates with  repeated presentations helps  explain how and  why it could  learn better estimates from a single presentation, but  it does  not  prove  that. What is  still needed  is  a  characterization of  the  learning rate of  TD  methods  that can be compared with  those  already available for  supervised-learning methods.

## 4.3  Temporal-difference methods as gradient  descent

Like  many  other  statistical learning methods,  TD  methods can be viewed as gradient descent  (hill climbing)  in  the  space  of  the  modifiable parameters (weights).  That is,  their goal  can be viewed as minimizing  an overall error measure a(w) over  the  space  of  weights  by  repeatedly incrementing the  weight vector  in (an  approximation to)  the  direction in  which a(w) decreases most steeply. Denoting  the  approximation to  this  direction of  steepest descent, or gradient, as  ~7~oJ(w),  such  methods are  typically written as

<!-- formula-not-decoded -->

where (~  is  a positive constant determining step  size.

For a multi-step prediction problem in which Pt =  P(zt, w) is  meant to approximate  E {z I  st},  a natural error measure is  the  expected value  of  the square  of  the  difference between  these  two quantities:

<!-- formula-not-decoded -->

where Ex{  } denotes  the  expectation operator over  observation vectors x. ,l(w) measures the error  for  a weight  vector  averaged  over  all  observation vectors, but  at  each  time  step one  usually obtains additional information about only  a single observation vector. The usual  next  step, therefore, is  to  define a per-observation error measure Q(w, x)  with  the  property that Ex{Q(w, x)}  = J(w). For  a multi-step  rediction problem,

<!-- formula-not-decoded -->

Each time step's weight increments  are  then  determined  using VwQ(w, st), relying on the  fact that  Ex{V,,,Q(w,x)} = VwJ(w), so  that the  overall effect of  the  equation  for  Aw,  given  above can be approximated  over  many  steps using  small  c~  by

<!-- formula-not-decoded -->

The quantity E {zlxt } is  not  directly known  and must be estimated. Depending  on how  this is  done,  one gets  either a supervised-learning method or a TD  method. If E {z  I xt} is  approximated by z,  the  outcome that  actually

occurs  following xt, then we  get  the  classical supervised-learning procedure (2).  Alternatively, if  E  {z]  xt} is  approximated  by P(xt+l, w), the  immediately following prediction, then  we get  the  extreme  TD  method, TD(0). Key to  this analysis is  tile recognition, in  the  definition of J(,w), that  our  real goal is  for  each  prediction to  match the expected .value of  the  subsequent outcome, not the  actual  outconw occurring in the  training set.  TD  methods can perform better than  supervised-learni:lg methods because  the  actual  outcome of a sequence  is  often not the  best  estimate of  its expected  value.

## 5. Generalizations of TD(A)

In this  article, we have chosen  to  analyze  particularly simple  cases  of  temporal-difference m thods. This  has clarified their operation and made  it  possible to  prove  theorems.  However. more realistic problems  may  require more complex TD  methods. In this  section, we briefly explore  some ways in  which the  simple  methods ('an be extended.  Except  where explicitly noted,  the  theorems I/resented earlier do not strictly apply  to  these  extensions.

## 5.1  Predicting cumulative  outcomes

Temporal-difference methods are  not  limited to  predicting only  tile final outcome of  a sequence; they  can also  be used  to  predict a quantity that  accumulates over  a sequence.  That  is, each  step  of  a sequence  may incur  a cost, where we wish to  predict tile expected  total cost  over  the  sequence.  A  common  way for  this to  arise is  ibr  tile costs to  be elapsed time.  For  example,  in  a bounded random walk one might want to  predict how  many  steps  will  be taken  before termination. In a poh,-balancing problem  one may  want to  predict time  until a failure in  balancing, and in a packet-switched telecomnnmications network one may want to  pre(tict he  total delay  in  sending  a packet.  In  game playing, points  may  be lost  or  won  throughout  a game, and we  may  be interested in predicting the  expected  net  gain  or  loss. In  all of  these examples,  the  quantity predi('ted is  the  cumulative sum of  a number of  parts, where tile parts become known  as the  sequence  evolves.  For convenience, we will  continue  to  refer to these  parts  as costs, even though their minimization will  not be a goal  in  all applications.

hl such problems,  it. is  nat, ural  to use the observation vector  received  at each  step  to  predict the  total cumulative  cost after that  step, rather than  the total cost  for  the  sequence  as  a whole. Thus, we will  want Pt to  predict the remaining cumulative  cost  given  the  t th observation rather  than the  overall cost  for  the  sequence.  Since  the  cost  for  the  preceding portion of  the  sequence is  already known, the  total sequence  cost can  always  be  estimated as  the  sum of the  known  cost-so-far and the  estimated cost-remaining (of. the  A*  algorithm. dynamic programming).

The procedures  presented earlier are  easily generalized to  include  the  case of  predicting cumulative  outconws. Let ct+l denote  the  actual cost  incurred t)etween  times  t  and t  +  1,  and let Cij  denote  the  expected  value  of  the  cost incurred  on transition from state  i to state j.  We  would like Pt to equal 1/7. the  expected  value  of  zt = ~k=t Ck+l.. where m  is  the  number of  observation

vectors in  the  sequence.  The prediction error can be represented in  terms  of temporal  differences as zt Pt rn --  Pt rn -= Ek=~(ck+~ + Pk+~ - P~), Ek=t ek  +  l where we define Pm+l = 0.  Then,  following the  same steps used  to  derive the TD(A) fmnily  of  procedures defined by (4), one can also  derive the cumulative TD(A) family  defined by

<!-- formula-not-decoded -->

The three theorems  presented earlier in  this article carry  over  to  the  cumulative outcome case  with the  obvious  modifications. For example,  the  ideal prediction for  each  state i  E N  is  the  expected value  of  the  cumulative sum of the  costs:

<!-- formula-not-decoded -->

If  we  let  h be the  vector with  components [h]~  =  ~3 P~J6iJ' i  E  N, then  (5) holds  for  this case  as  well. Following  steps  similar to those  in  the  proof  of Theorem 2,  one can show that, using  linear cumulative  TD(0),  the  expected value  of  the  weight  vector after n sequences  have  been  experienced is

<!-- formula-not-decoded -->

after which the  rest of  the  proof of  Theorem 2 follows unchanged.

## 5.2  Intra-sequence weight  updating

So far  we have  concentrated on TD  procedures in  which the  weight  vector is  updated  after the  presentation of  a  complete  sequence  or  training set. Since each  observation of  a  sequence  generates an increment to  the  weight  vector, in many respects it would be  simpler to  update  the  weight  vector immediately after each  observation. In  fact, all previously studied TD  methods  have  operated in  this more fully incremental way.

Extending  TD(A) to  allow  for  intra-sequence uI)dating requires a bit  of  care. The obvious  extension is

<!-- formula-not-decoded -->

However. if  w  is  changed within  a sequence,  then  tile temporal  changes in prediction during  the sequence, as defined  by this  procedure, will  be due to changes in w  as well  as to changes in x.  This is  probably  an undesirable feature; in  extreme  cases  it may even  lead  to  instability. The folh)wing uI)date rule  ensures that  only  changes  in  prediction due to  x are  effective in  causing weight  alterations:

<!-- formula-not-decoded -->

This refinement  is  used in Samuel's  (1959)  checker  player  and in Sutton's (1984)  Adaptive  Heuristic Critic, but  not  in  Holland's (1986)  bucket  brigade or  in  the  system  described by Barto  et  al.  (1983).

## 5.3 Prediction by  a fixed interval

Finally, consider tile problem of  making a prediction for  a particular fixed amount of  time  later. For example,  suppose  you are  interested in  predicting one  week in  advance  whether  or  not  it will rain  -  on each  Monday, you predict whether  it  will  rain  on the  following Monday, oi1  each Tuesday,  you predict whether  it  will  rain  on the  following Tuesday,  and so  on for  each  day of  tile week. Although  this problem involves a sequence  of  predictions, TD  methods cannot  be directly applied because  each  prediction is  of  a different event  and thus  there is  no clear desired relationship between  them.

In order  to  apply  TD  methods,  this problem nmst be embedded within  a larger family  of  prediction problems. At each  (lay  t,  we  must form not  only I) 7,  our  estimate of  the  probability of  rain  seven  days  later, but  also  Py,  P~, .... P~, where each  P~  is  an estimate of  the  probability of  rain  ~ days  later. This  will  provide  for  overlapping sequences  of  inter-related predictions, e.g., P7,  Pt6+l, Pts+2 ..... Pt~6, all of  the  same event., in  lifts case  of  whether  it  will rain  on (lay t  +  7. If  the  predictions are  accurate, we will  have P( =  Pt~-i l, Vt,  1  &lt; (~  &lt;  7,  where Pt ° is  defined as  tile actual outcome at  time  t  (e.g., 1 if it rains. 0 if it  does  not  rain). The update  rule  h)r  the  weight  vector 'w ° used  to comt)ute  Pt # would be

<!-- formula-not-decoded -->

As illustrated here.  there are  three key steps  ill constructing a TD  method for  a particular problem. First, embed the problen~  of  interest in a larger

('lass of  t)rohlems, if  necessary, in  order  to  produce  an appropriate sequence of  predictions. Second,  write  down recursive equations expressing the  desired relationship t)etween predictions at  different  ilnes in  the  sequence.  For the simplest cases, with  which this  article has been mostly  concerned, these  are just Pt --Pt+l, whereas  in  the  cumulative outcome  case  these  are Pt =  Pt+l  + ct+l. Third,  construct an update rule that  uses  the  mismatch  in  the  reeursive equations to  drive weight  changes  towards  a better match, These three steps arc  very  similar to  those taken  in  formulating a dynamic  programming  problem (e.g., Denardo,  1982).

## 6.  Related research

Although  temporal-difference methods  have  never  previously been  identified or  studied on their own, we can  view  some previous machine  learning research as having  used  them. In  this  section we briefly review  some of  this  previous work in  light, of  the  ideas  developed  here.

## 6.1  Samuel's checker-playing  program

The earliest known use  of  a TD  method was in  Samuel's  (1959)  celebrated checker-playing program. This  was in  his  "learning by generalization" procedure,  which modified  the  parameters of  the  function used to  evaluate board positions. The evaluation of  a position was thought  of  as  an estimate or  prediction of  how tile game would eventually turn  out  starting from  ttlat position. Thus,  the  sequence  of  positions from an actual game or  an anticipated continuation  naturally gave  rise to  a sequence  of  predictions, each  about  the  game's fizlal outcome.

In Samuel's  learning procedure, the  difference between the  evaluations of each  pair of  successive positions occurring in  a  game was used  as  an  error; that is, it  was used  to  alter the  prediction associated with  the  first position of  the pair  to  be  more like the  prediction associated with  the  second.  The predictions for  the  two positions were computed in different ways.  In most versions of the program, the prediction for  the first position was simply the result of applying  the  current evaluation flmction to  that  position. The prediction for tile second  position was the "backed-up"  or  minimax score  from a lookahead search  started at  that  position, using  the  current evaluation function. Samuel referred to  the  difference between  these  two predictions as delta. Although  his updating  procedure  was nmch  more complicated than  TD(0),  his  intent was to  use  delta rrmch as P,+~.  -  Pt is  used  in  (linear) TD(O).

However, Samuel's  learning procedure significantly differed from all the  TD methods  discussed here  in  its treatment of  the  final step  of  a  sequence. We  have considered each  sequence  to  end with  a definite, externally-supplied outcome (e.g.. 1 for  a victory and 0 for  a defeat). The prediction for  the  last position in a sequence  was  altered so as to match this  final outcome.  In Samuel's proced~re, on the  other  hand,  no position had a definite a priori evaluation, and the evaluation for  the last  position in a sequence  was  never  explicitly altered. Thus,  while  both  procedures constrained the  evaluations (predictions) of  nonterminal positions to  match those  that  follow  them,  Samuel's  provided

no additional constraint on the  evaluation of  terminal  positions. As he himself pointed out,  many useless evaluation functions satisfy just tile first constraint (e.g., any flmction that  is  constant for  all positions).

To discourage his  learning procedure  from finding useless evaluation flmctions, Samuel included in  the  evaluation flmction a non-modifiable term  measuring how many  more pieces  Ills program had than  its opponent.  However. although  this  modification may  have decreased the  likelihood of  finding useless  evaluation flmctions, it  (lid not prohibit them. For example,  a constant function could  still have been attained by setting the  modifiable terms  so as to  cancel the  effect of  tile non-modifiable one.

If Samuel's  learning procedure was not  constrained to  find  useflfl evaluation functions, then  it  should  have been possible fbr it to  become worse with  experience.  In fact, Samuel reported observing  this  dm'ing  extensive sell-play trahfing sessions. He  found that  a good way  to  get  tile program improving again  was to set  the  weight  with  the  largest ab.solute value back to zero. ttis interpretation was that  this drastic intervention jarred tile program out  of  local optima,  but  another possibility is  that  it jarred the  program out  of  evaluation flmctions that  changed little, but that  also  had little to  do with  winning  or losing the  game.

Nevertheless, SamueFs  learning procedure  was  overall very  successful; it played  an imt)ortant role  in  significantly improving  the  play  of  his  checkerplaying program  until it  rivaled human  checker masters.  Christensen and Korf have investigated a simplification of  Samuel's  procedure  that  also  does not constrain the evaluations of terminal  positions, and imve obtained  promising  prelinfinary results (Christensen, 1986:  Christensen &amp;  Korf.  1986).  Thus. although  a terminal  constraint may  be critical to good tenlporal-(til[i~rence theory, apparently it  is  not  strictly necessary to  obtait~ goo(t  performance.

## 6.2 Baekpropagation  in eonneetionist networks

Tim baekprotmgation technique of  Rumelhart  et  al. (1985)  is one  of  the  most exciting recent developments  in  incremental learning methods.  This  technique extends the  Widrow-Hoff  rule so  that  it can  be  apt)lied to  the  interior "'hidden" units of  multi-layer cmmectionist networks.  In  a  backpropagation network, the inlmt-output functions of  all units  at'(, deternfinistic and differentiable. As a res,flt, the  t)artial derivatives of  Ill(, error measure  wilh  respect to  each  connection  weight  are  well-defined, and one can at)ply a gradient-descent at)preach such as that  used in the  original Widrow-ttoff rule.  The term "backt)ropagati(m"  reDrs  to  th(' way the  partial derivatives are  efficiently computed in  a backward  propagating sweep lhrough  the  network.  As presented by  Rume/hart et  al., t)ackI)ropagation is  explicitly a sut)ervised-h~arning t)rocedure.

The purt)ose of  bolh backprot)agation and TD  methods is  accurate credit assignment.  Backprot)agalion de('ldes which part(s) of  a network 1o change so as to  influence the  network's output  and thus  to  reduce  its  overall err()r. whereas  TD  methods  decide how each  ou/tmt  of  a  temI)oral sequence  of  outputs should  be  changed.  Backpropagation addresses a ,slr~u:tural ('re(iil-assignment issue whereas  TD  metho(ts ad(lr(,ss a temporal cr(,dit-asslgnuient issue.

Although  it  currently seems that  backpropagation and TD  methods  address different parts  of  the  credit-assignment problem,  it  is  important to  note  that they  are  perfectly compatible and easily combined.  In  this article, we have  emphasized the  linear case. but  the  TD  methods  presented are  equally applicable to  predictions formed by nonlinear functions, such as backpropagation-stylc networks. The key requirement  is  that  the gradient V.~Pt be computable. In a linear system,  this  is  just xt. In a network of  differentiable nonlinear elements, it can  be  computed  by a  backpropagation process. For  example,  Anderson  (1986,  1987) has implemented  such  a combination of  backpropagation and a temporal-difference method (the  Adaptive  Heuristic Critie~ see  below), successfully applying it to  both  a  nonlinear broomstick-balancing task  and the Towers of  Hanoi problem.

## 6.3  Holland's  bucket brigade

Holland's  (1986)  bucket  brigade  is  a technique for  learning sequences  of rule  invocations in a kind of  adaptive  production system called a classifier .system. The production rules in  a classifier system  compete  to  become active and have  their right-hand sides (called messages) posted  to  a  working-memory data  structure (called the message list). Conflict resolution is  carried out  by a competitiw~  auction.  Each rule  that  matches the current contents of  the message list makes a bid that  depends on the  product  of  its  specificity and its ,strength, a modifiable numerical  parameter.  The highest bidders  become active and post  their messages  to  a new message  list for  the  next  round  of  the auction.

The bucket  brigade is  the  process that  adjusts the  strengths of  the  rules and thereby  determines which rules will  become active at  which times. When  a rule  becomes active, it  loses strength by the  amount of  its bid,  but  also  gains strength if  the  message it  posts  triggers other  rules  to become active in  the next  round  of  tile auction. The strength gained  is  exactly the  bids  of  the  other rules. If several rules post  the  same message,  then  the  bids of  all responders are pooled  and divided equally among the  posting rules. In  principle, hmg  chains of  rule  invocations can  be  learned in  this way,  with  strength being  passed  back from rule  to  rule, thus  tile name  %ueket brigade." For a chain  to  be stable, its final rule  must affect the  environment, achieve  a goal,  and thereby receive new strength in  the  form  of  a payoff from the  external environment.

Temporal-difference methods and the  bucket  brigade both  borrow  the  same key idea  from Sanmel's  work that  the  steps in  a sequence  should  be evaluated  and adjusted according to  their immediate  or  near-immediate successors, rather than  according to  tile final outcome.  The similarity between  TD  methods  and the  bucket  brigade can  be seen  at  a  more detailed level by considering tile latter's effect on a  linear chain  of  rule invocations. Each rule's  t.rength can be viewed as  a prediction of  the  payoff that  will  ultimately be obtained fl'om the  environment. Assuming equal  specificities, the  strength of  each  rule  experiences a  net  change  dependent on  the  difference between  that  strength and the strength of  the  succeeding rule. Thus,  like TD(0),  the  bucket  brigade  updates each  strength (prediction) in  a sequence  of  strengths (predictions) according to  the  immediately following temporal  difference in  strength (prediction).

There are also  mlmerous differences between the bucket  brigade  and the TD  methods presented here.  The most important  of  these  is  that  the  bucket brigade  assigns  credit based on the  rules  that cau,sed other  rules  to become active, whereas  TD  methods assign credit based  solely on temporal  succession. The bucket  brigade thus  performs  both  temporal artd structural credit assignment in a single  tnechanism. This contrasts with the TD/backpropagation combination  discussed in  the  preceding subsection, which uses  separate mechanisms for  each kind of  credit assignment.  The relative advantages  of  these two approaches  are  still to  be deternfined.

## 6.4  Infinite discounted  predictions and the  Adaptive Heuristic Critic

All the prediction problems we  have considered  so far  have had definite outcomes  That is, after some point  in  time  tile actual outcome corresponding to  each prediction became known.  Supervised-learning methods require  this property, because  they  make no learning changes  until the  actual outcome is discovered, but in some problems  it  never  becomes completely  known.  For example,  suppose  one wishes  to  predict the  total return  from investing in  the stock  of  various companies:  unless a company goes  out  of  business, total return is  never  flflly determined.

Actually, there  is  a t)roblem of  definition here:  if a company never  goes  out of  business and earns  income every  year, the  total return  can be infinite. For reasons  of  this  sort, infinite-horizon prediction problems  usually include  some form of  discounting. For  example,  if some process generates costs ct+l at  each transition from t  to  t  +  1,  we may  want Pt to  predict the  discounted sum:

<!-- formula-not-decoded -->

where the di,scount-rate parameter y, 0  \_&lt; "7 &lt;  1, determines  the extent  to which we are  concerned  with  short-range or  long-range prediction.

If Pt should  equal  the  above  zt, then  what are  the  reeursivc equations defining  the  desired relationship between temporally successive predictions? If  the predictions are  accurate, we can write

<!-- formula-not-decoded -->

The  mismatch or TD  error  is  the difference between the two sides  of this equation, (ct+i + q'Pt+l) - Pt,. 9 Sut, ton's (1984)  Adaptive  Heuristic Critic uses

9 ' Wltten (1977) first proposed updating  predictions of  a discounted  sum  based on a discrepancy  of  this  sort.

this error in  a  learning rule otherwise identical to  TD()0's:

<!-- formula-not-decoded -->

where Pt is  the  linear form wTxt, SO that VwPt  =  mr. Thus,  the  Adaptive Heuristic Critic is  probably best  understood as  using  the  linear TD  method for  predicting discounted cumulative outcomes.

## 7.  Conclusion

These  analyses and  experiments suggest that TD  methods  may be  the  learning  methods  of choice  for many real-world learning problems.  We  have  argued that  many of  these problems  involve temporal  sequences of  observations and predictions. Whereas conventional, supervised-learning approaches disregard this temporal  structure, TD  methods are  specially tailored to  it.  As a result, they  can  be computed  more incrementally and require significantly less memory  and peak computation.  One  TD  method makes exactly the  same predictions and learning changes  as  a supervised-learning method,  while  retaining these computational advantages. Another  TD  method  makes different learning changes, but  has  been  proved  to  converge  asymptotically to  the  same correct predictions. Empirically, TD  methods appear  to  learn  faster than supervised-learning methods,  and one TD  method has  been  proved  to  make optimal predictions for finite training sets that are  presented repeatedly. Overall, TD  methods  appear to  be  computationally cheaper and to  learn faster than conventional approaches to  prediction learning.

The progress made in  this paper  has  been due primarily to  treating TD methods as  general methods for  learning to  predict rather than  as  specialized  methods  fi)r learning evaluation flmctions, as  they  were in  all previous work.  This  simplification makes their theory  much  easier and also  greatly broadens  their range  of  applicability. It  is  now clear that TD  methods  can  be used  for any  pattern recognition problem  in  which  data are  gathered over  thne for  example,  speech  recognition, process  monitoring, target, identification, and market-trend prediction. Potentially, all of  these  can  benefit, from the advantages of  TD  methods  vis-a-vis supervised-learning methods.  In  speech recognition, for example, current learning methods  cannot be  applied until the correct classification of  the  word is known. This  means that all critical information  about  the  waveform  and  how it was processed must  be  stored for  later credit assignment. If learning proceeded simultaneously with  processing, as  in TD  methods,  this storage would be avoided, making it  practical to  consider far more features aim combinations of  features.

As general prediction-learning methods,  temporal-difference methods can also be  applied to  t.tte classic probteln of  learning an  internal model  of the  world. Much  of  what  we mean by having  such  a model is  the  ability to  predict the flm~re  based  on current actions and observations. This  prediction problem  is a  multi-step one,  aim  the  external world  is well  modeled  as  a  causal dynamical system;  hence  TD  methods  should  be applicable and advantageous. Sutton

and Pinette (1985)  and Sutton  and Barto  (1981b)  have  begun to  pursue  one approach  along  these  lines, using  TD  methods and recurrent connectionist networks.

Animals  must also face the  problem  of  learning internal models  of  the  world. The learning process that  seems to  perform  this function in  animals  is  called Pavlovian or classical conditioning. For example,  if a (:tog is  repeatedly presented  with  the  sound  of  a bell and then  fed. it will  learn to  predict the  meal given  just the  bell, as  evidenced  by salivation to  the  bell alone.  Some  of  the detailed features of  this learning process suggest that  animals  may  be  using  a TD  method (Kehoe,  S('hreurs. &amp; (h'aham,  1987:  Sutton  &amp;  Barto,  1987).

## Acknowledgements

The author  acknowledges  especially the  assistance of  Andy  Barto,  Martha Steenstrup, Chuck  Anderson, John Moore, and Harry Klopf. I also  thank ()liver Selfridge, Pat  Langley, Ron Rivest, Mike Grimaldi, John  Aspinall, Gene Cooperman, Bud Frawley, Jonathan  Bachrach,  Mike Seymour,  Steve  Epstein, Jim Kehoe, Les Servi, Ron Williams,  and Marie Goslin.  The early stages of this research were  supported by AFOSI2  contract F33615-83-C-1078.

## References

- Ackley,  D. H.,  Hintom G. HI., &amp;  Sejnowski, T. J.  (1985).  A  learning algorithm for  Boltzmann machines. Cognitive  S'cience, 9. 147 169.
- Anderson,  C. W.  (1986). Learning  and problem solving  with  multilayer connectionist sy,stcrn,s. Doctoral  dissertation, Department of  Computer and Information Science, University of  Massaclmsetts, Amherst.
- Anderson,  C.  W. (1987).  Strategy learning with  multilayer conneetionist representations. Proceedings  of  the  Fourth  International Work,shop  on Machine Learning (pp.  103 114).  Irvine. CA: Morgan Kauflnann.
- Barto,  A. G.  (1985). Learning  by statistical cooperation of  self-interested neuron-like computing  elements. Ituman Neurobiology, d,  229 256.
- Barto,  A. (;., Sutton  IL S.,  &amp;  Anderson.  (:. W.  (1983).  Neuronlike elements that  call solve difficult learning control problems. IEEE  Transactions on System.s, Man,  and Cybernetic.s. l&amp; 834 846.
- Booker.  L. B. (19~2). Intelligettt behavior  a.~  an adaptation  to the  task  et~t~'ronrnent. Doctoral dissertation. Department of  Computer and Colmmlnication Sciences, University of  Michigan.  Aim  Arbor.
- Christensen, J.  (1986).  Learning  static evaluation fimctions l)y  linear regression.  In  T.  M. Mitchell. J.  G.  Carbonell. &amp; R.  S.  Michalski (Eds.), Machine learninfl: A  guide  to  current  re.search. Boston:  Kluwor Academic.
- Christensen. ,l., &amp; Korf,  R.  E. (1986).  A unified theory  of  hem'istic evaluathm flmetions and its application to  learning. Proceedings  of  the  Fifth National Conference on Artificial Intelligence (Pt)1.18  152).  Philadelphia. PA: Morgan  Kaufmann.
- Denardo.  E.  V. (1982). Dynamic  programmin.g:  Model.s  and applicatio~.~'. Engh'wood (?lifts. N,]: Prentice-Hall.

- Dietterich, T. G.,  &amp;  Michalski, R. S.  (1986).  Learning to  predict, sequences. In  R. S.  Michalski, J.  G. Carbonell, &amp;  T. M.  Mitchell (Eds.), Machine learning:  An  artificial intelligence approach (Vol. 2). Los Altos,  CA: Morgan Kaufmann.
- Gelperin, A.,  Hopfield, J.  J., Tank,  D.  W.  (1985).  The logic of Limaz learning. In  A. Selverston (Ed.), Model neural  networks  and behavior. New  York: Plenum Press.
- Hampson, S.  E. (1983). A  neural  model of  adaptive behavior. Doctoral dissertation, Department  of  Information and Computer Science, University of California, Irvine.
- Hampson, S.  E.,  &amp; Volper, D. J.  (1987).  Disjunctive models  of  boolean  category  learning. Biological Cybernetics, 56, 121 137.
- Holland,  J.  H.  (1986).  Escaping  brittleness: The possibilities of generalpurpose  learning algorithms applied to  parallel rule-based systems.  In R. S.  Michalski, J.  G. Carbonell, &amp; T.  M. Mitchell (Eds.), Machine learning:  An artificial intelligence approach (Vol. 2).  Los Altos, CA: Morgan Kaufmann.
- Kehoe.  E.  J., Schreurs~ B. G.,  &amp;  Graham. P.  (1987). Temporal  primacy  overrides prior training in  serial compound conditioning of  the  rabbit's nictitating membrane response. Animal Learning  and Behavior,  15, 455-464.
- Kemeny,  J.  G., &amp;  Snell, J,  L. (1976). Finite  Markov  chain,s. New  York: Springer-Verlag.
- Klop[,  A. H. (1987). A  neuronal  model of  classical conditioning (Technical Report  87-1139). OH:  Wright-Patterson Air  Force  Base,  Wright  Aeronautical Laboratories.
- Moore,  J.  W., Desmond, J.  E.,  Berthier, N. E.,  Blazis, D. E.  J., Sutton, R. S., &amp;  Barto,  A. G. (1986). Simulation of  the  classically conditioned nictitating membrane response by a neuron-like adaptive element:  Response topography, neuronal firing and interstinmlus intervals. Behavioral Brain Research.  21, 143-154.
- Rumelhart,  D. E.,  Hinton,  G. E.,  &amp;  Williams, R. J.  (1985). Learning  internal  representations by error  propagation (Technical Report  No. 8506).  La Jolla: University of  California, San  Diego,  Institute for Cognitive Science. Also  in  D. E.  Rumelhart  &amp;  J.  L.  MeClelland  (Eds.), Parallel dietributed proees~ing: Exploratione in  the  microstrueture of  cognition (Vot. 1).  Cambridge, MA:  MIT  Press.
- Samuel, A. L. (1959). Some  studies in  machine learning using  the  game of  checkers. IBM  Journal  on Research and Development, 3, 210-229. Reprinted  in  E. A. Feigenbaum &amp;  J.  Feldman (Eds.), Computers and thought. New  York: McGraw-Hill.
- Sutton, R. S. (1984). Temporal credit assignment  in  reinforcement learning. Doctoral dissertation, Department  of  Computer  and  Information Science, University of  Massachusetts, Amherst.
- Sutton, R. S..  &amp;  Barto, A. G. (1981a).  Toward a modern theory of  adaptive networks: Expectation and  prediction. Psychological Review,  38, 135-171.

- Sutton, t2,. S.,  &amp;  Barto.  A. (;. (1981b).  An  adaptive network  that  constructs and uses  an internal model of  its environment. Cognition and Brai~ Th.eor.y, 4. 217 246.
- Sutton, 12,  S., &amp;  Barto, A. G. (1987).  A temporal-difference model of  classical conditioning. Proceeding,s of  the  Ninth  Anmtal  Co~@rettce  of  the  Cognitive Science  Soeiel 9 (pp.  355 378).  Seattle, WA:  Lawrence Erlbaum.
- Sutton.  R. S.,  &amp;  Pinette, B. (1985). The learning of  world  models by connectionist networks. Proeeedinqs of  the  Seventh  Annu(d Conference  of  the Cognitive ,qeience ,qociety (pp.  54 64).  Irvine, CA: Lawrence  Erlbaum.
- Varga,  12. S.  (1962). Matrix  iterative analysi,s. Englewood Clifl~. N J:  PrenticeHall.
- Widrow B..  &amp; Hoff. M. E.  (1960).  Adaptive  switching circuits. 1960 WES(;'ON Convention  Record,  Part  IV (pp.  96  104).
- Widrow, B.,  &amp;  Stearns, S.  D. (1985). Adaptive  signal processing. Englewood (:lifts, N  J:  Prentice-Hall.
- Williams,  R..l. (1986). Reinfi)rcement learning in  conneetionist network,s.: A mathematical  anal~,sis (Technical Report  No. 8605).  La Jolla: University of  California. San Diego,  Institute for  Cognitive Scien('e.
- Witten.  I. H. (1977).  An adaptive optimal  controller for  discrete-time Markov environments. Infl)rmation and Control, 3/j, 286 295.

## Appendix:  Accessory Theorems

Theorem A.1  Iflimn-~  A  n = 0, then  I - A   has  an  inverse, and (I-A)  -1 = E~O  Ai  ·

PROOF: See  Kemeny  and Snell  (1976,  p.  22).

Theorem A.2 For any matrix  A  with  linearly independent columna,  ATA  ia nonsingular.

PROOF: If ATA were singular, then  there  would exist a vector  y ¢  0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which  would imply  that Ay = 0,  contradicting the  assumptions that  y ¢  0 and that  A  has  linearly independent cohmms.

Theorem  A.3 is  positive definite. A  square  matrix A  is  positive definite if and only  if A + A  T

P ROOF  :

<!-- formula-not-decoded -->

But tile second  term  is  0,  because yT  ( A--  AT)y = (yT  ( g- AT)y)  T = yT  (AT-A)y =  --yT(A  - AT)y, and the  only  mmlber that  equals  its own inverse is  0. Therefore,

<!-- formula-not-decoded -->

implying  that yTAy and yT(A + AT)y always  have the  same sign,  and tMs either A  and A + A  T are  both  positive definite, or  neither is  positive definite.

## Erratum

An  error  appeared  on page 20 of  Richard S. Sutton's paper, "Learning  to Predict by the  Methods of  Temporal  Differences," which appeared  in  volume 3, number 1,  of Machine Learning. The correct version of  Figure  3 appears below,  along  with  its caption.

Figure 3.  Average  error on  the  random-walk  problem  under  repeated presentations. All  data are  from  TD(A) with  different values of  ·. The dependent  measure used  is  the  RMS error between  the  ideal predictions and those  found  by the  learning procedure  after being  repeatedly presented with  the  training set  until convergence of  the  weight  vector. This  measure  was averaged over 100 training sets  to  produce  the  data  shown.  The A =  1 data  point  is the  performance level attained by the  Widrow-Hoff  procedure.  For each data  point, the  standard error is  approximately cr  = 0.01, so  the  differences between  the  Widrow-Hoff  procedure and the  other  procedures are  highly significant.

<!-- image -->