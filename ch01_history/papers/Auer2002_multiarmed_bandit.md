<!-- image -->

©

## Finite-time Analysis of the Multiarmed Bandit Problem*

## PETER AUER

pauer@igi.tu-graz.ac.at

University of Technology Graz, A-8010 Graz, Austria

## NICOL ' O CESA-BIANCHI

cesa-bianchi@dti.unimi.it

DTI, University of Milan, via Bramante 65, I-26013 Crema, Italy

## PAUL FISCHER

fischer@ls2.informatik.uni-dortmund.de

Lehrstuhl Informatik II, Universit‹ at Dortmund, D-44221 Dortmund, Germany

Editor:

Jyrki Kivinen

Abstract. Reinforcement learning policies face the exploration versus exploitation dilemma, i.e. the search for a balance between exploring the environment to find profitable actions while taking the empirically best action as often as possible. A popular measure of a policy's success in addressing this dilemma is the regret, that is the loss due to the fact that the globally optimal policy is not followed all the times. One of the simplest examples of the exploration/exploitation dilemma is the multi-armed bandit problem. Lai and Robbins were the first ones to show that the regret for this problem has to grow at least logarithmically in the number of plays. Since then, policies which asymptotically achieve this regret have been devised by Lai and Robbins and many others. In this work we show that the optimal logarithmic regret is also achievable uniformly over time, with simple and efficient policies, and for all reward distributions with bounded support.

Keywords:

bandit problems, adaptive allocation rules, finite horizon regret

## 1. Introduction

The exploration versus exploitation dilemma can be described as the search for a balance between exploring the environment to find profitable actions while taking the empirically best action as often as possible. The simplest instance of this dilemma is perhaps the multi-armed bandit, a problem extensively studied in statistics (Berry &amp; Fristedt, 1985) that has also turned out to be fundamental in different areas of artificial intelligence, such as reinforcement learning (Sutton &amp; Barto, 1998) and evolutionary programming (Holland, 1992).

In its most basic formulation, a K -armed bandit problem is defined by random variables Xi , n for 1 ≤ i ≤ K and n ≥ 1, where each i is the index of a gambling machine (i.e., the 'arm' of a bandit). Successive plays of machine i yield rewards Xi , 1 , Xi , 2 , . . . which are

∗ Apreliminary version appeared in Proc. of 15th International Conference on Machine Learning, pages 100-108. Morgan Kaufmann, 1998

independent and identically distributed according to an unknown law with unknown expectation µ i . Independence also holds for rewards across machines; i.e., Xi , s and X j , t are independent (and usually not identically distributed) for each 1 ≤ i &lt; j ≤ K and each s , t ≥ 1.

A policy , or allocation strategy , A is an algorithm that chooses the next machine to play based on the sequence of past plays and obtained rewards. Let Ti ( n ) be the number of times machine i has been played by A during the fi rst n plays. Then the regret of A after n plays is de fi ned by

<!-- formula-not-decoded -->

and I E [ · ] denotes expectation. Thus the regret is the expected loss due to the fact that the policy does not always play the best machine.

In their classical paper, Lai and Robbins (1985) found, for speci fi c families of reward distributions (indexed by a single real parameter), policies satisfying

<!-- formula-not-decoded -->

where o ( 1 ) → 0 as n →∞ and

<!-- formula-not-decoded -->

is the Kullback-Leibler divergence between the reward density pj of any suboptimal machine j and the reward density p ∗ of the machine with highest reward expectation µ ∗ . Hence, under these policies the optimal machine is played exponentially more often than any other machine, at least asymptotically. Lai and Robbins also proved that this regret is the best possible. Namely, for any allocation strategy and for any suboptimal machine j , I E [ Tj ( n ) ] ≥ ( ln n )/ D ( pj ‖ p ∗ ) asymptotically, provided that the reward distributions satisfy some mild assumptions.

These policies work by associating a quantity called upper con fi dence index to each machine. The computation of this index is generally hard. In fact, it relies on the entire sequence of rewards obtained so far from a given machine. Once the index for each machine is computed, the policy uses it as an estimate for the corresponding reward expectation, picking for the next play the machine with the current highest index. More recently, Agrawal (1995) introduced a family of policies where the index can be expressed as simple function of the total reward obtained so far from the machine. These policies are thus much easier to compute than Lai and Robbins ' , yet their regret retains the optimal logarithmic behavior (though with a larger leading constant in some cases). 1

In this paper we strengthen previous results by showing policies that achieve logarithmic regret uniformly over time, rather than only asymptotically. Our policies are also simple to implement and computationally ef fi cient. In Theorem 1 we show that a simple variant of Agrawal ' s index-based policy has fi nite-time regret logarithmically bounded for arbitrary sets of reward distributions with bounded support (a regret with better constants is proven

Deterministic policy: UCB1.

Initialization: Play each machine once.

Loop:

- Play machine j that maximizes i; + V

2 Inn

, where aj is the average reward obtained from machine j, nj is the number of

number of plays done so far.

times machine j has been played so far, and n is the overall in Theorem 2 for a more complicated version of this policy). A similar result is shown in Theorem 3 for a variant of the well-known randomized ε -greedy heuristic. Finally, in Theorem 4 we show another index-based policy with logarithmically bounded fi nite-time regret for the natural case when the reward distributions are normally distributed with unknown means and variances.

Throughout the paper, and whenever the distributions of rewards for each machine are understood from the context, we de fi ne

<!-- formula-not-decoded -->

where, we recall, µ i is the reward expectation for machine i and µ ∗ is any maximal element in the set { µ 1 , . . . , µ K } .

## 2. Main results

Our fi rst result shows that there exists an allocation strategy, UCB1, achieving logarithmic regret uniformly over n and without any preliminary knowledge about the reward distributions (apart from the fact that their support is in [0 , 1]). The policy UCB1 (sketched in fi gure 1) is derived from the index-based policy of Agrawal (1995). The index of this policy is the sum of two terms. The fi rst term is simply the current average reward. The second term is related to the size (according to Chernoff-Hoeffding bounds, see Fact 1) of the one-sided con fi dence interval for the average reward within which the true expected reward falls with overwhelming probability.

Theorem 1. For all K &gt; 1 , if policy UCB1 is run on K machines having arbitrary reward distributions P 1 , . . . , PK with support in [0 , 1] , then its expected regret after any number n of plays is at most

<!-- formula-not-decoded -->

where µ 1 , . . . , µ K are the expected values of P 1 , . . . , PK.

Figure 1 . Sketch of the deterministic policy UCB1 (see Theorem 1).

Deterministic policy: UCB2.

Parameters: 0 &lt; a &lt; 1.

Initialization: Set rj = 0 for j = 1,..., K. Play each machine once.

Loop:

1. Select machine j maximizing xj + On,r;, where Ij is the average reward obtained from machine j, On,r; is defined in (3), and n is

the overall number of plays done so far.

2. Play machine j exactly (Tj + 1) - (Tj) times.

3. Set rj t rj + 1.

Figure 2 . Sketch of the deterministic policy UCB2 (see Theorem 2).

To prove Theorem 1 we show that, for any suboptimal machine j ,

<!-- formula-not-decoded -->

plus a small constant. The leading constant 8 //Delta1 2 i is worse than the corresponding constant 1 / D ( pj ‖ p ∗ ) in Lai and Robbins ' result (1). In fact, one can show that D ( pj ‖ p ∗ ) ≥ 2 /Delta1 2 j where the constant 2 is the best possible.

Using a slightly more complicated policy, which we call UCB2 (see fi gure 2), we can bring the main constant of (2) arbitrarily close to 1 /( 2 /Delta1 2 j ) . The policy UCB2 works as follows.

The plays are divided in epochs. In each new epoch a machine i is picked and then played τ( ri + 1 ) -τ( ri ) times, where τ is an exponential function and ri is the number of epochs played by that machine so far. The machine picked in each new epoch is the one maximizing ¯ xi + an , ri , where n is the current number of plays, ¯ xi is the current average reward for machine i , and

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

In the next result we state a bound on the regret of UCB2. The constant c α , here left unspeci fi ed, is de fi ned in (18) in the appendix, where the theorem is also proven.

Theorem 2. For all K &gt; 1 , if policy UCB2 is run with input 0 &lt; α &lt; 1 on K machines having arbitrary reward distributions P 1 , . . . , PK with support in [0 , 1] , then its expected regret after any number

<!-- formula-not-decoded -->

Randomized policy: En-GREEDY.

Parameters: c &gt; 0 and 0 &lt; d &lt; 1.

Initialization: Define the sequence En € (0,11, n = 1,2,..., by

Loop: For each n = 1, 2, ...

- Let in be the machine with the highest current average reward.

random arm.

of plays is at most

<!-- formula-not-decoded -->

where µ 1 , . . . , µ K are the expected values of P 1 , . . . , PK.

Remark . By choosing α small, the constant of the leading term in the sum (4) gets arbitrarily close to 1 /( 2 /Delta1 2 i ) ; however, c α →∞ as α → 0. The two terms in the sum can be traded-off by letting α = α n be slowly decreasing with the number n of plays.

A simple and well-known policy for the bandit problem is the so-called ε -greedy rule (see Sutton, &amp; Barto, 1998). This policy prescribes to play with probability 1 -ε the machine with the highest average reward, and with probability ε a randomly chosen machine. Clearly, the constant exploration probability ε causes a linear (rather than logarithmic) growth in the regret. The obvious fi x is to let ε go to zero with a certain rate, so that the exploration probability decreases as our estimates for the reward expectations become more accurate. It turns out that a rate of 1 / n , where n is, as usual, the index of the current play, allows to prove a logarithmic bound on the regret. The resulting policy, ε n -GREEDY, is shown in fi gure 3.

Theorem 3. For all K &gt; 1 and for all reward distributions P 1 , . . . , PK with support in [0 , 1] , if policy ε n -GREEDY is run with input parameter

<!-- formula-not-decoded -->

Figure 3 . Sketch of the randomized policy ε n -GREEDY (see Theorem 3).

Deterministic policy: UCB1-NORMAL.

Loop: For each n = 1,2,...

- If there is a machine which has been played less then [8 log n]

times then play this machine.

- Otherwise play machine j that maximizes

In(n - 1)

nj the number of times machine j has been played so far.

then the probability that after any number n ≥ cK / d of plays ε n -GREEDY chooses a suboptimal machine j is at most

<!-- formula-not-decoded -->

Remark . For c large enough (e.g. c &gt; 5) the above bound is of order c /( d 2 n ) + o ( 1 / n ) for n →∞ , as the second and third terms in the bound are O ( 1 / n 1 + ε ) for some ε &gt; 0 (recall that 0 &lt; d &lt; 1). Note also that this is a result stronger than those of Theorems 1 and 2, as it establishes a bound on the instantaneous regret. However, unlike Theorems 1 and 2, here we need to know a lower bound d on the difference between the reward expectations of the best and the second best machine.

Our last result concerns a special case, i.e. the bandit problem with normally distributed rewards. Surprisingly, we could not fi nd in the literature regret bounds (not even asymptotical) for the case when both the mean and the variance of the reward distributions are unknown. Here, we show that an index-based policy called UCB1-NORMAL, see fi gure 4, achieves logarithmic regret uniformly over n without knowing means and variances of the reward distributions. However, our proof is based on certain bounds on the tails of the χ 2 and the Student distribution that we could only verify numerically. These bounds are stated as Conjecture 1 and Conjecture 2 in the Appendix.

The choice of the index in UCB1-NORMAL is based, as for UCB1, on the size of the onesided con fi dence interval for the average reward within which the true expected reward falls with overwhelming probability. In the case of UCB1, the reward distribution was unknown, and we used Chernoff-Hoeffding bounds to compute the index. In this case we know that

Figure 4 . Sketch of the deterministic policy UCB1-NORMAL (see Theorem 4).

the distribution is normal, and for computing the index we use the sample variance as an estimate of the unknown variance.

Theorem 4. For all K &gt; 1 , if policy UCB1-NORMAL is run on K machines having normal reward distributions P 1 , . . . , PK , then its expected regret after any number n of plays is at most

<!-- formula-not-decoded -->

where µ 1 , . . . , µ K and σ 2 1 , . . . , σ 2 K are the means and variances of the distributions P 1 , . . . , PK.

As a fi nal remark for this section, note that Theorems 1 -3 also hold for rewards that are not independent across machines, i.e. Xi , s and X j , t might be dependent for any s , t , and i /negationslash= j . Furthermore, we also do not need that the rewards of a single arm are i.i.d., but only the weaker assumption that I E [ Xi , t | Xi , 1 , . . . , Xi , t -1 ] = µ i for all 1 ≤ t ≤ n .

## 3. Proofs

Recall that, for each 1 ≤ i ≤ K , I E [ Xi , n ] = µ i for all n ≥ 1 and µ ∗ = max1 ≤ i ≤ K µ i . Also, for any fi xed policy A , Ti ( n ) is the number of times machine i has been played by A in the fi rst n plays. Of course, we always have ∑ K i = 1 Ti ( n ) = n . We also de fi ne the r.v. ' s I 1 , I 2 , . . . , where I t denotes the machine played at time t .

For each 1 ≤ i ≤ K and n ≥ 1 de fi ne

<!-- formula-not-decoded -->

Given µ 1 , . . . , µ K , we call optimal the machine with the least index i such that µ i = µ ∗ . In what follows, we will always put a superscript ' ∗ ' to any quantity which refers to the optimal machine. For example we write T ∗ ( n ) and ¯ X ∗ n instead of Ti ( n ) and ¯ Xi , n , where i is the index of the optimal machine.

Some further notation: For any predicate /Pi1 we de fi ne { /Pi1( x ) } to be the indicator fuction of the event /Pi1( x ) ; i.e., { /Pi1( x ) } = 1 if /Pi1( x ) is true and { /Pi1( x ) } = 0 otherwise. Finally, Var[ X ] denotes the variance of the random variable X .

Note that the regret after n plays can be written as

<!-- formula-not-decoded -->

So we can bound the regret by simply bounding each I E [ Tj ( n ) ].

Wewill make use of the following standard exponential inequalities for bounded random variables (see, e.g., the appendix of Pollard, 1984).

Fact 1 (Chernoff-Hoeffding bound) . Let X 1 , . . . , Xn be random variables with common range [0 , 1] and such that IE [ Xt | X 1 , . . . , Xt -1 ] = µ . Let Sn = X 1 +··· + Xn. Then for all a ≥ 0

<!-- formula-not-decoded -->

Fact 2 (Bernstein inequality) . Let X 1 , . . . , Xn be random variables with range [0 , 1] and

<!-- formula-not-decoded -->

Let Sn = X 1 +··· + Xn. Then for all a ≥ 0

<!-- formula-not-decoded -->

Proof of Theorem 1: Let ct , s = √ ( 2 ln t )/ s . For any machine i , we upper bound Ti ( n ) on any sequence of plays. More precisely, for each t ≥ 1 we bound the indicator function of I t = i as follows. Let /lscript be an arbitrary positive integer.

<!-- formula-not-decoded -->

Now observe that ¯ X ∗ s + ct , s ≤ ¯ Xi , si + ct , si implies that at least one of the following must hold

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We bound the probability of events (7) and (8) using Fact 1 (Chernoff-Hoeffding bound)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For /lscript = /ceilingleft ( 8 ln n )//Delta1 2 i /ceilingright , (9) is false. In fact

<!-- formula-not-decoded -->

for si ≥ ( 8 ln n )//Delta1 2 i . So we get

<!-- formula-not-decoded -->

which concludes the proof.

Proof of Theorem 3: Recall that, for n ≥ cK / d 2 , ε n = cK /( d 2 n ) . Let

<!-- formula-not-decoded -->

The probability that machine j is chosen at time n is

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Nowthe analysis for both terms on the right-hand side is the same. Let T R j ( n ) be the number of plays in which machine j was chosen at random in the fi rst n plays. Then we have

<!-- formula-not-decoded -->

✷

and

<!-- formula-not-decoded -->

by Bernstein ' s inequality (2) we get

<!-- formula-not-decoded -->

Finally it remains to lower bound x 0 . For n ≥ n ′ = cK / d 2 , ε n = cK /( d 2 n ) and we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in the last line we dropped the conditioning because each machine is played at random independently of the previous choices of the policy. Since

<!-- formula-not-decoded -->

Thus, using (10) -(13) and the above lower bound on x 0 we obtain

<!-- formula-not-decoded -->

This concludes the proof.

## 4. Experiments

For practical purposes, the bound of Theorem 1 can be tuned more fi nely. We use

<!-- formula-not-decoded -->

as un upper con fi dence bound for the variance of machine j . As before, this means that machine j , which has been played s times during the fi rst t plays, has a variance that is at most the sample variance plus √ ( 2 ln t )/ s . We then replace the upper con fi dence bound √ 2 ln ( n )/ n j of policy UCB1 with

<!-- formula-not-decoded -->

(the factor 1 / 4 is an upper bound on the variance of a Bernoulli random variable). This variant, which we call UCB1-TUNED, performs substantially better than UCB1 in essentially all of our experiments. However, we are not able to prove a regret bound.

We compared the empirical behaviour policies UCB1-TUNED, UCB2, and ε n -GREEDY on Bernoulli reward distributions with different parameters shown in the table below.

|    |    1 |    2 | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
|----|------|------|------|------|------|------|------|------|------|------|
|  1 | 0.9  | 0.6  |      |      |      |      |      |      |      |      |
|  2 | 0.9  | 0.8  |      |      |      |      |      |      |      |      |
|  3 | 0.55 | 0.45 |      |      |      |      |      |      |      |      |
| 11 | 0.9  | 0.6  | 0.6  | 0.6  | 0.6  | 0.6  | 0.6  | 0.6  | 0.6  | 0.6  |
| 12 | 0.9  | 0.8  | 0.8  | 0.8  | 0.7  | 0.7  | 0.7  | 0.6  | 0.6  | 0.6  |
| 13 | 0.9  | 0.8  | 0.8  | 0.8  | 0.8  | 0.8  | 0.8  | 0.8  | 0.8  | 0.8  |
| 14 | 0.55 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 |

✷

100

90-

% best machine played

70

60-

50°

10°

- UCB2 a=0.1

UCB2 x=0.01

- UCB2 a=0.001

UCB2 a=0.0001

10'

10'

plays

10'

10*

10'

40-

35-

30

25-

120-

15-

10-

5

10°

- UCB2 a=0.01

• UCB2 a=0.1

— UCB2 a=0.001

UCB2 a=0.0001

10'

107

10€

10'

Rows 1 -3 de fi ne reward distributions for a 2-armed bandit problem, whereas rows 11 -14 de fi ne reward distributions for a 10-armed bandit problem. The entries in each row denote the reward expectations (i.e. the probabilities of getting a reward 1, as we work with Bernoulli distributions) for the machines indexed by the columns. Note that distributions 1 and 11 are ' easy ' (the reward of the optimal machine has low variance and the differences µ ∗ -µ i are all large), whereas distributions 3 and 14 are ' hard ' (the reward of the optimal machine has high variance and some of the differences µ ∗ -µ i are small).

We made experiments to test the different policies (or the same policy with different input parameters) on the seven distributions listed above. In each experiment we tracked two performance measures: (1) the percentage of plays of the optimal machine; (2) the actual regret, that is the difference between the reward of the optimal machine and the reward of the machine played. The plot for each experiment shows, on a semi-logarithmic scale, the behaviour of these quantities during 100 , 000 plays averaged over 100 different runs. We ran a fi rst round of experiments on distribution 2 to fi nd out good values for the parameters of the policies. If a parameter is chosen too small, then the regret grows linearly (exponentially in the semi-logarithmic plot); if a parameter is chosen too large then the regret grows logarithmically, but with a large leading constant (corresponding to a steep line in the semi-logarithmic plot).

Policy UCB2 is relatively insensitive to the choice of its parameter α , as long as it is kept relatively small (see fi gure 5). A fi xed value 0 . 001 has been used for all the remaining experiments. On other hand, the choice of c in policy ε n -GREEDY is dif fi cult as there is no value that works reasonably well for all the distributions that we considered. Therefore, we have roughly searched for the best value for each distribution. In the plots, we will also show the performance of ε n -GREEDY for values of c around this empirically best value. This shows that the performance degrades rapidly if this parameter is not appropriately tuned. Finally, in each experiment the parameter d of ε n -GREEDY was set to

<!-- formula-not-decoded -->

Figure 5 . Search for the best value of parameter α of policy UCB2.

<!-- image -->

100

100

90

90

% best machine played

% best machine played

80-

80

70

70-

60

50€

50%

40

40

10°

10°

UCB2 a=0.001

UCB2 a=0.001

→ UCB-tuned

- UCB-tuned

- &amp;-GREEDY c=0.05

- E-GREEDY c=0.05

E-GREEDY c=0.10

&amp;-GREEDY c=0.10

&amp;-GREEDY c=0.15

A &amp;-GREEDY c=0.15

10'

10'

102

plays plays

103

103

10*

\_ UCB-tuned

UCB2 a=0.001

.. UCB2 a=0.001

→ UCB-tuned

- E-GREEDY c=0.05

- E-GREEDY c=0.10

- &amp;-GREEDY c=0.10

- ¿-GREEDY c=0.15

E-GREEDY c=0.15

&amp;-GREEDY c=0.20|

-

10'

10'

10°

10'

10*

10°

10

## 4.1. Comparison between policies

10

plays plays

We can summarize the comparison of all the policies on the seven distributions as follows (see Figs. 6 -12).

- -An optimally tuned ε n -GREEDY performs almost always best. Signi fi cant exceptions are distributions 12 and 14: this is because ε n -GREEDY explores uniformly over all machines, thus the policy is hurt if there are several nonoptimal machines, especially when their reward expectations differ a lot. Furthermore, if ε n -GREEDY is not well tuned its performance degrades rapidly (except for distribution 13, on which ε n -GREEDY performs well a wide range of values of its parameter).
- -In most cases, UCB1-TUNED performs comparably to a well-tuned ε n -GREEDY. Furthermore, UCB1-TUNED is not very sensitive to the variance of the machines, that is why it performs similarly on distributions 2 and 3, and on distributions 13 and 14.
- -Policy UCB2 performs similarly to UCB1-TUNED, but always slightly worse.

Figure 6 . Comparison on distribution 1 (2 machines with parameters 0 . 9 , 0 . 6).

<!-- image -->

Figure 7 . Comparison on distribution 2 (2 machines with parameters 0 . 9 , 0 . 8).

<!-- image -->

102

140

250

120-

200-

100-

150-

100-

40-

50-

103

103

10*

10*

105

10'

100

100

100

90-

90

80

80

70

80

60

70

50

40-

60-

30-

50+

20-

20-

0

40-

1000

10°

10°

% best machine played

% best machine played

% best machine played

UCB2 a=0.001

UCB2 x=0.001

UCB2 a=0.001

- UCB-tuned

- UCB-tuned

- E-GREEDY c=0.05

→ UCB-tuned

- E-GREEDY c=0.10

E-GREEDY c=0.15

£-GREEDY c=0.15

— E-GREEDY c=0.20

£-GREEDY c=0.10

-A-

A E-GREEDY c=0.15

E-GREEDY c=0.20

- &amp;-GREEDY c=0.30

10

10'

10'

102

102

102

plays plays

plays

10₴

103

103

-0-0

10€

10*

10%

10'

300

250

1000

250-

200-

.. UCB2 a=0.001

• UCB-tuned

UCB2 a=0.001

UCB2 x=0.001

— UCB-tuned

— UCB-tuned

800-

200-

150 -

33150-

600-

400-

100 -

100 -

200-

50-

50-

0e

10°

10°

10°

• E-GREEDY c=0.15

• E-GREEDY c=0.10

- E-GREEDY c=0.15

&amp;-GREEDY c=0.20

- E-GREEDY c=0.05

— £-GREEDY c=0.10

&amp;-GREEDY c=0.30

* E-GREEDY c=0.15

&amp;-GREEDY c=0.20

10'

10'

10'

102

plays plays

10'

10*

10'

102

10'

10*

10'

plays

<!-- image -->

Figure 8 . Comparison on distribution 3 (2 machines with parameters 0 . 55 , 0 . 45).

Figure 9 . Comparison on distribution 11 (10 machines with parameters 0 . 9 , 0 . 6 , . . . , 0 . 6).

<!-- image -->

Figure 10 . Comparison on distribution 12 (10 machines with parameters 0 . 9 , 0 . 8 , 0 . 8 , 0 . 8 , 0 . 7 , 0 . 7 , 0 . 7 , 0 . 6 , 0 . 6 , 0 . 6).

<!-- image -->

10₴

10%

103

10*

10'

100

100

80

80

% best machine played

% best machine played

60

60

40

40

20

20

0

0

10°

10°

UCB2 0=0.001

UCB2 a=0.001

- UCB-tuned

+ UCB-tuned

- E-GREEDY c=0.05

→ &amp;-GREEDY c=0.20

£-GREEDY c=0.30

• E-GREEDY c=0.10

• E-GREEDY c=0.40

- E-GREEDY c=0.15

10'

10'

102

plays plays

10'

10€

10'

300

400

350-

250-

300-

200 L

250-

75200-

5150-

150-

100 -

100-

50-

10°

10'

102

10'

10*

10'

103

10*

10'

10°

10'

102

103

10*

10'

<!-- image -->

plays plays

Figure 11 . Comparison on distribution 13 (10 machines with parameters 0 . 9 , 0 . 8 , . . . , 0 . 8).

Figure 12 . Comparison on distribution 14 (10 machines with parameters 0 . 55 , 0 . 45 , . . . , 0 . 45).

<!-- image -->

## 5. Conclusions

Wehaveshownsimpleandef fi cient policies for the bandit problem that, on any set of reward distributions with known bounded support, exhibit uniform logarithmic regret. Our policies are deterministic and based on upper con fi dence bounds, with the exception of ε n -GREEDY, a randomized allocation rule that is a dynamic variant of the ε -greedy heuristic. Moreover, our policies are robust with respect to the introduction of moderate dependencies in the reward processes.

This work can be extended in many ways. A more general version of the bandit problem is obtained by removing the stationarity assumption on reward expectations (see Berry &amp; Fristedt, 1985; Gittins, 1989 for extensions of the basic bandit problem). For example, suppose that a stochastic reward process { Xi , s : s = 1 , 2 , . . . } is associated to each machine i = 1 , . . . , K . Here, playing machine i at time t yields a reward Xi , s and causes the current

UCB2 a=0.001

UCB2 a=0.001

• UCB-tuned

• UCB-tuned

- &amp;-GREEDY c=0.20

o E-GREEDY c=0.05

- &amp;-GREEDY c=0.10|

&amp;-GREEDY c=0.30

— 8-GREEDY c=0.15

\_ E-GREEDY c=0.40

state s of i to change to s + 1, whereas the states of other machines remain frozen. A wellstudied problem in this setup is the maximization of the total expected reward in a sequence of n plays. There are methods, like the Gittins allocation indices, that allow to fi nd the optimal machine to play at each time n by considering each reward process independently from the others (even though the globally optimal solution depends on all the processes). However, computation of the Gittins indices for the average (undiscounted) reward criterion used here requires preliminary knowledge about the reward processes (see, e.g., Ishikida &amp; Varaiya, 1994). To overcome this requirement, one can learn the Gittins indices, as proposed in Duff (1995) for the case of fi nite-state Markovian reward processes. However, there are no fi nite-time regret bounds shown for this solution. At the moment, we do not know whether our techniques could be extended to these more general bandit problems.

## Appendix A: Proof of Theorem 2

Note that

<!-- formula-not-decoded -->

for r ≥ 1. Assume that n ≥ 1 /( 2 /Delta1 2 j ) for all j and let ˜ r j be the largest integer such that

<!-- formula-not-decoded -->

Note that ˜ r j ≥ 1. We have

<!-- formula-not-decoded -->

Now consider the following chain of implications machine j fi nishes its r -th epoch

<!-- formula-not-decoded -->

where the last implication hold because at , r is increasing in t . Hence

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The assumption n ≥ 1 /( 2 /Delta1 2 j ) implies ln ( 2 en /Delta1 2 j ) ≥ 1. Therefore, for r &gt; ˜ r j , we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

We start by bounding the fi rst sum in (15). Using (17) and Fact 1 (Chernoff-Hoeffding bound) we get

<!-- formula-not-decoded -->

for α &lt; 1 / 10. Now let g ( x ) = ( x -1 )/( 1 + α) . By (14) we get g ( x ) ≤ τ( r -1 ) for τ( r -1 ) ≤ x ≤ τ( r ) and r ≥ 1. Hence

<!-- formula-not-decoded -->

where c = (/Delta1 j α) 2 &lt; 1. Further manipulation yields

<!-- formula-not-decoded -->

We continue by bounding the second sum in (15). Using once more Fact 1, we get

<!-- formula-not-decoded -->

Now, as ( 1 + α) x -1 ≤ τ( i ) ≤ ( 1 + α) x + 1 for i ≤ x ≤ i + 1, we can bound the series in the last formula above with an integral

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we set

<!-- formula-not-decoded -->

As 0 &lt; α, /Delta1 j &lt; 1, we have 0 &lt; λ&lt; 1 / 4. To upper bound the bracketed formula above, consider the function

<!-- formula-not-decoded -->

with derivatives

<!-- formula-not-decoded -->

In the interval ( 0 , 1 / 4 ) , F ′ is seen to have a zero at λ = 0 . 0108 . . . . As F ′′ (λ) &lt; 0 in the same interval, this is the unique maximum of F , and we fi nd F ( 0 . 0108 . . .) &lt; 11 / 10. So we have

<!-- formula-not-decoded -->

Piecing everything together, and using (14) to upper bound τ( ˜ r j ) , we fi nd that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

This concludes the proof.

✷

## Appendix B: Proof of Theorem 4

The proof goes very much along the same lines as the proof of Theorem 1. It is based on the two following conjectures which we only veri fi ed numerically.

Conjecture 1. Let X be a Student random variable with s degrees of freedom. Then, for all 0 ≤ a ≤ √ 2 ( s + 1 ) ,

<!-- formula-not-decoded -->

Conjecture 2. Let X be a χ 2 random variable with s degrees of freedom. Then

<!-- formula-not-decoded -->

We now proceed with the proof of Theorem 4. Let

<!-- formula-not-decoded -->

Fix a machine i and, for any s and t , set

<!-- formula-not-decoded -->

Let c ∗ t , s be the corresponding quantity for the optimal machine. To upper bound Ti ( n ) , we proceed exactly as in the fi rst part of the proof of Theorem 1 obtaining, for any positive integer /lscript ,

<!-- formula-not-decoded -->

The random variable ( ¯ Xi , si -µ i )/ √ ( Qi , si -si ¯ X 2 i , si )/( si ( si -1 )) has a Student distribution with si -1 degrees of freedom (see, e.g., Wilks, 1962, 8.4.3 page 211). Therefore, using Conjecture 1 with s = si -1 and a = 4 √ ln t , we get

<!-- formula-not-decoded -->

for all si ≥ 8 ln t . The probability of ¯ X ∗ s ≤ µ ∗ -c ∗ t , s is bounded analogously. Finally, since ( Qi , si -si ¯ X 2 i , si )/σ 2 i is χ 2 -distributed with si -1 degrees of freedom (see, e.g., Wilks, 1962,

8.4.1 page 208). Therefore, using Conjecture 2 with s = si -1 and a = 4 s , we get

<!-- formula-not-decoded -->

for

<!-- formula-not-decoded -->

Setting

<!-- formula-not-decoded -->

completes the proof of the theorem.

## Acknowledgments

The support from ESPRIT Working Group EP 27150, Neural and Computational Learning II (NeuroCOLT II), is gratefully acknowledged.

## Note

1. Similar extensions of Lai and Robbins ' results were also obtained by Yakowitz and Lowe (1991), and by Burnetas and Katehakis (1996).

## References

- Agrawal, R. (1995). Sample mean based index policies with O ( log n ) regret for the multi-armed bandit problem. Advances in Applied Probability , 27 , 1054 -1078.

Berry, D., &amp; Fristedt, B. (1985). Bandit problems . London: Chapman and Hall.

- Burnetas, A., &amp; Katehakis, M. (1996). Optimal adaptive policies for sequential allocation problems. Advances in Applied Mathematics , 17:2 , 122 -142.
- Duff, M. (1995). Q-learning for bandit problems. In Proceedings of the 12th International Conference on Machine Learning (pp. 209 -217).
- Gittins, J. (1989). Multi-armed bandit allocation indices , Wiley-Interscience series in Systems and Optimization. New York: John Wiley and Sons.

Holland, J. (1992). Adaptation in natural and arti fi cial systems . Cambridge: MIT Press/Bradford Books.

Ishikida, T., &amp; Varaiya, P. (1994). Multi-armed bandit problem revisited. Journal of Optimization Theory and Applications , 83:1 , 113 -154.

- Lai, T., &amp; Robbins, H. (1985). Asymptotically ef fi cient adaptive allocation rules. AdvancesinAppliedMathematics , 6 , 4 -22.

Pollard, D. (1984). Convergence of stochastic processes . Berlin: Springer.

✷

Sutton, R., &amp; Barto, A. (1998). Reinforcement learning, an introduction . Cambridge: MIT Press/Bradford Books.

Wilks, S. (1962). Matematical statistics . New York: John Wiley and Sons.

Yakowitz, S., &amp; Lowe, W. (1991). Nonparametric bandit methods. Annals of Operations Research , 28 , 297 -312.

Received September 29, 2000 Revised May 21, 2001 Accepted June 20, 2001

Final manuscript June 20, 2001