                                                      The Thirty-Second AAAI Conference
                                                       on Artificial Intelligence (AAAI-18)




                  A Combinatorial-Bandit Algorithm for the Online Joint
            Bid / Budget Optimization of Pay-per-Click Advertising Campaigns

                      Alessandro Nuara, Francesco Trovò, Nicola Gatti, Marcello Restelli
                                    Dipartimento di Elettronica, Informazione e Bioingegneria
                                                Politecnico di Milano, Milano, Italy
                           {alessandro.nuara, francesco1.trovo, nicola.gatti, marcello.restelli}@polimi.it


                            Abstract                                        is not truthful—i.e., the best bid for an advertiser may be
                                                                            different from the actual value per click—, and learning al-
  Pay-per-click advertising includes various formats (e.g.,
                                                                            gorithms are commonly used to learn the optimal bids. The
  search, contextual, and social) with a total investment of more
  than 140 billion USD per year. An advertising campaign                    Vicrey-Clarke-Groves mechanism (VCG) is instead used for
  is composed of some subcampaigns—each with a different                    contextual and social advertising (Varian and Harris 2014;
  ad—and a cumulative daily budget. The allocation of the ads               Gatti et al. 2015). This auction is truthful only in the unreal-
  is ruled exploiting auction mechanisms. In this paper, we pro-            istic case in which the daily budget is unlimited.1 Thus, each
  pose, for the ﬁrst time to the best of our knowledge, an al-              advertiser needs to ﬁnd the best bid and budget values in an
  gorithm for the online joint bid/budget optimization of pay-              online fashion, and this problem is currently open in the lit-
  per-click multi-channel advertising campaigns. We formulate               erature. In the present paper, we provide a novel algorithm,
  the optimization problem as a combinatorial bandit prob-                  based on combinatorial bandit techniques (Chen, Wang, and
  lem, in which we use Gaussian Processes to estimate stochas-              Yuan 2013), capable of automating this task.
  tic functions, Bayesian bandit techniques to address the ex-
  ploration/exploitation problem, and a dynamic programming                    Related works. Only a few works in the algorithmic eco-
  technique to solve a variation of the Multiple-Choice Knap-               nomic literature tackle the campaign advertising problem
  sack problem. We experimentally evaluate our algorithm both               by combining learning and optimization techniques. More
  in simulation—using a synthetic setting generated a Yahoo!                speciﬁcally, Zhang et al. (2012) study a scenario similar to
  dataset—and in a real-world application for two months.                   the one we analyze in this paper. The authors take into ac-
                                                                            count the problem of the joint bid/budget optimization of
                                                                            subcampaigns in an ofﬂine fashion. However, this algorithm
                        Introduction                                        suffers from some drawbacks. More precisely, the model
Online advertising has been given wide attention from the                   of each subcampaign requires a huge number of parame-
scientiﬁc world as well as from industry. in 2016 more than                 ters s.t. even achieving rough estimates requires a consid-
72 billion USD has been spent in search advertising (IAB                    erable amount of data, usually available only after the sys-
2016), which represents about 50% of the total market. The                  tem has been running for several months. Furthermore, some
development of automatic techniques is crucial both for the                 parameters (e.g., the position of the ad for each impression
publishers and the advertisers, and artiﬁcial intelligence can              and click) cannot be observed by an advertiser, not allow-
play a prominent role in this context. In this paper, we focus              ing the employment of the model in practice. Finally, the
on pay-per-click advertising—including different formats,                   optimization problem formulated therein is nonlinear and
e.g., search, contextual, and social—in which an advertiser                 ﬁnding a good-quality approximate solution requires long
pays only once a user has clicked her ad.                                   computational time. Thomaidou, Liakopoulos, and Vazir-
   An advertising campaign is characterized by a set                        giannis (2014) separate the optimization of the bid from that
of subcampaigns—each with a potentially different pair                      one of the budget, using a genetic algorithm to optimize the
ad/targeting—and by a cumulative daily budget. Remark-                      budget and subsequently applying some bidding strategies.
ably, a campaign may include subcampaigns on different                      Markakis and Telelis (2010) study the convergence of some
channels, e.g., Google, Bing, Facebook. In pay-per-click ad-                bidding strategies in a single-subcampaign scenario.
vertising, to get an ad impressed, the advertisers take part                   Online learning results are known only for the restricted
in an auction, specifying a bid and a daily budget for each                 cases with a single subcampaign and a budget constraint
subcampaign (Qin, Chen, and Liu 2015). The advertisers’                     over all the length of the campaign without temporal dead-
goal is to select these variables to maximize the expected                  lines.2 Ding et al. (2013) and Xia et al. (2015) work on a
revenue they get from the advertising campaign. The Gen-
eralized Second Price auction (GSP) is used for search ad-                       1
vertising (King, Atkins, and Schwarz 2007). This auction                         Notably, the value per click is unknown at the setup of the
                                                                            campaign, not allowing an advertiser to bid truthfully.
Copyright  c 2018, Association for the Advancement of Artiﬁcial               2
                                                                                 This last assumption rarely holds in real-world applications
Intelligence (www.aaai.org). All rights reserved.                           where, instead, results are expected by a given deadline.




                                                                     2379
ﬁnite number of bid values and exploit a multi-armed bandit                                 Problem Formulation
approach. Trovò et al. (2016) work on a continuous space of             We are given an advertising campaign C = {C1 , . . . , CN },
bids and show that assuring worst-case guarantees leads to               with N ∈ N, where Cj is the j-th subcampaign, a ﬁ-
the worsening of the average-case performance.                           nite time horizon of T ∈ N days, and a spending plan
   Less related works concern daily budget optimization (Xu              B = {y 1 , . . . , y T }, where y t ∈ R+ is the cumulative budget
et al. 2015; Italia et al. 2017), bidding strategies in display          one is willing to spend at day t ∈ {1, . . . , T }.3,4 While the
advertising (Wang, Zhang, and Yuan 2016; Weinan et al.                   proposed deﬁnition is general w.r.t. the channel we target, in
2016; Zhang, Yuan, and Wang 2014; Lee, Jalali, and Das-                  the speciﬁc case of search engines (where the subcampaigns
dan 2013) and video advertising (Geyik et al. 2016). Finally,            are commonly targeted to multiple keywords), we assume
some works deal with the attribution problem of the conver-              that each subcampaign has been set s.t. its keywords behave
sions in display advertising (Geyik, A-Saxena, and Dasdan                similarly. As a consequence, a single decision for each group
2014; Kireyev, Pauwels, and Gupta 2016).                                 of keywords is required. For day t ∈ {1, . . . , T } and for ev-
                                                                         ery subcampaign Cj , the advertiser needs to specify the bid
   Original contributions. We formulate the optimization                 xj,t ∈ [xj,t , xj,t ], where xj,t , xj,t ∈ R+ are the minimum
problem as a combinatorial-bandit problem (Chen, Wang,                   and the maximum bid we can set, respectively, and the bud-
and Yuan 2013), where the different arms are the bid/budget              get yj,t ∈ [y j,t , y j,t ], where y j,t , y j,t ∈ R+ are the minimum
pairs. In the standard multi-armed bandit problem, we are
given a set of options called arms, at each turn we choose               and the maximum budget we can set, respectively. The goal
a single arm, and we observe only its reward. Differently,               is, for every day t ∈ {1, . . . , T }, to ﬁnd the best values of
in a combinatorial-bandit problem we are given a set of su-              bids and budgets, maximizing the subcampaigns cumulative
perarms, i.e., an element of the power set of the arms—here              expected revenue. These values can be found by solving the
corresponding to a combination of bid/budget pairs for each              following optimization problem:
subcampaign—, whose elements satisfy some set of com-                                            N
                                                                                                 
binatorial constraints—in our case, knapsack-like. At each                           max               vj nj (xj,t , yj,t )              (1a)
                                                                                    xj,t ,yj,t
round, we simultaneously play of all the arms contained                                          j=1
in the selected superarm and, subsequently, observe their                                        N
rewards. We use Gaussian Process (GP) regression mod-                                            
                                                                                         s.t.          yj,t ≤ y t                        (1b)
els (Rasmussen and Williams 2006) to estimate, for each
                                                                                                 j=1
subcampaign, the expected daily number of clicks for each
bid/budget pair and the value per click. We estimate the                                         xj,t ≤ xj,t ≤ xj,t            ∀j        (1c)
value per click of a subcampaign separately from the number                                      y j,t ≤ yj,t ≤ y j,t          ∀j        (1d)
of clicks since, while the number of daily clicks and, thus, of
the observed samples is usually large allowing one to obtain             where nj (xj,t , yj,t ) is the expected number of clicks given
accurate estimates in short time, the acquisitions are usually           the bid xj,t and the budget yj,t for subcampaign Cj and vj
much more sporadic and the estimation of the value per click             is the value per click for the subcampaign Cj . Basically,
may require a longer time. As a result, at the very beginning            this is a special case of Multiple-Choice Knapsack prob-
of the learning phase our algorithm maximizes the number                 lem (Kellerer, Pferschy, and Pisinger 2004) in which the ob-
of clicks, whereas, subsequently, the objective function is              jective function (1a)—corresponding to the value provided
gradually tuned by more accurate estimates of the values                 by knapsack—is the weighted sum of the expected number
per click of each subcampaign. This rationale is the same                of clicks of all the subcampaigns, where the weights are the
one currently followed by human experts. We design two                   subcampaigns’ value per click. Constraint (1b) is a budget
Bayesian bandit techniques to balance exploration and ex-                constraint, forcing one not to spend more than the budget
ploitation in the learning process, that return samples of the           limit, while constraints (1c) and (1d) deﬁne the ranges of
stochastic variables estimated by the GPs. Finally, we dis-              the variables. Similarly to the knapsack problem, here we
cretize the bid/budget space, and we formulate the optimiza-             have items–i.e., the subcampaigns—, each of which is char-
tion problem as a special case of Multiple-Choice Knapsack               acterized by a value and requires a portion of the budget. The
problem (Sinha and Zoltners 1979) in which we use the sam-               differences are: the occupancy of an item is not a constant,
ples returned by the bandit algorithms, and we solve it in               being controllable by the assigned budget; we can decide on
polynomial time by dynamic programming in a fashion sim-                 a further parameter, the bid, that does not have a correspond-
ilar to the approximation scheme for the knapsack problem.               ing parameter in the knapsack problem; the value of each
The optimization is repeated every day.                                  item is not constant, but it depends on the decisions taken.
   We experimentally evaluate, using a realistic simulator                  Since the function of the number of clicks nj (·, ·) and the
based on the Yahoo! Webscope A3 dataset, the convergence                 parameter specifying the value per click vj need to be es-
of our algorithm to the optimal (clairvoyant) solution and                   3
                                                                              We assume that campaign C and spending plan B are given.
its regret as the size of the problem varies. Furthermore, we                4
                                                                              For the sake of presentation, from now on we set the day as
evaluate our algorithm in a real-world campaign with sev-                unitary temporal step of our algorithm. The use of shorter time
eral subcampaigns in Google AdWords for two consecutive                  units can be used to tackle situations in which the user’s behaviour
months, obtaining the same number of acquisitions obtained               is non-stationary over the day. The application of the proposed al-
by human experts, but halving the cost per acquisition.                  gorithm to different time units is straightforward.




                                                                  2380
timated online, not being a priori known, the optimization                   Algorithm 1 AdComB
problem can be naturally formulated in a sequential decision                 1: Parameters: sets {Xj }N                            N
                                                                                                      j=1 of bid values, sets {Yj }j=1 of budget values, prior
learning fashion (Cesa-Bianchi and Lugosi 2006), or, more                                     (0)
                                                                                   model {Mj }N
                                                                                              j=1 , spending plan B, time horizon T
precisely, as a combinatorial bandit problem (Chen, Wang,                    2: for t ∈ {1, . . . , T } do
and Yuan 2013).5 Here, we would like to gather as much in-                   3:    for j ∈ {1, . . . N } do
formation as possible about the stochastic functions during                  4:        if t = 1 then
                                                                                                        (0)
the operational life of the system and, at the same time, we                 5:             Mj ← Mj
do not want to lose too much revenue in doing so (a.k.a. ex-                 6:        else
ploration/exploitation dilemma). More precisely, the avail-                  7:             Get (ñj,t−1 , c̃j,t−1 , r̃j,t−1 , ṽj,t−1 )
able options (a.k.a. arms) are all the values of bid xj,t and                8:             Mj ← Update (Mj , (x̂j,t−1 , ŷj,t−1 , ñj,t−1 , c̃j,t−1 ,
                                                                                                                                             r̃j,t−1 , ṽj,t−1 ))
budget yj,t satisfying the combinatorial constraints of the
optimization problem, while nj (·, ·) and vj are stochastic                  9:          Xj,t ← Xj ∩ [xj,t , xj,t ]
                                                                             10:         Yj,t ← Yj ∩ [y , y j,t ]
functions deﬁned on the feasible region of the variables that                                               j,t
                                                                             11:          (nj (·, ·), vj ) ← Sampling (Mj , Xj,t , Yj,t )
we need to estimate during the time horizon T . A policy U
solving such a problem is an algorithm returning, for each                   12:       {(x̂j,t , ŷj,t )}j∈N ← Optimize ({nj (·, ·), vj , Xj,t , Yj,t }j∈N , y t )
                                                                             13:       Set ({(x̂j,t , ŷj,t )}j∈N )
day t and subcampaign Cj , a bid/budget pair (x̂j,t , ŷj,t ).
Given a policy U, we deﬁne the pseudo regret as:
                               T N                          
                               
                                                                                                                                             6SHQGLQJ3ODQ
                       ∗
        RT (U) := T G − E                vj nj (x̂j,t , ŷj,t ) ,
                               t=1 j=1                                                                                 %DQGLW
                 N
                                                                                            (VWLPDWLRQ                                      2SWLPL]DWLRQ
                                                                                                                       &KRLFH
          ∗                     ∗ ∗
where G :=          j=1 vj nj (xj , yj ) is the expected value
provided by a clairvoyant algorithm, the set of bid/budget
pairs {(x∗j , yj∗ )}N
                                                                                                                      6HDUFK
                    j=1 is the optimal clairvoyant solution to the
problem in Equations (1a)–(1d), and the expectation E[·] is                                                           6RFLDO


taken w.r.t. the stochasticity of the policy U. Our goal is the                                                     &RQWH[WXDO

design of algorithms minimizing the pseudo regret RT (U).
                                                                             Figure 1: The information ﬂow in the AdComB algorithm
                    Proposed Method                                          along the three phases.
Initially, we provide an overview of our algorithm named
AdComB—Advertising Combinatorial Bandit algorithm—,
and, subsequently, we describe in detail the phases compos-                  actual number of clicks ñj,t−1 , the actual total cost of the
ing the algorithm.                                                           subcampaign c̃j,t−1 , the time when the daily budget y t ﬁn-
                                                                             ished r̃j,t−1 , if so, and the actual value per click ṽj,t−1 . Sub-
The Main Algorithm                                                           sequently, the model of each subcampaign Mj is updated
                                                                             using those observations (Line 8).
Algorithm 1 reports the high-level pseudocode of our                            In the second phase (Lines 9–11), named Bandit Choice
method. For the sake of presentation, we distinguish three                   in Fig. 1, the algorithm chooses the values for the function
phases that are repeated each day t (see Fig. 1). For each                   nj (·, ·) and the parameter vj using the model Mj just up-
subcampaign Cj , the parameters to the algorithm are: a ﬁ-                   dated. More precisely, for each subcampaign Cj , the algo-
nite set Xj of feasible bid values, a ﬁnite set Yj of feasible               rithm initially selects the bids Xj,t := Xj ∩ [xj,t , xj,t ] and
                              (0)
budget values, a model Mj capturing a prior knowledge                        budgets Yj,t := Yj ∩ [y j,t , y j,t ] that are feasible according to
about the function nj (·, ·) and of the parameter vj , a spend-              the given ranges (Lines 9–10). Subsequently, the algorithm
ing plan B and a time horizon T .                                            chooses, according to the probability distributions of Mj ,
   In the ﬁrst phase (Lines 4–8), named Estimation in                        the samples of the function nj (·, ·) for the feasible values of
Fig. 1, the algorithm learns, from the observations of days                  bid and budget in Xj,t and Yj,t and for vj (Line 11).
{1, . . . , t − 1}, the model Mj of the user behavior for each                  In the third phase (Lines 12–13), named Optimization in
subcampaign Cj . More precisely, the model Mj provides a                     Fig. 1, the algorithm uses the values of nj (·, ·) and of vj as
probability distribution over the number of clicks nj (x, y)                 parameters of the problem in Equations (1a)–(1d) and solves
as the bid x and the budget y vary and over the value per                    this problem returning the bid/budget pairs to be set for the
click vj . The ﬁrst day the algorithm is executed, no obser-                 current day t (Line 13) in each different channel (denoted by
vation is available, and thus the model Mj is based on the                   Search, Social and Contextual in Fig. 1).6
             (0)
prior Mj (Line 5). Conversely, the subsequent days, for                         In what follows, we provide a detailed description of the
each subcampaign Cj , the algorithm also gets the obser-                     model Mj and of the subroutines Update(·), Sampling(·),
vations corresponding to day t − 1 (Line 7) including: the
                                                                                  6
                                                                                    Notice that, since the bid and the budget can assume a ﬁnite
    5
     Another approach to solving the problem is to use a multistage          set of values, the problem in Equations (1a)–(1d) can be easily for-
method, e.g., backward induction, but it would require a huge com-           mulated as a Mixed Integer Linear Program (see the Supplemental
putational effort that makes the problem intractable.                        Material for the mathematical programming formulation).




                                                                      2381
and Optimize(·) used in Algorithm 1.                                      x, the number of clicks increases linearly in the budget y
                                                                          where the coefﬁcient is the average cost per click given x
Model and Update Subroutine                                                     nsat (x)
                                                                                  j
                                                                          (i.e., csat
                                                                                 j    (x) ), until the maximum number of obtainable clicks
As mentioned before, the goal of this subroutine is the
estimation of the functions nj (·, ·) and the parameter vj .              is achieved. Notice that the values of nsat          sat
                                                                                                                    j (x) and cj (x) de-
The crucial issue, in this case, concerns the employment                  pend on the average position in which the ad is displayed
of a practical estimation model, providing a good tradeoff                when bid x is used and on the daily number of auctions. The
between accuracy and time needed for the learning pro-                    larger x the larger nsat           sat
                                                                                                j (x) and cj (x).
cess. For instance, let us observe that a straightforward ap-                Now we focus on the modeling of the maximum number
proach employing independent estimates of nj (·, ·) for ev-               of clicks nsat
                                                                                      j (·) with a GP regression model. The application
ery bid/budget pair (x, y) is not practical since it would re-            of such techniques to estimate the maximum cost csat    j (·) is
quire a huge amount of observations, and, thus, too many                  analogous. We model nsat  j (·) in a subcampaign Cj with a GP
days to have accurate estimates. Suppose, for instance, to                over Xj , i.e., we use a collection of random variables s.t. any
use 10 bid values and 10 budget values, with a total num-                 ﬁnite subset has a joint Gaussian distribution. Following the
ber of 100 bid/budget pairs. Such a discretization would                  deﬁnition provided in (Rasmussen and Williams 2006), a GP
require a period of 100 days only to have a single obser-                 is completely speciﬁed by its mean m : Xj → R and covari-
vation per estimates and years to have accurate estimates,                ance k : Xj × Xj → R functions. Hence, we denote the GP
thus making the algorithm useless in practice. Most of the                that models the maximum number of clicks in Cj as follows:
methods for combinatorial bandits available in the state of
                                                                                     nsat
                                                                                      j (x) := GP (m(x), k(x, ·)) , ∀x ∈ Xj .
the art (Chen, Wang, and Yuan 2013; Chen et al. 2016;
Gai, Krishnamachari, and Jain 2010; Ontañón 2017) suffers               If we have a priori information about the process, we can
from the same issue, not exploiting any correlation among                 use it to design a function m(x) over the input space Xj
the random variables corresponding to the arms rewards.                   which speciﬁes the initial mean value, e.g., if we have in-
   To address this issue, we assume that the function nj (·, ·)           formation about the maximum number of clicks we might
presents some regularities and, in particular, that the values            reach θ for any bid, one might consider a linearly increasing
of the function at different points in the bid/budget space are           function over the bid space as m(x) = maxθt xj,t x. If no a
correlated. We capture this regularity resorting to GPs (Ras-             priori information is available, we set m(x) = 0, ∀x ∈ Xj .
mussen and Williams 2006). These models, developed in the                    At the beginning of the optimization procedure (t = 1),
statistical learning ﬁeld, express the correlation of the nearby          we have the same predictive distribution at each point of the
points in the input space exploiting a kernel function. More-             input space, i.e., nsat
                                                                                               j (x) ∼ N (m(x), k(x, x)), where we
over, they provide a probability distribution over the output
space—in our case the number of clicks—for each point of                  denote with N (μ, σ 2 ) the Gaussian distribution with mean
the input space—in our case the space of bid and budget—,                 μ and variance σ 2 . For each day t we obtain a value for the
thus giving information both on the expected values of the                maximum number of clicks by relying on:
quantities to estimate as well as their uncertainty.                                             ñsat
                                                                                                   j,t := d(r̃j,t , ñj,t ),
   In particular, we propose two approaches for nj (·, ·). A
straightforward approach, which will be used as a baseline                where d(·, ·) is a function specifying the distribution of the
in our experimental activity, employs a single GP deﬁned on               clicks over the day. The function d(·, ·) can be estimated
a 2-dimension input space (details are provided in the Sup-               from historical data coming from past advertising campaigns
plemental Material). Even if this method provides a ﬂexible               of products belonging to the same category (e.g., toys, in-
way of modeling the advertising phenomenon, it requires,                  surances, beauty products). The vector of the bid set so far
                                                                                                               T
due to the curse of dimensionality, a long initial phase before           x̂j,t−1 := (x̂j,1 , . . . , x̂j,t−1 ) and the vector of maximum
being effective. This issue is addressed by our second ap-                                                                       T
                                                                          number of clicks ñsat                 sat        sat
                                                                                                 j,t−1 := ñj,1 , . . . , ñj,t−1    are used by
proach which exploits an assumption on the structure of the
                                                                                                                                    (0)
problem. More precisely, the dependency of nj (x, y) from                 the algorithm to reﬁne the initial prior model Mj . From
bid x and budget y is modeled by two 1-dimensional GPs                    the deﬁnition of GP, its restriction over a ﬁnite number of
combined in a nonlinear fashion. Formally, we assume:                     points is a multivariate Gaussian random variable, which can
                                                                        be used, for every x ∈ Xj , to predict the expected value
                                                                                                     2
                          sat                 y                           μj,t−1 (x) and variance σj,t−1  (x) of the maximum number
            nj (x, y) := nj (x) min 1, sat           ,       (2)
                                           cj (x)                         of clicks in the following way:
                                                                                                                       
where the two GPs employed for each subcampaign Cj are:                    μj,t−1 (x) = m(x) + K(x, x̂j,t−1 )Φ−1 ñsat      j,t−1 −
                                                                                                                                          
– the maximum number of clicks nsat                +
                                    j : Xj → R that can                                                                                 T
                                                                                                        (m(x̂j,1 ), . . . , m(x̂j,t−1 )) ,
be obtained with a given bid x without any budget constraint
(or equivalently if we let y → +∞);                                        2
                                                                          σj,t−1 (x) = k(x, x) − K(x, x̂j,t−1 ) Φ−1 K(x, x̂j,t−1 )T ,
– the maximum cost incurred csat             +
                                j : Xj → R with a given
bid x without any budget constraint, as above;                            where we deﬁne Φ := K(x̂j,t−1 , x̂j,t−1 ) + σn2 I, I is the
where the bid space is deﬁned as Xj := ∪Tt=1 [xj,t , xj,t ]. The          identity matrix of order t − 1, and the (i, h)-element of the
rationale behind this decoupled model is that, given a bid                matrix K(x, x ) is the value of the kernel computed over



                                                                   2382
the i-th element of the generic vector x and the h-th ele-                           provided in the previous section). Similarly, for the value per
ment of the generic vector x . Hence, the maximum num-                              click we draw a new sample:
ber of clicks for the bid x is distributed as the Gaussian
                2
N (μj,t−1 (x), σj,t−1 (x)).                                                                            v̂j ∼ N (νj,t−1 , φ2j,t−1 ).
  Regarding the value per click vj , at day t we estimate it by                         Conversely, the AdComB-BUCB algorithm generates
exploiting the data ṽj,h recorded during the days h < t. Re-                        samples for nsat            sat
                                                                                                    j (x) and cj (x) exploiting different time-
sorting to the Central Limit theorem, we have that the mean                          varying quantiles of the posterior distributions. More pre-
value per click vj is asymptotically Gaussian distributed,                           cisely, we use a high quantile (of order 1 − 1t ) for nsat
                                                                                                                                            j (x)
thus, at each day t, it is sufﬁcient to estimate its mean νj,t                                                             1       sat
                                                                                     and vj and a low quantile (of order t ) for cj (x). This as-
and variance φ2j,t as follows:                                                       sures us to generate optimistic bounds, that are necessary
            t−1                            t−1                      2              for the convergence of the algorithm to the optimal solution.
               h=1 ṽj,h                      h=1 (ṽj,h − νj,t−1 )                  Let us denote with q(μ, σ 2 , p) the quantile of order p of a
νj,t−1 :=                  ,   φ2j,t−1 :=                                 .
                t−1                              (t − 1)(t − 2)                      Gaussian distribution with mean μ and variance σ 2 . At day
                                                                                     t, for each bid x ∈ Xj,t , we generate samples for nsatj (x)
   Overall, the model Mj corresponding to a subcampaign                                    sat
                                                                                     and cj (x) as:
Cj at a day t consists of the following vectors: the values per
                                            T
click ṽj,t−1 := (ṽj,1 , . . . , ṽj,t−1 ) , the selected bids x̂j,t−1 ,                                                            1
                                                                                            nsat                    2
                                                                                             j (x) = q μj,t−1 (x), σj,t−1 (x), 1 −          ,
the maximum number of clicks nsat              j,t−1 , and the maximum                                                               t
        sat                                   sat
costs cj,t−1 (deﬁned similarly to nj,t−1 ). Therefore, the Up-                                                                       1
date subroutine of Algorithm 1 includes the incoming data                                         csat                    2
                                                                                                   j (x) = q ηj,t−1 (x), sj,t−1 (x),        ,
                                                                                                                                     t
(ñj,t−1 , c̃j,t−1 , r̃j,t−1 , ṽj,t−1 ), properly transformed, in the
aforementioned vectors.7                                                             assuring that nsat
                                                                                                      j (x) is a high-probability upper bound
                                                                                     for the maximum number of clicks and csat j (x) is a high-
Sampling Subroutine                                                                  probability lower bound for the maximum costs. Similarly,
                                                                                     for the value per click vj we assign:
The models Mj we estimate for each subcampaign pro-
vide a probability distribution over the function nj (·, ·) and                                                                     1
of the values vj and, therefore, over the possible instances                                       v̂j = q νj,t−1 , φ2j,t−1 , 1 −       .
                                                                                                                                    t
of the optimization problem in Equations (1a)–(1d). The
Sampling subroutine generates, from Mj , a single instance                              Finally, given the values for nsat            sat
                                                                                                                        j (x) and cj (x) gener-
of the optimization problem, assigning a value to nj (x, y)                          ated by one of the two aforementioned methods, we compute
for every x ∈ Xj , y ∈ Yj and a value to vj . In the                                 nj (x, y) as prescribed by Equation (2) for each x ∈ Xj,t and
present paper, we propose two novel Bayesian approaches,                             for each y ∈ Yj,t .
namely AdComB-TS and AdComB-BUCB, taking inspira-
tion from the Thompson Sampling (TS) algorithm (Thomp-                               Optimize Subroutine
son 1933) and the BayesUCB algorithm (Kaufmann, Cappé,                              Finally, we need to decide a single bid/budget pair to set
and Garivier 2012), respectively. We resort to the Bayesian                          at day t for each subcampaign Cj . We resort to a modiﬁed
approach since, in most of the bandit scenarios, it leads to                         version of the algorithm in (Kellerer, Pferschy, and Pisinger
better performance than that one of their frequentist coun-                          2004) used for the solution of the knapsack problem. For the
terparts, see, e.g., (Chapelle and Li 2011; Granmo 2010;                             sake of simplicity, let us assume we set an evenly spaced
May et al. 2012; Paladino et al. 2017).                                              discretization Y of the daily cumulative budget y t and that
   The AdComB-TS algorithm generates the values for                                  the feasible values for the budget are a subset of such a dis-
nsat           sat
  j (x) and cj (x) by drawing samples from the posterior                             cretization, i.e., Yj,t ⊆ Y, ∀j, t. At ﬁrst, for each value of
distributions provided by the GPs. More formally, at a given                         budget y ∈ Yj,t we deﬁne zj (y) ∈ Xj,t as the bid maximiz-
day t for each bid in x ∈ Xj,t , we draw a sample for nsat
                                                         j (x)                       ing the number of clicks, formally:
and a sample for csat (x) as follows:
                    j                                                                                zj (y) := arg max nj (x, y).
                                                                                                                   x∈Xj,t
              nsat                       2
                j (x) ∼ N (μj,t−1 (x), σj,t−1 (x)),
                                                                                     The value zj (y) is easily found by enumeration. Then, for
              csat                     2
               j (x) ∼ N (ηj,t−1 (x), sj,t−1 (x)),                                   each value of budget y ∈ Y we deﬁne wj (y) as the value we
                                                                                     expect to receive by setting the budget of subcampaign Cj
where ηj,t−1 (x) and s2j,t−1 (x) are the mean and the variance                       equal to y and the bid equal to zj (y), formally:
for bid x, respectively, estimated by the GP modeling csat
                                                         j (x)                                     
                                                  2
(we recall that the deﬁnitions of μj,t−1 (x) and σj,t−1 (x) are                                       v̂j nj (zj (y), y) y j,t ≤ y ≤ y j,t
                                                                                        wj (y) :=                                              .
                                                                                                      0                  y < y j,t ∨ y > y j,t
     7
       The computation cost of the proposed solution can be dramat-
ically reduced by using an alternative, but much more involved, so-                  This allows one to discard x from the set of the variables
lution where the inverse of the Gram matrix K(x̂j,t−1 , x̂j,t−1 )−1                  of the optimization problem deﬁned in Equations (1a)–(1d),
is stored and updated iteratively at each day; see (Bishop 2006).                    letting variables y the only variables to deal with.



                                                                              2383
   Finally, the optimization problem is solved in dynamic                  uniform distribution of the clicks over the day, thus the func-
programming fashion. We use a matrix M (j, y) with j ∈                     tion d(·, ·) has the following expression (r̃j,h is expressed in
{1, . . . , N } and y ∈ Y . We ﬁll iteratively the matrix as fol-          hours):
lows. Each row is initialized as M (j, y) = 0 for every j and                                                       24
                                                                                              d(r̃j,h , ñj,h ) :=       ñj,h .
y ∈ Y . For j = 1, we set M (1, y) = w1 (y) for every y ∈ Y ,                                                      r̃j,h
corresponding to the best budget assignment for every value
of y if the subcampaign Cj were the only subcampaign in                      We compare the AdComB-TS and AdComB-BUCB al-
the problem. For j > 1, we set for every y ∈ Y :                           gorithms with:8
                                                                           • AdComB-2D-TS, a version of AdComB using a single
   M (j, y) =      max          M (j − 1, y  ) + wj (y − y  ) .            GP over the two dimensional bid/budget space to estimate
                y  ∈Y,y  ≤y
                                                                             the number of clicks (see the Supplemental Material);
That is, the value in each cell M (j, y) is found by scan-                 • AdComB-Mean, a version of AdComB selecting at each
ning all the elements M (j − 1, y  ) for y  ≤ y, taking the                day the average values μj,t−1 (x) and ηj,t−1 (x) for bid x
corresponding value, adding the value given by assigning a                   and νj,t−1 to be used in the optimization procedure.
budget of y − y  to subcampaign Cj and, ﬁnally, taking the
maximum among all these combinations. At the end of the                    For the GPs used in the algorithms, we adopt a squared ex-
recursion, the optimal value of the optimization problem can               ponential kernel of the form:
be found in the cell corresponding to maxy∈Y M (N, y). To                                                                  
                                                                                                               (z − z  )2
ﬁnd the optimal assignment of the budget, it is sufﬁcient to                           k(z, z  ) := σf2 exp −               ,
also store the partial assignments of the budget correspond-                                                       l
ing to the optimal value. The complexity of the aforemen-                  where σf , l ∈ R+ are kernel parameters, whose values are
tioned algorithm is O(N H 2 ), i.e., it is linear in the number            chosen as prescribed by the GP literature, see (Rasmussen
of subcampaigns N and quadratic in the number of different                 and Williams 2006) for details.
values of the budget H := |Y |, where | · | is the cardinal-                  In addition to the pseudo regret Rt (U), we also evaluate
ity operator. (Let us observe that, although the complexity                the instantaneous reward which is deﬁned as:
is polynomial in H, it is pseudopolynomial in y j,t .) When
                                                                                                       N
                                                                                                       
H is huge, the algorithm could require a long time. In that
case, it is sufﬁcient to reduce H by rounding the values of                                Pt (U) :=         vj nj (x̂j,t , ŷj,t ).
the budget as in the FPTAS of the knapsack problem. This                                               j=1
produces a (1 − ε)-approximation of the optimal solution.
                                                                           We average the results over 100 independent runs.
                                                                              In Fig. 2a, we report Pt (U) of the 4 algorithms, while, in
               Experimental Evaluation                                     Fig. 2b, we report their average Rt (U). By inspecting the
We experimentally evaluate our algorithm both in a synthetic               instantaneous reward, we can see that all the algorithms but
setting, necessary to evaluate the convergence and the regret              AdComB-Mean present a slightly varying reward even at
of our algorithm, and in a real-world setting, necessary to as-            the end of the of the time horizon since they incorporate
sess its effectiveness when compared with the performance                  the variance of the GP as information to select the budget
of human experts.                                                          over time. This variance is larger at the beginning of the
                                                                           process, thus incentivising exploration, and it fades as the
Evaluation in the Synthetic Setting                                        number of observations increases, allowing the algorithm to
The synthetic setting we use has been generated according to               reach the optimum asymptotically. Moving to Fig. 2b, we
data of the Yahoo! Webscope A3 dataset using the simulator                 observe that the 2 Bayesian algorithms, namely AdComB-
developed by Farina and Gatti (2017).                                      TS and AdComB-BUCB, present essentially the same per-
   Experiment 1. We consider N = 4 subcampaigns, each                      formance, providing the best regret for every t ≤ T . For
with a variable number of daily auctions drawn from z                      t ≤ 20 the regret of AdComB-Mean is essentially the same
with z ∼ N (1000, 10). Each auction presents 5 slots and                   one of the 2 Bayesian algorithms. Instead, for larger values
10 advertisers. Each subcampaign is associated with a dif-                 of t its relative performance worsens reaching, at t = 100,
ferent truncated Gaussian distribution that is used at every               a regret about 35% larger than the one of the AdComB-
day t to draw the bid of each ad, while, in the same way,                  BUCB algorithm. This is because the exploration performed
the ads’ click probability is drawn from a Beta distribution;              by AdComB-Mean is not sufﬁcient. As a result, in all the
see (Farina and Gatti 2017) for details. We set a constant cu-             runs, AdComB-Mean does not change the policy for, ap-
mulative budget per day y t = 100 over a time horizon of                   proximately, t ≥ 40 and, in some of the 100 independent
T = 100 days, with limits y j,t = 0, y j,t = 100 for every                 runs, it gets stuck in a suboptimal solution, as we can ob-
                                                                           serve in Fig. 2a. Conversely, the 2 Bayesian algorithms,
t, j, and we set bid limits to xj,t = 0 and xj,t = 1 for every
t, j. Furthermore, we use an evenly spaced discretization of                   8
                                                                                The comparison with the algorithm by Chen, Wang, and
|Xj | = 5 bids and |Yj | = 10 budgets over the aforemen-                   Yuan (2013)—in a version accounting for Gaussian distributions—
tioned intervals. The values per click of each subcampaign                 is unfair. Indeed, this algorithm requires 50 days to have a single
are constant—thus letting the clicks to be the only source of              sample per random variable, and, for all t ≤ T , it purely explores
randomness—and drawn uniformly from [0, 1]. We assume a                    the space of arms without any form of exploitation.




                                                                    2384
                                                           300                                               300
                                                                                                                                            300
         18
                                                                                                             250
         16                                                200
Pt (U)                                            Rt (U)                                            RT (U)
                                                                                                             200                            200
         14                      AdComB-TS
                                 AdComB-Mean               100                                               150
         12
                                 AdComB-BUCB
         10                      AdComB-2D-TS                                                                100                            100
                                                             0
              0   20   40         60   80   100                  0   20    40       60   80   100                  5    10             20         3   4         5   6
                             t                                                  t                                              |Xj |                        N

                       (a)                                                (b)                                            (c)                              (d)

Figure 2: Results for the synthetic setting: instantaneous reward of Experiment 1 (a), pseudo regret over time for Experiment
1 (b), pseudo regret at the end of the time horizon for Experiment 2 (c) and Experiment 3 (d). The optimum value of the
instantaneous reward G∗ is represented with a dashed line in (a).


thanks to a wider exploration, converge to the optimal solu-                                            geneous over time as possible. During the ﬁrst 2 months, the
tion asymptotically in all the runs (the phenomenon is more                                             bid/budget optimization has been performed by human ex-
evident on a longer time horizon, see the Supplemental Ma-                                              perts, leading to an average of 350 acquisitions per month
terial). Finally, we observe that AdComB-2D-TS suffers                                                  with an average cost per acquisition of about 83 e. After the
from a larger regret than that one of the other 3 algorithms—                                           ﬁrst 2 months, the optimization has been performed by the
more than 100% compared with the regret of AdComB-TS                                                    AdComB-TS algorithm in a completely automated fashion.
and AdComB-BUCB at t = 100—and this is mainly accu-                                                     The goal of the company was to reduce the cost per acquisi-
mulated over the ﬁrst half of the time horizon.                                                         tion, given that the cost of 104 e was considered excessively
   Experiment 2. The experimental setting is the same used                                              large, keeping the same number of acquisitions per month
above, except that we use |Xj | ∈ {5, 10, 20}, s.t. the set                                             obtained during the ﬁrst 2 months. We used a discretization
of bids we used Xj with |Xj | = 20 includes Xj with                                                     of 5 e for the budget values and 0.10 e for the bid values.
|Xj | = 10 that, in its turn, includes Xj with |Xj | = 5.                                               The algorithm, implemented in Python 2.7.12 and executed
In Fig. 2c we report the average RT (U) of the 4 algorithms.                                            on Ubuntu 16.04.1 LTS with an Intel(R) Xeon(R) CPU E5-
Even if increasing the number of bid values may allow one                                               2620 v3 2.40GHz, was used at the midnight of each day to
to increase the value of the optimal solution, we observe                                               decide the bid/budget pairs for the next day. The maximum
that the regret slightly increases as |Xj | increases for the                                           computation time of the algorithm during the 2 months was
AdComB-TS and AdComB-2D-TS algorithms. This is be-                                                      less than 1 minute. During the 2 months AdComB-TS was
cause the algorithms pays a larger exploration cost. Remark-                                            executed, it obtained 353 conversions with an average cost
ably, the extra cost from |Xj | = 10 to |Xj | = 20 is small.                                            per acquisition of about 56 e. More precisely, in the ﬁrst
This result shows that the performance of the algorithms is                                             month the algorithm was used, the cost per acquisition was
robust to an increase of the number of possible bids.                                                   about 62 e, while, in the second one, about 50 e. Thus, the
   Experiment 3. The experimental setting is the same used                                              average reduction of the cost per acquisition during the ﬁrst
in the Experiment 1, except that N ∈ {3, 4, 5, 6}. In Fig. 2d,                                          month of execution of the algorithm has been about 25%,
we report the average regret RT (U) of the 4 algorithms. All                                            while during the second one about 40%.
the algorithms (except AdComB-2D-TS) do not suffer from
a signiﬁcant increase in the regret as the number of the sub-                                                          Conclusions and Future Works
campaigns increases, showing that they scale well as the
number of subcampaigns increases.                                                                       In the current paper, we present AdComB, an algorithm ca-
                                                                                                        pable of deciding automatically the values of the bid and
                                                                                                        the budget to set in an advertising campaign to maximize
Evaluation in a Real-world Setting                                                                      in online fashion the value of the campaign given a spend-
We advertised a campaign for a loan product of a large Ital-                                            ing plan. The algorithm exploits Gaussian Processes to es-
ian ﬁnance company with the AdComB-TS algorithm (the                                                    timate the users’ model, combinatorial bandit techniques to
names of the product and the company, and other details                                                 address the exploration/exploitation dilemma, and optimiza-
omitted below are not provided due to reasons of indus-                                                 tion techniques to solve a knapsack-like problem. We pro-
trial secrecy). The advertising campaign was composed of                                                pose two ﬂavours of the algorithm, namely AdComB-TS
N = 13 subcampaigns. The campaign has been advertised                                                   and AdComB-BUCB, differing for the criterion used for the
for T = 120 days (4 months) in 2017 during which no                                                     bandit choice. Experiments on both a realistic synthetic set-
further advertising campaigns (e.g., video, radio, television)                                          ting and real-world setting show that our algorithms tackle
were conducted to avoid mutual effects between the cam-                                                 the problem properly, outperforming other naive algorithms
paigns. The 4 months during which the experiments have                                                  based on existing solutions and the human expert.
been conducted were chosen in such a way, according to past                                                As future work, we plan to study the theoretical properties
observations, the click behaviour of the users was as homo-                                             of the pseudo regret of our algorithm, as well as the study of



                                                                                              2385
techniques to provide a proper setup of the subcampaigns.                   Kellerer, H.; Pferschy, U.; and Pisinger, D. 2004. The Multiple-
While in the present work we assume that the environment,                   Choice Knapsack Problem. Springer. 317–347.
including the users and the other advertisers, is stationary                King, M.; Atkins, J.; and Schwarz, M. 2007. Internet advertis-
over time, we will investigate non-stationary environments,                 ing and the generalized second-price auction: Selling billions of
e.g., including in the model the option that there exists peri-             dollars worth of keywords. AM ECON REV 97(1):242–259.
odicity in the user behaviour, as well as some sudden change                Kireyev, P.; Pauwels, K.; and Gupta, S. 2016. Do display ads
due the modiﬁcation of the competitors marketing policy.                    inﬂuence search? attribution and dynamics in online advertising.
Moreover, another interesting line of research is to design                 INT J RES MARK 33(3):475–490.
methods to set up the subcampaigns and possibly modify                      Lee, K.-C.; Jalali, A.; and Dasdan, A. 2013. Real time bid
their targeting over time, basing on their performance.                     optimization with smooth budget delivery in online advertising.
   Acknowledgments. This research has been funded by the                    In ADKDD, 1–9.
Mediamatic company, part of the MMM group. We sin-                          Markakis, E., and Telelis, O. 2010. Discrete strategies in key-
cerely thank Roberto Coronel Da Silva, Enrico Dellavalle,                   word auctions and their inefﬁciency for locally aware bidders.
and Paola Corbani for their valuable support.                               In WINE, 523–530.
                                                                            May, B. C.; Korda, N.; Lee, A.; and Leslie, D. S. 2012. Op-
                         References                                         timistic bayesian sampling in contextual-bandit problems. J
Bishop, C. M. 2006. Pattern recognition and machine learning.               MACH LEARN RES 13(Jun):2069–2106.
Springer.                                                                   Ontañón, S. 2017. Combinatorial multi-armed bandits for real-
Cesa-Bianchi, N., and Lugosi, G. 2006. Prediction, learning,                time strategy games. J ARTIF INTELL RES 58:665–702.
and games. Cambridge University Press.                                      Paladino, S.; Trovò, F.; Restelli, M.; and Gatti, N. 2017. Uni-
Chapelle, O., and Li, L. 2011. An empirical evaluation of                   modal thompson sampling for graph-structured arms. In AAAI.
thompson sampling. In NIPS, 2249–2257.                                      Qin, T.; Chen, W.; and Liu, T.-Y. 2015. Sponsored search auc-
Chen, W.; Wang, Y.; Yuan, Y.; and Wang, Q. 2016. Combina-                   tions: Recent advances and future directions. ACM T INTEL
torial multi-armed bandit and its extension to probabilistically            SYST TEC 5(4):60:1–60:34.
triggered arms. J MACH LEARN RES 17(1):1746–1778.                           Rasmussen, C. E., and Williams, C. K. 2006. Gaussian pro-
Chen, W.; Wang, Y.; and Yuan, Y. 2013. Combinatorial multi-                 cesses for machine learning, volume 1. MIT Press.
armed bandit: General framework and applications. In ICML,                  Sinha, P., and Zoltners, A. A. 1979. The multiple-choice knap-
151–159.                                                                    sack problem. OPER RES 27(3):503–515.
Ding, W.; Qin, T.; Zhang, X.-D.; and Liu, T. 2013. Multi-armed              Thomaidou, S.; Liakopoulos, K.; and Vazirgiannis, M. 2014. To-
bandit with budget constraint and variable costs. In AAAI, 232–             ward an integrated framework for automated development and
238.                                                                        optimization of online advertising campaigns. INTELL DATA
Farina, G., and Gatti, N. 2017. Adopting the cascade model in               ANAL 18(6):1199–1227.
ad auctions: Efﬁciency bounds and truthful algorithmic mecha-               Thompson, W. R. 1933. On the likelihood that one unknown
nisms. J ARTIF INTELL RES 59:265–310.                                       probability exceeds another in view of the evidence of two sam-
Gai, Y.; Krishnamachari, B.; and Jain, R. 2010. Learning mul-               ples. BIOMETRIKA 25(3/4):285–294.
tiuser channel allocations in cognitive radio networks: A com-              Trovò, F.; Paladino, S.; Restelli, M.; and Gatti, N. 2016. Bud-
binatorial multi-armed bandit formulation. In DySPAN, 1–9.                  geted multi-armed bandit in continuous action space. In ECAI,
IEEE.                                                                       560–568.
Gatti, N.; Lazaric, A.; Rocco, M.; and Trovò, F. 2015. Truthful            Varian, H. R., and Harris, C. 2014. The VCG auction in theory
learning mechanisms for multi-slot sponsored search auctions                and practice. AM ECON REV 104(5):442–445.
with externalities. ARTIF INTELL 227:93–139.
                                                                            Wang, J.; Zhang, W.; and Yuan, S. 2016. Display advertising
Geyik, S. C.; A-Saxena; and Dasdan, A. 2014. Multi-touch                    with real-time bidding (RTB) and behavioural targeting. CoRR
attribution based budget allocation in online advertising. In AD-           abs/1610.03013.
KDD, 1–9.
                                                                            Weinan, W.; Rong, Y.; Wang, J.; Zhu, T.; and Wang, X. 2016.
Geyik, S. C.; Faleev, S.; Shen, J.; O’Donnell, S.; and Kolay, S.            Feedback control of real-time display advertising. In WSDM,
2016. Joint optimization of multiple performance metrics in on-             407–416.
line video advertising. In SIGKDD, 471–480.
                                                                            Xia, Y.; Li, H.; Qin, T.; Yu, N.; and Liu, T.-Y. 2015. Thompson
Granmo, O.-C. 2010. Solving two-armed bernoulli bandit prob-                sampling for budgeted multi-armed bandits. In IJCAI, 3960–
lems using a bayesian learning automaton. IJICC 3(2):207–234.               3966.
IAB. 2016. Iab internet advertising revenue report 2016, full               Xu, J.; Lee, K.-C.; Li, W.; Qi, H.; and Lu, Q. 2015. Smart pac-
year results. https://www.iab.com. Online; accessed 21 July                 ing for effective online ad campaign optimization. In SIGKDD,
2017.                                                                       2217–2226.
Italia, E. M.; Nuara, A.; Trovò, F.; Restelli, M.; Gatti, N.; and          Zhang, W.; Zhang, Y.; Gao, B.; Yu, Y.; Yuan, X.; and Liu, T.-
Dellavalle, E. 2017. Internet advertising for non-stationary en-            Y. 2012. Joint optimization of bid and budget allocation in
vironments. In AMEC, 1–15.                                                  sponsored search. In SIGKDD, 1177–1185.
Kaufmann, E.; Cappé, O.; and Garivier, A. 2012. On bayesian                Zhang, W.; Yuan, S.; and Wang, J. 2014. Optimal real-time
upper conﬁdence bounds for bandit problems. In AISTATS, 592–                bidding for display advertising. In SIGKDD, 1077–1086.
600.




                                                                     2386
