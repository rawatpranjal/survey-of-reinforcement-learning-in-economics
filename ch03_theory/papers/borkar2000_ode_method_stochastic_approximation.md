## The O/.D/.E/. Method for Convergence of Stochastic Approximation and Reinforcement Learning V/.S/. Borkar / S/.P/. Meyn y February /9/, /1/9/9/9

Stochastic approximation/, o/.d/.e/. method/, forcement learning/.

stability/, asynchronous algorithms/, rein/-

Abstract It is shown here that stability of the stochastic approximation algorithm is implied by the asymptotic stability of the origin for an associated o/.d/.e/. This in turn implies convergence of the algorithm/. Several speci/c classes of algorithms are considered as applications/. It is found that the results provide /(i/) a simpler derivation of known results for reinforcement learning algorithms/; /(ii/) a proof for the /rst time that a class of asynchronous stochastic approximation algorithms are convergent without using any a priori assumption of stability/. /(iii/) a proof for the /rst time that asynchronous adaptive critic and Q/-learning algorithms are convergent for the average cost optimal control problem/.

/ Department of Computer Science and Automation/, Indian Institute of Science/, Bangalore /5/6/0 /0/1/2/, India /(borkar/@csa/.iisc/.ernet/.in/)/. Work supported in part by the Dept/. of Science and Technology /(Govt/. of India/) grant no/.III/5/(/1/2/)///9/6/-ET/. y Department of Electrical and Computer Engg/. and the Coordinated Sciences Laboratory/, Uni/. of Illinois at Urbana/-Champaign/, Urbana/, IL /6/1/8/0/1/, U/.S/.A/. /(s/-meyn/@uiuc/.edu/)/. Work supported in part by NSF grant ECS /9/4/0/3/7/2/, JSEP grant N/0/0/0/1/4/-/9/0/-J/-/1/2/7/0/. This research was completed while the second author was a visiting scientist at the Indian Institute of Science under a Fulbright Research Fellowship/.

AMS Math/. Subject Classi/cation /(/1/9/9/0/)/: /6/2L/2/0/, /9/3E/2/5/, /9/3E/1/5

Key Words/:

The stochastic approximation algorithm considered in this paper is described by the d/dimensional recursion X/(n /+ /1/) /= X/(n/) /+ a/(n/)/[h/(X/(n/)/) /+ M/(n /+ /1/)/]/; n / /0/; /(/1/) where X/(n/) /= /[X /1 /(n/)/; / / / /; X d /(n/)/] T /2 I R d /, h /: I R d /! I R d /, and fa/(n/)g is a sequence of positive numbers/. The sequence fM/(n/) /: n / /0g is uncorrelated with zero mean/. Though more than four decades old/, the stochastic approximation algorithm is now of renewed interest due to novel applications to reinforcement learning /[/2/0/] and as a model of learning by boundedly rational economic agents /[/1/9/]/. Traditional convergence analysis usually shows that the recursion /(/1/) will have the desired asymptotic behavior provided that the iterates remain bounded with probability one/, or that they visit a prescribed bounded set in/nitely often with probability one /[/3/, /1/4/]/. Under such stability or recurrence conditions one can then approximate the sequence X /= fX/(n/) /: n / /0g with the solution to the ordinary di/erential equation /(o/.d/.e/./) /\_ x/(t/) /= h/(x/(t/)/) /(/2/) with identical initial conditions x/(/0/) /= X/(/0/)/. The recurrence assumption is crucial/, and in many practical cases this becomes a bottle/neck in applying the o/.d/.e/. method/. The most successful technique for establishing stochastic stability is the stochastic Lyapunov function approach /(see e/.g/. /[/1/4/]/)/. One also has tech/niques based upon the contractive properties or homogeneity properties of the functions involved /(see/, e/.g/./, /[/2/0/] and /[/1/2/] respectively/)/. The main contribution of this paper is to add to this collection another general technique for proving stability of the stochastic approximation method/. This technique is inspired by the /uid model approach to stability of networks developed in /[/9/, /1/0/]/, which is itself based upon the multistep drift criterion of /[/1/5/, /1/6/]/. The idea is that the usual stochastic Lyapunov function approach can be di/cult to apply due to the fact that time/-averaging of the noise may be necessary before a given positive valued function of the state process will decrease towards zero/. In general such time averaging of the noise will require infeasible calculation/. /1

on the magnitude of the initial state/, to replace the stochastic system of interest with a simpler deterministic process/. The scaling applied in this paper to approximate the model /(/1/) with a deterministic

/ / /

process is similar to the construction of the /uid model of /[/9/, /1/0/]/. Suppose that the state is scaled by its initial value to give e X/(n/) /= X/(n/)/= max/(jX/(/0/)j/; /1/)/, n / /0/. We then scale time to obtain a continuous function / /: I R /+ /! I R d which interpolates the values of f e X/(n/)g/: At a sequence of times ft/(j/) /: j / /0g we set //(t/(j/)/) /= e X/(j/)/, and for arbitrary t / /0 we extend the de/nition by linear interpolation/. The times ft/(j/) /: j / /0g are de/ned in terms of the constants fa/(j/)g used in /(/1/)/. For any r /&gt; /0 the scaled function h r /: I R d /! I R d is given by hr /(x/) /= h/(rx/)/=r/; x /2 I R d /: /(/3/) Then through elementary arguements we /nd that the stochastic process / approximates the solution b / to the associated o/.d/.e/. /\_ x/(t/) /= h r /(x/(t/)/)/; t / /0/: /(/4/) with b //(/0/) /= //(/0/) and r /= max/(jX/(/0/)j/; /1/)/. With our attention on stability considerations/, we are most interested in the behavior of X when the magnitude of the initial condition jX/(/0/)j is large/. Assuming that the limiting function h /1 /= lim r/!/1 h r exists/, for large initial conditions we /nd that / is approximated by the solution / /1 of the limiting o/.d/.e/. /\_ x/(t/) /= h /1 /(x/(t/)/)/: /(/5/) where again we take identical initial conditions / /1 /(/0/) /= //(/0/)/. So/, for large initial conditions all three processes are approximately equal/, / / b /1 Using these observations we /nd in Theorem /2/./1 that the stochastic model /(/1/) is stable in a strong sense provided the origin is asymptotically stable for the limiting o/.d/.e/. /(/5/)/. Equation /(/5/) is precisely the /uid model of /[/9/, /1/0/]/. /2

to establish both the stability and convergence of the stochastic approximation method/, as opposed to only the latter/. The result /[/1/4/, Theorem /4/./1/, p/. /1/1/5/] arrives at a similar conclusion/: if the o/.d/.e/. /(/2/) possesses a /`global/' Lyapunov function with bounded partial derivatives/, then this will serve as a stochastic Lyapunov function/, thereby establishing re/currence of the algorithm/. Though similar in /avor/, there are signi/cant di/erences between these results/. First/, in the present paper we consider a scaled o/.d/.e/./, not the usual o/.d/.e/. /(/2/)/. The former retains only terms with dominant growth and is frequently simpler/. Second/, while it is possible that the stability of the scaled o/.d/.e/. and the usual one go hand in hand/, this does not imply that a Lyapunov function for the latter is easily found/. The reinforce/ment learning algorithms for ergodic/-cost optimal control/, and asynchronous algorithms/, both considered as applications of the theory in this paper/, are examples where the scaled o/.d/.e/. is conveniently analyzed/. Though the assumptions made in this paper are explicitly motivated by applications to reinforcement learning algorithms for Markov decision processes/, this approach is likely to /nd a broader range of applications/. The paper is organized as follows/. The next section presents the main results for the stochastic approximation algorithm with vanishing stepsize or with bounded/, non/-vanishing stepsize/. Section /2 also gives a useful error bound for the constant stepsize case/, and brie/y sketches an extension to asynchronous algorithms/, omitting details that can be found in /[/6/]/. Section /3 gives examples of algorithms for reinforcement learning of Markov decision processes to which this analysis is applicable/. The proofs of the main results are collected together in Section /4/. /2 Main Results Here we collect together the main general results concerning the stochastic approximation algorithm/. Proofs not included here may be found in Section /4/.

We shall impose the following additional conditions on the functions fhr /: r / /1g de/ned in /(/3/)/, and the sequence M /= fM/(n/) /: n / /1g used in /(/1/)/. Some relaxations of the assumption /(A/1/) are discussed in Section /2/./4/. /3

x /2 I R

n / /0/:

- lim r/!/1 hr /(x/) /= h /1 /(x/)/; Furthermore/, the origin in I R d is an asymptotically stable equilibrium for the o/.d/.e/. /(/5/)/. /(A/2/) The sequence fM/(n/)/; F n /: n / /1g/, with Fn /= //(X/(i/)/; M/(i/)/; i / n/)/, is a martingale di/erence sequence/. Moreover/, for some C /0 /&lt; /1 and any initial condition X/(/0/) /2 I R d /, E/[kM/(n /+ /1/)k /2 j Fn/] / C /0 /(/1 /+ kX/(n/)k /2 /)/; n / /0/: The sequence fa/(n/)g is deterministic and is assumed to satisfy one of the following two assumptions/. Here TS stands for /`tapering stepsize/' and BS for /`bounded stepsize/'/. /(TS/) The sequence fa/(n/)g satis/es /0 /&lt; a/(n/) / /1/; n / /0/, and X n a/(n/) /= /1/; X n a/(n/) /2 /&lt; /1/: /(BS/) The sequence fa/(n/)g satis/es for some constants /1 /&gt; / /&gt; / /&gt; /0/,
- sup n kX/(n/)k /&lt; /1 a/:s/: /(ii/) Under /(BS/) there exists / / /&gt; /0 and C/1 /&lt; /1 such that for all /0 /&lt; / /&lt; / / and X/(/0/) /2 I R d /, lim sup n/!/1 E/[kX/(n/)k /2 /] / C /1 /: u t /4

d

/:

/ / a/(n/) / //; /2/./1 Stability and convergence The /rst result shows that the algorithm is stabilizing for both bounded and tapering step sizes/. Theorem /2/./1 Assume that /(A/1/)/, /(A/2/) hold/. Then/, /(i/) Under /(TS/)/, for any initial condition X/(/0/) /2 I R d /,

The proof is a standard application of the Hirsch lemma /(see /[/1/1/, Theorem /1/, pp/./3/3/9/]/, or /[/3/, /1/4/]/)/, but we give the details below for sake of completeness/. Theorem /2/./2 Suppose that /(A/1/)/, /(A/2/)/, /(TS/) hold and that the o/.d/.e/. /(/2/) has a unique globally asymptotically stable equilibrium x / /. Then X/(n/) /! x / a/.s/. as n /! /1 for any initial condition X/(/0/) /2 I R d /. Proof/: We may suppose that X/(/0/) is deterministic without any loss of generality so that

- /(b/) f b / /(t/)/; t /&gt; /0g is piecewise continuous/, de/ned so that/, for any j / /0/, b /  is the solution to /(/2/) for t /2 /[T /(j/)/; T /(j /+ /1/)/)/, with the initial condition b / /(T /(j/)/) /= / /(T /(j/)/)/. Let / /&gt; /0 and let B/(//) denote the open ball centered at x / of radius //. We may then choose /(i/) /0 /&lt; / /&lt; / such that x/(t/) /2 B/(//) for all t / /0 whenever x/( / /) is a solution of /(/2/) satisfying

x/(/0/) /2 B/(//)/. /(ii/) we have x/(t/) /2 B/(//=/2/) for all the conclusion of Theorem /2/./1 /(i/) holds that the sample paths of X are bounded with probability one/. Fixing such a sample path/, we see that X remains in a bounded set H/, which may chosen so that x / /2 int/(H/)/. The proof depends on an approximation of X with the solution to the primary o/.d/.e/. /(/2/)/. To perform this approximation/, /rst de/ne t/(n/) /" /1/, T/(n/) /" /1 as follows/: Set t/(/0/) /= T /(/0/) /= /0 and for n / /1/, t/(n/) /= Pn/?/1 i/=/0 a/(i/)/. Fix T /&gt; /0 and de/ne inductively T/(n /+ /1/) /= minft/(j/) /: t/(j/) /&gt; T /(n/) /+ T g/; n / /0/: Thus T/(n/) /= t/(m/(n/)/) for some m/(n/) /" /1/, and T / T /(n /+ /1/) /? T /(n/) / T /+ /1 for n / /0/. We then de/ne two functions from I R /+ to I R d /: /(a/) f/ /(t/)/; t /&gt; /0g is de/ned by / /(t/(n/)/) /= X/(n/) with linear interpolation on /[t/(n/)/; t/(n /+ /1/)/] for each n / /0/.

- T /&gt; /0 so large that for any solution of /(/2/) with x/(/0/) /2 H t / T /. Hence/, b / /(T /(j/)/?/) /2 B/(//=/2/) for all j / /1/. /5

limit k/ /(t/) /? b / /(t/)k /! /0/; a/:s/:/; t /! /1/: /(/6/) Hence we may choose j/0 /&gt; /0 so that we have k/ /(T /(j/)/?/) /? b / /(T /(j/)/?/)k / //=/2/; j / j /0 /: Since / /( / /) is continuous/, we conclude from /(ii/) and /(iii/) that / /(T /(j/)/) /2 B/(//) for j / j /0 /. Since b / /(T /(j/)/) /= / /(T /(j/)/) it then follows from /(i/) that b / /(t/) /2 B/(//) for all t / T /(j /0 /)/. Hence by /(/6/)/, lim sup t/!/1 k/ /(t/) /? x / k / //; a/:s/: This completes the proof since / /&gt; /0 was arbitrary/. u t We now consider /(BS/)/, focusing on the absolute error de/ned by e/(n/) /:/= kX/(n/) /? x / k/; n / /0/: /(/7/) Theorem /2/./3 Assume that /(A/1/)/, /(A/2/) and /(BS/) hold/, and suppose that /(/2/) has a globally asymptotically stable equilibrium point x / /. Then for any /0 /&lt; / / / / /, where / / is introduced in Theorem /2/./1 /(ii/)/, /(i/) For any / /&gt; /0/, there exists b /1 /= b /1 /(//) /&lt; /1 such that lim sup n/!/1 P/(e/(n/) / //) / b /1 //: /(ii/) If x / is a globally exponentially asymptotically stable equilibrium for the o/.d/.e/. /(/2/)/, then there exists b /2 /&lt; /1 such that for every initial condition X/(/0/) /2 I R d /, lim sup n/!/1 E/[e/(n/) /2 /] / b /2 //: u t /2/./2 Rate of convergence A uniform bound on the mean square error E/[e/(n/) /2 /] for n / /0 can be obtained under slightly stronger conditions on M via the theory of / /-irreducible Markov chains/. We /nd that this /6

/ /# /0/, while the second decays to zero exponentially as n /! /1/. To illustrate the nature of these bounds consider the linear recursion X/(n /+ /1/) /= X/(n/) /+ //[/?/(X/(n/) /? x / /) /+ W/(n /+ /1/)/]/; n / /0/; where fW/(n/)g is i/.i/.d/. with mean zero/, and variance / /2 /. This is of the form /(/1/) with h/(x/) /= /?/(x /? x / /) and M/(n/) /= W/(n/)/. The error e/(n /+ /1/) e/ned in /(/7/) may be bounded as follows/: E/[e/(n /+ /1/) /2 /] / / /2 / /2 /+ /(/1 /? //) /2 E/[e/(n/) /2 /] / // /2 /=/(/2 /? //) /+ exp/(/?/2/n/)E/[e/(/0/) /2 /]/; n / /0/: For a deterministic initial condition X/(/0/) /= x/, and any / /&gt; /0/, we thus arrive at the formal bound/, E/[e/(n/) /2 j X/(/0/) /= x/] / B /1 /(//) /+ B /2 /(kxk /2 /+ /1/) exp/(/?/ /0 /(//)n/) /(/8/) where B/1 /; B /2 and / /0 are positive/-valued functions of //. The bound /(/8/) is of the form that we seek/: the /rst term on the r/.h/.s/. decays to zero with //, while the second decays exponentially to zero with n/. However/, the rate of convergence for the second term becomes vanishingly small as / /# /0/. Hence to maintain a small probability of error the variable / should be neither too small/, nor too large/. This recalls the well known tradeo/ between mean and variance that must be made in the application of stochastic approximation algorithms/. A bound of this form carries over to the nonlinear model under some additional condi/tions/. For convenience we take a Markov model of the form X/(n /+ /1/) /= X/(n/) /+ //[h/(X/(n/)/) /+ m/(X/(n/)/; W /(n /+ /1/)/)/]/; /(/9/) where again fW/(n/)g is i/.i/.d/./, and also independent of the initial condition X/(/0/)/. We assume that the functions h /: I R d /! I R d and m /: I R d /- I R q /! I R d are smooth /(C /1 /)/, and that assumptions /(A/1/) and /(A/2/) continue to hold/. The recursion /(/9/) then describes a Feller Markov chain with stationary transition kernel to be denoted by P/. Let V /: I R d /! /[/1/; /1/) be given/. The Markov chain X with transition function P is called V /-uniformly ergodic if there is a unique invariant probability //, an R /&lt; /1/, and / /&lt; /1 such /7

n /;

x /2 I R

d

/; n / /0/;

/(/1/0/)

jE/[g/(X/(n/)/) j X/(/0/) /= x/] /? E / /[g/(X/(n/)/)/]j / RV /(x/)/ where E/ /[g/(X/(n/)/)/] /= R g/(x/) //(dx/)/, n / /0/. The following result establishes bounds of the form /(/8/) using V /-ergodicity of the model/. Assumptions /(/1/1/) and /(/1/2/) below are required to establish / /-irreducibility of the model in Lemma /4/./1/0/: There exists a w / /2 I R q with m/(x / /; w / /) /= /0/, and for a continuous function q

/8

- p /: I R /! /[/0/; /1/] with p/(w / /) /&gt; /0/, P/(W/(/1/) /2 A/) / Z A p/(z/)dz/; A /2 B/(I R q /)/; /(/1/1/) The pair of matrices /(F/; G/) is controllable with F /= d dx h/(x / /) /+ /@ /@x m/(x / /; w / /) and G /= /@ /@w m/(x / /; w / /)/; /(/1/2/) Theorem /2/./4 Suppose that /(A/1/)/, /(A/2/)/, /(/1/1/)/, and /(/1/2/) hold for the Markov model /(/9/) with /0 /&lt; / / / / /. Then the Markov chain X is V /-uniformly ergodic/, with V /(x/) /= kxk /2 /+ /1/, and we have the following bounds/: /(i/) There exist positive/-valued functions A /1 and / /0 of //, and a constant A/2 independent of //, such that Pfe/(n/) / / j X/(/0/) /= xg / A /1 /(//) /+ A /2 /(kxk /2 /+ /1/) exp/(/?/ /0 /(//)n/)/: The functions satisfy A /1 /(//) /! /0/, / /0 /(//) /! /0 as / /# /0/. /(ii/) If in addition the o/.d/.e/. /(/2/) is exponentially asymptotically stable/, then the stronger bound /(/8/) holds/, where again B/1 /(//) /! /0/, / /0 /(//) /! /0 as / /# /0/, and B/2 is independent of //. Proof/: The V /-uniform ergodicity is established in Lemma /4/./1/0/. From Theorem /2/./3 /(i/) we have when X/(/0/) / / P/ /(e/(n/) / //) /= P / /(e/(/0/) / //) / b /1 //;

u

t

We assume that / kk /(n/) /= /0 for all n/, and that f/ kj /(n/)g have a common upper bound / /&lt; /1 /(/[/6/] considers a slightly more general situation/./) To relate the present work to /[/6/]/, we recall that the /`centralized/' algorithm of /[/6/] is X/(n /+ /1/) /= X/(n/) /+ a/(n/)f/(X/(n/)/; W /(n /+ /1/)/)

P/(e/(n/) / / j X/(/0/) /= x/) / P/ /(e/(n/) / //) /+ jP/(e/(n/) / / j X/(/0/) /= x/) /? P / /(e/(n/) / //)j / b /1 / /+ RV /(x/)/ n /; n / /0/: This and the de/nition of V establishes /(i/)/. The proof of /(ii/) is similar/. The fact that / /= / / /! /1 as / /# /0 is discussed in Section /4/./3/. /2/./3 The asynchronous case The conclusions above also extend to the model of asynchronous stochastic approximation analysed in /[/6/]/. We now assume that each component of X/(n/) is updated by a separate processor/. We postulate a set/-valued process fY /(n/)g taking values in the set of subsets of f/1/; /2/; / / / /; dg/, with the interpretation/: Y /(n/) /= f indices of the components updated at time ng/. For n / /0/; /1 / i / d/, de/ne //(i/; n/) /= n X m/=/0 Ifi /2 Y /(m/)g/; the number of updates executed by the i/-th processor up to time n/. A key assumption is that there exists a deterministic / /&gt; /0 such that for all i/, lim inf n/!/1 //(i/; n/) n / / a/:s/: This ensures that all components are updated comparably often/. At time n/, the kth processor has available the following data/: /(i/) Processor /(k/) is given //(k/; n/)/, but it may not have n/, the /`global clock/'/. /(ii/) There are interprocessor communication delays / kj /(n/)/; /1 / k/; j / d/; n / /0/, so that at time n/, processor /(k/) may use the data X j /(m/) only for m / n /? / kj /(n/)/.

/9

the present set/-up is obtained by setting h/(x/) /= F /(x/) and M/(n /+ /1/) /= f/(X/(n/)/; W /(n /+ /1/)/) /? F /(X/(n/)/) for n / /0/. The asynchronous version then is Xi/(n /+ /1/) /= Xi/(n/) /+ a/(//(i/; n/)/)f /(X /1 /(n /? / i/1 /(n/)/)/; X /2 /(n /? / i/2 /(n/)/)/; / / / /(/1/3/) / / / /; Xd/(n /? / id /(n/)/)/; W /(n /+ /1/)/)Ifi /2 Y /(n/)g/; n / /0/;

singletons without any loss of generality/. We shall do likewise/. What this entails is simply unfolding a single update at time n into jY /(n/)j separate updates/, each involving a single component/. This blows up the delays at most d/-fold/, which does not a/ect the analysis in any way/. The main result of /[/6/] is the analog of our Theorem /2/./2 given that the conclusions of our Theorem /2/./1 hold/. In other words/, stability implies convergence/. Under /(A/1/) and /(A/2/)/, our arguments above can be easily adapted to show that the conclusions of Theorem /2/./2 also for /1 / i / d/. Note that this can be executed by the i/-th processor without any knowledge of the global clock which/, in fact/, can be a complete arti/ce as long as causal relationships are respected/. The analysis presented in /[/6/] depends upon the following additional conditions on fa/(n/)g/: /(i/) a/(n /+ /1/) / a/(n/) eventually/; /(ii/) For x /2 /(/0/; /1/)/, sup n a/(/[xn/]/)/=a/(n/) /&lt; /1/; /(iii/) For x /2 /(/0/; /1/)/; / /[xn/] X i/=/0 a/(i/) //./ n X i/=/0 a/(i/) / /! /1/; where /[ / /] stands for /`the integer part of /( / /)/'/. A fourth condition is imposed in /[/6/]/, but this becomes irrelevant when the delays are bounded/. Examples of fa/(n/)g satisfying the /(i/)/{/(iii/) are a/(n/) /= /1/=/(n /+ /1/)/, or /1/=/(/1 /+ nlog/(n /+ /1/)/)/. As a /rst simplifying step/, it is observed in /[/6/] that fY /(n/)g may be assumed to be

/1/0

the suitably interpolated and rescaled trajectory of the algorithm tracks an appropriate o/.d/.e/./. The only di/erence is a scalar factor /1/=d multiplying the r/.h/.s/. of the o/.d/.e/. /(i/.e/./, /\_ x/(t/) /= /1 d h/(x/(t/)/)/)/. This factor/, which re/ects the asynchronous sampling/, amounts to a time scaling that does not a/ect the qualitative behavior of the o/.d/.e/. Theorem /2/./5 Under the conditions of Theorem /2/./2 and the above hypotheses on fa/(n/)g/, fY /(n/)g and f/ ij /(n/)g/, the asynchronous iterates given by /(/2/0/) remain a/.s/. bounded and /(therefore/) converge to x / a/.s/. u t

/2/./4 Further extensions Although satis/ed in all of the applications treated in Section /3/, in some other models the to the o/.d/.e/. /(/5/) with h /1 /2 //, j/ /1 /(t/)j / be /?/t j/ /1 /(/0/)j/; t / /0/: /1/1

assumption /(A/1/) that hr /! h/1 pointwise may be violated/. If this convergence does not hold then we may abandon the /uid model and replace /(A/1/) by /(A/1/'/) The function h is Lipschitz/, and there exists T /&gt; /0/, R /&gt; /0 such that j b //(t/)j / /1 /2 /; t / T/; for any solution to /(/4/) with r / R/, and with initial condition satisfying j b //(/0/)j / /1/. Under the Lipschitz condition on h/, at worst we may /nd that the pointwise limits of fhr /: r / /1g will form a family / of Lipschitz functions on I R d /. That is/, h /1 /2 / if and only if there exists a sequence fr i g /" /1 such that hri /(x/) /! h /1 /(x/)/; i /! /1/; where the convergence is uniform for x in compact subsets of I R d /. Under /(A/1/'/) we then /nd/, using the same arguments as in the proof of Lemma /4/./1/, that the family / is uniformly stable/: Lemma /2/./6 Under /(A/1/'/) the family of o/.d/.e/.s de/ned via / is uniformly exponentially asymptotically stable in the following sense/: For some b /&lt; /1/, / /&gt; /0/, and any solution / /1

Using this lemma the development of Section /4 goes through with virtually no changes/, and hence Theorems /2/./1/{/2/./5 are valid with /(A/1/) replaced by /(A/1/'/)/.

section we analyse reinforcement learning algorithms for Markov decision processes/. The reader is referred to /[/4/] for a general background of the subject and to other references listed below for further details/. /3/./1 Markov decision processes We consider a Markov decision process / /= f//(t/) /: t /2 Z Z /+ g taking values in a /nite state

Another extension is to broaden the class of scalings/. Consider a nonlinear scaling de/ned by a function g/: I R /+ /! I R /+ satisfying g/(r/) /! /1 as r /! /1/, and suppose that h r /( / /) rede/ned as h r /(x/) /= h/(rx/)/=g/(r/) satis/es hr /(x/) /! h /1 /(x/) uniformly on compacts as r /! /1/. Then a completely analogous development of the stochastic gradient algorithm is possible/. An example would be a /`stochastic gradient/' scheme where h/( / /) is the gradient of an even degree polynomial/, with degree/, say/, /2n/. Then g/(r/) /= r /2n/?/1 will do/. We do not pursue this further because the reinforcement learning algorithms we consider below do conform to the case g/(r/) /= r/. /3 Reinforcement learning As both an illustration of the theory and an important application in its own right/, in this

/1/2

space S /= f/1/; /2/; / / / /; sg and controlled by a control sequence Z /= fZ/(t/) /: t /2 Z Z /+ g taking values in a /nite action space A /= fa /0 /; / / / /; a r g/. We assume that the control sequence is admissible in the sense that Z/(n/) /2 /f//(t/) /: t / ng for each n/. We are most interested in stationary policies of the form Z/(t/) /= w/(//(t/)/)/, where the feedback law w is a function w/: S /! A/. The controlled transition probabilities are given by p/(i/; j/; a/) for i/; j /2 S/; a /2 A/. Let c /: S /- A /! R be the one/-step cost function/, and consider /rst the in/nite horizon discounted cost control problem of minimizing over all admissible Z the total discounted

Q/(i/; a/) /= c/(i/; a/) /+ / X

j

The minimal value function is de/ned as p/(i/; j/; a/) min

b

Q/(j/; b/)/;

/;

i /2 S/; a /2 A/:

i /2 S/; a /2 A/; n / /0/;

/1/3

p/(i/; j/; a/) min

b

Qn/(j/; b/)/;

Qn/+/1 /(i/; a/) /= c/(i/; a/) /+ / X

where Q/0 / /0 is arbitrary/.

j

J/(i/; Z /) /= E t/=/0 / c/(//(t/)/; Z/(t/)/) j //(/0/) /= i where / /2 /(/0/; /1/) is the discount factor/. V /(i/) /= minJ/(i/; Z /)/; where the minimum is over all admissible control sequences Z/. The function V satis/es the dynamic programming equation V /(i/) /= min a h c/(i/; a/) /+ / X j p/(i/; j/; a/)V /(j/) i /; i /2 S/; and the optimal control minimizing J is given as the stationary policy de/ned through the feedback law w / given as any solution to w / /(i/) /:/= arg min a h c/(i/; a/) /+ / X j p/(i/; j/; a/)V /(j/) i /; i /2 S/: The value iteration algorithm is an iterative procedure to compute the minimal value function/. Given an initial function V /0 /: S /! I R /+ one obtains a sequence of functions fV n g through the recursion Vn/+/1 /(i/) /= min a h c/(i/; a/) /+ / X j p/(i/; j/; a/)V n /(j/) i /; i /2 S/; n / /0/: /(/1/4/) This recursion is convergent for any initialization V /0 / /0/. If we de/ne Q/-values via Q/(i/; a/) /= c/(i/; a/) /+ / X j p/(i/; j/; a/)V /(j/)/; i /2 S/; a /2 A/; then V /(i/) /= min a Q/(i/; a/) and the matrix Q satis/es The matrix Q can also be computed using the equivalent formulation of value iteration/,

/(/1/5/)

trast/, the policy iteration algorithm is initialized with a feedback law w /, and gener/ates a sequence of feedback laws fw n /: n / /0g/. At the nth stage of the algorithm a feedback law w n is given/, and the value function for the resulting control sequence Z n /= fw n /(//(/0/)/)/; w n /(//(/1/)/)/; w n /(//(/2/)/)/; /: /: /:g is computed to give Jn/(i/) /= J/(i/; Z n /)/; i /2 S/: Interpreted as a column vector in I R s /, the vector J n satis/es the equation /(I /? /P n /)J n /= c n /(/1/6/) where the s /- s matrix P n is de/ned by P n /(i/; j/) /= p/(i/; j/; w n /(i/)/)/, i/; j /2 S/, and the column vector c n is given by c n /(i/) /= c/(i/; w n /(i/)/)/, i /2 S/. The equation /(/1/6/) can be solved for /xed n by the /`/xed/-policy/' version of value iteration given by Jn/(i /+ /1/) /= /P n J n /(i/) /+ c n /; i / /0/; /(/1/7/) where Jn/(/0/) /2 I R s is given as an initial condition/. Then Jn/(i/) /! J n /, the solution to /(/1/6/)/, at a geometric rate as i /! /1/. Given Jn /, the next feedback law w n/+/1 is then computed via w n/+/1 /(i/) /= min a h c/(i/; a/) /+ / X j p/(i/; j/; a/)J n /(j/) i /; i /2 S/: /(/1/8/) Each step of the policy iteration algorithm is computationally intensive for large state spaces since the computation of J n requires the inversion of the s /- s matrix I /? /P n /. In the average cost optimization problem one seeks to minimize over all admissible Z/, lim sup n/!/1 /1 n n/?/1 X t/=/0 E/[c/(//(t/)/; Z/(t/)/)/]/: /(/1/9/)

The policy iteration and value iteration algorithms to solve this optimization problem re/main unchanged with three exceptions/. One is that the constant / must be set equal to unity in equations /(/1/4/) and /(/1/8/)/. Secondly/, in the policy iteration algorithm the value function J n is replaced by a solution J n to Poisson/'s equation X p/(i/; j/; w n /(i/)/)J n /(j/) /= J n /(i/) /? c/(i/; w n /(i/)/) /+ / n /; i /2 S/; /1/4

involves matrix inversions via / n /(I /? P n /+ ee /0 /) /= e /0 /; / n /= / n c n /; /(I /? P n /+ ee /0 /)J n /= c n /; where e /2 I R s is the column vector consisting of all ones/, and the row vector / n is the invariant probability for P n /. The introduction of the outer product ensures that the matrix /(I /? P n /+ ee /0 /) is invertible/, provided that the invariant probability / n is unique/. Lastly/, the value iteration algorithm is replaced by the /`relative value iteration/' where a common scalar o/set is subtracted from all components of the iterates at each iteration /(likewise for the Q/-value iteration/)/. The choice of this o/set term is not unique/. We shall be considering one particular choice/, though others can be handled similarly /(see /[/1/]/)/. /3/./2 Q/-learning If the matrix Q de/ned in /(/1/5/) can be computed via value iteration or some other scheme

De/ne F/(Q/) /= /[F ia /(Q/)/] i/;a by

<!-- formula-not-decoded -->

/1/5

then the optimal control is found through a simple minimization/. If transition probabilities are unknown so that value iteration is not directly applicable/, one may apply a stochastic approximation variant known as the Q/-learning algorithm of Watkins /[/1/, /2/0/, /2/1/]/. This is de/ned through the recursion Qn/+/1 /(i/; a/) /= Q n /(i/; a/) /+ a/(n/) h / min b Qn/(/	 n/+/1 /(i/; a/)/; b/) /+ c/(i/; a/) /? Q n /(i/; a/) i /; i /2 S/; a /2 A/; where /	 n/+/1 /(i/; a/) is an independently simulated S/-valued random variable with law p/(i/; //; a/)/. Making the appropriate correspondences with our set/-up/, we have X/(n/) /= Qn and h/(Q/) /= /[h ia /(Q/)/] i/;a with hia /(Q/) /= / X j p/(i/; j/; a/) min b Q/(j/; b/) /+ c/(i/; a/) /? Q/(i/; a/)/; i /2 S/; a /2 A/: The martingale is given by M/(n /+ /1/) /= /[M ia /(n /+ /1/)/] i/;a with Mia/(n /+ /1/) /= / / min Qn/(/	 n/+/1 /(i/; a/)/; b/) /? X p/(i/; j/; a/) / min Qn/(j/; b/) // /; i /2 S/; a /2 A/:

/\_ Q /= F/(Q/) /? Q /:/= h/(Q/)/: /(/2/0/) The map F /: I R s/-/(r/+/1/) /! I R s/-/(r/+/1/) is a contraction w/.r/.t/. the max norm k / k /1 /. The global asymptotic stability of its unique equilibrium point is a special case of the results of /[/8/]/. This h/( / /) /ts the framework of our analysis/, with the /(i/; a/)/-th component of h /1 /(Q/) given by / X j p/(i/; j/; a/) min b Q/(j/; b/) /? Q/(i/; a/)/; i /2 S/; a /2 A/: This also is of the form h /1 /(Q/) /= F /1 /(Q/) /? Q where F/1/( / /) is an k/:k /1 /- contraction/, and thus the asymptotic stability of the unique equilibrium point of the corresponding o/.d/.e/. is guaranteed /(see /[/8/]/)/. We conclude that assumptions /(A/1/) and /(A/2/) hold/, and hence also Theorems /2/./1/{/2/./4 hold for the Q/-learning model/. /3/./3 Adaptive critic algorithm Next we shall consider the adaptive critic algorithm/, which may be considered as the rein/- forcement learning analog of policy iteration /(see /[/2/, /1/3/] for a discussion/)/. There are several variants of this/, one of which/, taken from /[/1/3/]/, is as follows/: For i /2 S/, Vn/+/1 /(i/) /= Vn/(i/) /+ b/(n/)/[c/(i/; /  n /(i/)/) /+ /V n /(/	 n /(i/; /  n /(i/)/)/) /? V n /(i/)/]/; /(/2/1/) b wn/+/1 /(i/) /= /? n b wn/(i/) /+ a/(n/) r X /`/=/1 / /[c/(i/; a /0 /) /+ /V n /(/ n /(i/; a /0 /)/)/] /? /[c/(i/; a /` /) /+ /V n /(/ n /(i/; a /` /)/)/]e /` /o /: /(/2/2/) Here fVng are s/-vectors and for each i/; f b wn/(i/)g are r/-vectors lying in the simplex fx /2 I R r j x /= /[x /1 /; / / / /; x r /]/; x i / /0/; P i x i / /1g/. /?/( / /) is the projection onto this simplex/. The sequences fa/(n/)g/; fb/(n/)g satisfy X n a/(n/) /= X n b/(n/) /= /1/; X n /(a/(n/) /2 /+ b/(n/) /2 /) /&lt; /1/; a/(n/) /= o/(b/(n/)/)/: The rest of the notation is as follows/: For /1 / /` / r/; e /` is the unit r/-vector in the /`/-th coordinate direction/. For each i/, n/, w n /(i/) /= w n /(i/; /:/) is a probability vector on A de/ned by/: For b wn/(i/) /= /[ b wn/(i/; /1/)/; /:/:/:/; b wn/(i/; r/)/]/, wn/(i/; a /` /) /= / b wn/(i/; /`/) for /` /6/= /0/; /1 /? P j/6/=/0 b wn/(i/; j/) for /` /= /0/. /1/6

Likewise/, /	 n /(i/; /  n /(i/)/) are S/-valued random variables which are independently simulated /(given /  n /(i/)/) with law p/(i/; /:/; /  n /(i/)/) and f/ n /(i/; a /` /)g are S/-valued random variables indepen/dently simulated with law p/(i/; /:/; a /` /) respectively/. To see why this is based on policy iteration/, recall that policy iteration alternates be/tween two steps/: One step solves the linear system of equations /(/1/6/) to compute the /xed/- the second as nearly static/, thus justifying viewing it as a /xed/-policy iteration/. In turn/, the second sees the /rst as almost equilibrated/, justifying the search sheme for minimization over A/. See /[/1/3/] for details/. The boundedness of f b wng is guaranteed by the projection /?/( / /)/. For fVng/, the fact that b/(n/) /= o/(a/(n/)/) allows one to treat b wn/(i/) as constant/, say w/(i/) /{ see/, e/.g/./, /[/1/3/]/. The appropriate o/.d/.e/. then turns out to be

policy value function corresponding to the current policy/. We have seen that solving /(/1/6/) can be accomplished by performing the /xed/-policy version of value iteration given in /(/1/7/)/. The /rst step /(/2/1/) in the above iteration is indeed the /`learning/' or /`simulation/-based stochastic approximation/' analog of this /xed/-policy value iteration/. The second step in policy iter/ation updates the current policy by performing an appropriate minimization/. The second iteration /(/2/2/) is a particular search algorithm for computing this minimum over the simplex of probability measures on A/. This search algorithm is by no means unique/: The paper /[/1/3/] gives two alternative schemes/. However/, the /rst iteration /(/2/1/) is common to all/. The di/erent choices of stepsize schedules for the two iterations /(/2/1/)/, /(/2/2/) induces the /`two time/-scale/' e/ect discussed in /[/5/]/. Thus the /rst iteration sees the policy computed by

/\_ v /= G/(v/) /? v /:/= h/(v/) /(/2/3/) where G /: I R s /! I R s is de/ned by/: Gi/(x/) /= X /` w/(i/; a /` /) h / X j p/(i/; j/; a /` /)x j /+ c/(i/; a /` /) i /? x i /; i /2 S/: Once again/, G/( / /) is an k / k/1 /-contraction and it follows from the results of /[/8/] that /(/2/3/) is globally asymptotically stable/. The limiting function h /1 /(x/) is again of the form /1/7

p/(i/; j/; a /` /)x j

/? x i /:

discussed/, di/ering only in the f b wng iteration/. The iteration for fV n g is common to all and is given by Vn/+/1 /(i/) /= V n /(i/) /+ b/(n/)/[c/(i/; /  n /(i/)/) /+ V n /(/	 n /(i/; /  n /; /(i/)/)/) /? V n /(i/) /? V n /(i /0 /)/]/; i /2 S where i/0 /2 S is a /xed state prescribed beforehand/. This leads to the o/.d/.e/. /(/2/3/) with G rede/ned as Gi/(x/) /= X i /2 S/:

<!-- formula-not-decoded -->

X /` w/(i/; a /` /) / X j We see that G/1 is also a k / k/1 /- contraction and the global asymptoyic stability of the origin for the corresponding limiting o/.d/.e/. follows as before from the results of /[/8/]/. /3/./4 Average cost optimal control For the average cost control problem we impose the additional restriction that the chain / has a unique invariant probability measure under any stationary policy so that the steady state cost /(/1/9/) is independent of the initial condition/. For the average cost optimal control problem the Q/-learning algorithm is given by the recursion Qn/+/1 /(i/; a/) /= Q n /(i/; a/) /+ a/(n/) / min b Qn/(/	 n /(i/; a/)/; b/) /+ c/(i/; a/) /? Q n /(i/; a/) /? Q n /(i /0 /; a /0 /) / /; where i/0 /2 S/, a/0 /2 A are /xed a priori/. The appropriate o/.d/.e/. now is /(/2/0/) with F/( / /) rede/ned as F ia /(Q/) /= P j p/(i/; j/; a/) min b Q/(j/; b/) /+ c/(i/; a/) /? Q/(i/; a/) /? Q/(i /0 /; a /0 /)/. The global asymptotic stability for the unique equilibrium point for this o/.d/.e/. has been established in /[/1/]/. Once again this /ts our framework with h /1 /(x/) /= F /1 /(x/) /? x for F /1 de/ned the same way as F/, except for the terms c/(//; //) which are dropped/. We conclude that /(A/1/) and /(A/2/) are satis/ed for this version of the Q/-learning algorithm/. Another variant of Q/-learning for average cost/, based on a /`stochastic shortest path/' formulation/, is presented in /[/1/]/. This also can be handled similarly/. In /[/1/3/]/, three variants of the adaptive critic algorithm for the average cost problem are established in /[/7/]/. Once more/, this /ts our framework with h /1 /(x/) /= G /1 /(x/) /? x for G/1 de/ned just like G/, but without the c/(//; //) terms/. Asynchronous versions of all the above can be written down along the lines of /(/2/0/)/. Then

by Theorem /2/./5/, they have bounded iterates a/.s/. The important point to note here is that to date/, a/.s/. boundedness for Q/-learning and adaptive critic is proved by other methods for centralized algorithms /[/1/, /1/2/, /2/0/]/. For asynchronous algorithms/, it is proved for discounted cost only /[/1/, /1/3/, /2/0/]/, or by introducing a projection to enforce stability /[/1/4/]/. /4 Derivations Here we provide proofs for the main results given in Section /2/. Throughout this section we assume that /(A/1/) and /(A/2/) hold/. /4/./1 Stability The functions fh r /; r / /1g and the limiting function h /1 are Lipschitz with the same Lipschitz constant as h under /(A/1/)/. It follows from Ascoli/'s Theorem that the convergence h r /! h /1 is uniform on compact subsets of I R d /. This observation is the basis of the following lemma/. Lemma /4/./1 Under /(A/1/)/, the o/.d/.e/. /(/5/) is globally exponentially asymptotically stable/. Proof/: The function h/1 satis/es h/1 /(cx/) /= ch /1 /(x/)/; c /&gt; /0/; x /2 I R d /: Hence the origin / /2 I R d is an equilibrium for /(/5/)/, i/.e/./, h/1 /(//) /= //. Let B/(//) be the closed ball of radius / centered at / with / chosen so that x/(t/) /! / as t /! /1 uniformly for initial conditions in B/(//)/. Thus there exists a T /&gt; /0 such that kx/(T/)k / //=/2 whenever kx/(/0/)k / //. For an arbitrary solution x/( / /) of /(/5/)/, y/( / /) /= /x/( / /)/=kx/(/0/)k is another/, with ky/(/0/)k /= //. Hence ky/(T/)k /&lt; //=/2/, implying kx/(T/)k / /1 /2 kx/(/0/)k/. The global exponential asymptotic stability follows/. u t With the scaling parameter r given by r/(j/) /= max/(/1/; kX/(m/(j/)/)k/)/, j / /0/, we de/ne three piecewise continuous functions from I R /+ to I R d as in the introduction/: /1/9

a function / j on the interval /[T /(j/)/; T /(j /+ /1/)/] by / j /(t/(n/)/) /= X/(n/)/=r/(j/)/; m/(j/) / n / m/(j /+ /1/)/; with / j /( / /) de/ned by linear interpolation on the remainder of /[T /(j/)/; T /(j /+ /1/)/] to form a piecewise linear function/. We then de/ne / to be the piecewise continuous function //(t/) /= / j /(t/)/; t /2 /[T /(j/)/; T /(j /+ /1/)/)/; j / /0/: /(b/) f b //(t/) /: t / /0g is continuous on each interval /[T /(j/)/; T /(j /+ /1/)/)/, and on this interval it is the solution to the o/.d/.e/. /\_ x/(t/) /= h r/(j/) /(x/(t/)/)/; /(/2/4/) with initial condition b //(T /(j/)/) /= //(T /(j/)/)/, j / /0/. /(c/) f/ /1 /(t/) /: t / /0g is also continuous on each interval /[T /(j/)/; T /(j /+ /1/)/)/, and on this interval it is the solution to the /\/uid model/" /(/5/) with the same initial condition / /1 /(T /(j/)/) /= b //(T /(j/)/) /= //(T /(j/)/) j / /0/: Boundedness of b //( / /) and / /1 /( / /) is crucial in deriving useful approximations/. Lemma /4/./2 Under /(A/1/) and /(A/2/)/, and either /(TS/) or /(BS/)/, there exists / C /&lt; /1 such that for any initial condition X/(/0/) /2 I R d b //(t/) / / C and / /1 /(t/) / / C/; t / /0/: Proof/: To establish the /rst bound use the Lipschitz continuity of h to obtain the bound d dt k b //(t/)k /2 /= /2 b //(t/) T h r/(j/) /( b //(t/)/) / C/(k b //(t/)k /2 /+ /1/)/; T/(j/) / t /&lt; T /(j /+ /1/)/; where C is a deterministic constant/, independent of j/. The claim follows with / C /= /2 exp/(/(T /+ /1/)C/) since k b //(T /(j/)/)k / /1/. The proof of the second bound is identical/. u t The following version of the Bellman Gronwall Lemma will be used repeatedly/. /2/0

Then for all n / /1/,

A/(n /+ /1/) / / /+ X

Then for all n / /1/, where //(n/) /=

/(ii/)

Suppose f//(n/)g/, fA/(n/)g/, f//(n/)g are nonnegative sequences such that

Proof/:

<!-- formula-not-decoded -->

De/ne fR/(n/)g inductively by R/(/0/) /= A/(/0/) and

/:

<!-- formula-not-decoded -->

n / /0/:

and / /1 /( / /)/.

k/=/0 A simple induction shows that A/(n/) / R/(n/)/, n / /0/. An alternative expression for R/(n/) is is R/(n/) /= / n Y k/=/1 /(/1 /+ //(k/) // //(/0/)A/(/0/) /+ / / The inequality /(i/) then follows from the bound /1 /+ x / e x /. To see /(ii/) /x n / /0 and observe that on summing both sides of the bound A/(k /+ /1/) /? A/(k/) / //(k/)A/(k/) /+ //(k/)/; over /0 / k / /` we obtain for all /0 / /` /&lt; n/, A/(/` /+ /1/) / A/(/0/) /+ //(n/) /+ /` X k/=/0 //(k/)A/(k/)/: The result then follows from /(i/)/. The following lemmas relate the three functions //( / /)/, b //( / /)/, /2/1

Pn

/0

u

t

such that for any r /&gt; R and any solution to the o/.d/.e/. /(/4/) satisfying kx/(/0/)k / /1/, we have kx/(t/)k / / for t /2 /[T/; T /+ /1/]/. Proof/: By global asymptotic stability of /(/5/) we can /nd T /&gt; /0 such that k/ /1 /(t/)k / //=/2/, t / T /, for solutions / /1 /( / /) of /(/5/) satisfying k/ /1 /(/0/)k / /1/. With T /xed/, choose R so large that j b //(t/) /? / /1 /(t/)j / //=/2 whenever b / is a solution to /(/4/)

satisfying b //(/0/) /= / /1 /(/0/)/; j b //(/0/)j / /1/; and r / R/. This is possible since/, as we have already observed/, h r /! h /1 as r /! /1 uniformly on compact sets/. The claim then follows from the triangle inequality/. u t De/ne the following/: For j / /0/, m/(j/) / n /&lt; m/(j /+ /1/)/, e X/(n/) /:/= X/(n/)/=r/(j/) f M/(n /+ /1/) /:/= M/(n /+ /1/)/=r/(j/) and for n / /1/, //(n/) /:/= n/?/1 X m/=/0 a/(m/) f M/(m /+ /1/)/: Under /(A/1/)/, /(A/2/) and either /(TS/) or /(BS/)/, for each initial condition X/(/0/) /2

/2

Lemma /4/./5 I R d satisfying E/[kX/(/0/)k /2 /] /&lt; /1/, we have /(i/) sup n//0 E/[k e X/(n/)k /2 /] /&lt; /1/, /(ii/) sup j//0 E/[kX/(m/(j /+ /1/)/)/=r/(j/)k /2 /] /&lt; /1/, /(iii/) sup j//0/;T /(j/)/t/T /(j/+/1/) E/[k//(t/)k /2 /] /&lt; /1/, /(iv/) Under /(TS/) the sequence f//(n/)/; F n g is a square integrable martingale with sup n//0 E/[k//(n/)k /] /&lt; /1/: Proof/: To prove /(i/) note /rst that under /(A/2/) and the Lipschitz condition on h there exists C /&lt; /1 such that for all n / /1/, E/[kX/(n/)k /2 j Fn/?/1 /] / /(/1 /+ Ca/(n /? /1/)/)kX/(n /? /1/)k /2 /+ Ca/(n /? /1/)/; n / /0/: /(/2/5/) /2/2

E/[k e

X/(n/)k

/2

j Fn/?/1 /] / /(/1 /+ Ca/(n /? /1/)/)k e

X/(n /? /1/)k

/2

/+ Ca/(n /? /1/)/;

so that by Lemma /4/./3 /(ii/)/, for all such n/,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Claim /(i/) follows/, and claim /(ii/) follows similarly/. We then obtain /(iii/) from the de/nition of //( / /)/. From /(i/)/, /(ii/) and /(A/2/)/, we have sup n E/[k f M/(n/)k /2 /] /&lt; /1/. Using this and the square summability of fa/(n/)g assumed in /(TS/)/, the bound /(iv/) immediately follows/. u t Lemma /4/./6 Suppose E/[kX/(/0/)k /2 /] /&lt; /1/. Under /(A/1/)/, /(A/2/) and /(TS/)/, with probability one/, /(i/) k//(t/) /? b //(t/)k /! /0 as t /! /1/, /(ii/) sup t//0 k//(t/)k /&lt; /1/. Proof/: Express

<!-- formula-not-decoded -->

where / /1 /(j/) /= O t/(m/(j /+ /1/)/) We also have by de/nition //(t/(n /+ /1/)/?/) /= //(T /(j/)/) /+ n X i/=m/(j/) a/(i/)/[h r/(j/) /(//(t/(i/)/)/) /+ f M/(i /+ /1/)/]/: /(/2/7/) For m/(j/) / n / m/(j /+ /1/) let /"/(n/) /= k//(t/(n/)/?/) /? b //(t/(n/)/?/)k/. Combining /(/2/6/)/, /(/2/7/)/, and the Lipschitz continuity of h/, we have /"/(n /+ /1/) / /"/(m/(j/)/) /+ / /1 /(j/) /+ k//(n /+ /1/) /? //(m/(j/)/)k /+ C n X i/=m/(j/) a/(i/)/"/(i/)/;

/"/(n/)

/ exp/(C/(T /+ /1/)/)/(/ /1 /(j/) /+ / /2 /(j/)/)/;

/2/4

m/(j/) / n / m/(j /+ /1/)/;

where / /2 /(j/) /= maxm/(j/)/n/m/(j/+/1/) k//(n /+ /1/) /? //(m/(j/)/)k/. By /(iv/) of Lemma /4/./5 and the martingale convergence theorem /[/1/8/, p/. /6/2/]/, f//(n/)g converges a/.s/./, thus / /2 /(j/) /! /0 a/.s/. as j /! /1/. Since / /1 /(j/) /! /0 as well/, sup m/(j/)/n/m/(j/+/1/) k//(t/(n/)/?/) /? b //(t/(n/)/?/)k /= sup m/(j/)/n/m/(j/+/1/) /"/(n/) /! /0 as j /! /1/, which implies the /rst claim/. Result /(ii/) then follows from Lemma /4/./2 and the triangle inequality/. u t Lemma /4/./7 Under /(A/1/)/, /(A/2/) and /(BS/)/, these exists a constant C /2 /&lt; /1 such that for all j / /0/, /(i/) sup j//0/;T /(j/)/t/T /(j/+/1/) E/[k//(t/) /? b //(t/)k /2 j Fn/(j/) /] / C /2 //, /(ii/) sup j//0/;T /(j/)/t/T /(j/+/1/) E/[k//(t/)k /2 j F n/(j/) /] / C /2 /. Proof/: Mimic the proof of Lemma /4/./6 to obtain /"/(n /+ /1/) / n X i/=m/(j/) Ca/(i/)/"/(i/) /+ / /0 /(j/)/; m/(j/) / n /&lt; m/(j /+ /1/) where /"/(n/) /= E/[k//(t/(n/)/?/) /? b //(t/(n/)/?/)k /2 j F m/(j/) /] /1/=/2 for m/(j/) / n / m/(j /+ /1/)/, and the error term has the upper bound j/ /0 /(j/)j /= O/(//)/; where the bound is deterministic/. By Lemma /4/./3 /(i/) we obtain the bound/, /"/(n/) / exp/(C/(T /+ /1/)/)/ /0 /(j/)/; m/(j/) / n / m/(j /+ /1/)/; which proves /(i/)/. We then obtain /(ii/) using Lemma /4/./2/, /(i/)/, and the triangle inequality/.

u

t

deterministic without any loss of generality/. In particular/, E/[kX/(/0/)k /] /&lt; /1 trivially/. By Lemma /4/./6 /(ii/)/, it now su/ces to prove that sup n kX/(m/(n/)/)k /&lt; /1 a/.s/. Fix a sample point outside the zero probability set where Lemma /4/./6 fails/. Pick T /&gt; /0 as above and R /&gt; /0 such that for every solution x/( / /) of the o/.d/.e/. /(/4/) with kx/(/0/)k / /1 and r / R/, we have kx/(t/)k / /1 /4 for t /2 /[T/; T /+ /1/]/. This is possible by Lemma /4/./4/. Hence by Lemma /4/./6 /(i/) we can /nd an j/0 / /1 such that whenever j / j /0 and kX/(m/(j/)/)k / R/, kX/(m/(j /+ /1/)/)k kX/(m/(j/)/)k /= //(T /(j /+ /1/)/?/) / /1 /2 /: This implies that fX/(m/(j/)/) /: j / /0g is a/.s/. bounded/, and the claim follows/. /(ii/) For m/(j/) /&lt; n / m/(j /+ /1/)/, E/[kX/(n/)k /2 j Fm/(j/) /] /1/=/2 /= E/[k//(t/(n/)/?/)k /2 j F m/(j/) /] /1/=/2 /(kX/(m/(j/)/)k /\_ /1/) /(/2/8/) / E/[k//(t/(n/)/?/) /? b //(t/(n/)/?/)k /2 j Fm/(j/) /] /1/=/2 /(kX/(m/(j/)/)k /\_ /1/) /(/2/9/) /+E/[k b //(t/(n/)/?/)k /2 j Fm/(j/) /] /1/=/2 /(kX/(m/(j/)/)k /\_ /1/) Let /0 /&lt; / /&lt; /1 /2 /, and let / / /= //=/(/2C /2 /)/, for C/2 as in Lemma /4/./7/. We then obtain for / / / / /, E/[kX/(n/)k /2 j Fm/(j/) /] /1/=/2 / /(//=/2/)/(kX/(m/(j/)/)k /\_ /1/) /+E/[k b //(t/(n/)/?/)k /2 j Fm/(j/) /] /1/=/2 /(kX/(m/(j/)/)k /\_ /1/) /(/3/0/) Choose R/; T /&gt; /0 such that for any solution x/( / /) of the o/.d/.e/. /(/4/)/, kx/(t/)k /&lt; //=/2 for t /2

/2/5

/[T/; T /+ /1/]/, whenever kx/(/0/)k /&lt; /1 and r / R/. When kX/(m/(j/)/)k / R we then obtain E/[kX/(m/(j /+ /1/)/)k /2 j Fm/(j/) /] /1/=/2 / /kX/(m/(j/)/)k /(/3/1/) while by Lemma /4/./7 /(ii/) there exists a constant C such that the l/.h/.s/. of the inequality above is bounded by C a/.s/. when kX/(m/(j/)/)k / R/. Thus/, E/[kX/(m/(j /+ /1/)/)k /2 /] / /2/ /2 E/[kX/(m/(j/)/)k /2 /] /+ /2C /2 /: This establishes boundedness of E/[kX/(m/(j /+ /1/)/)k /2 /]/, and the proof then follows from /(/3/0/) and Lemma /4/./2/. u t

Lemma /4/./8 Suppose that /(A/1/)/, /(A/2/) and /(BS/) hold/, and that / / / / /. Then for some constant C/3 /&lt; /1/, sup t//0 E/[k b / /(t/) /? / /(t/)k /2 /] / C /3 //: Proof/: By /(A/2/) and Theorem /2/./1 /(ii/)/, sup n E/[kX/(n/)k /2 /] /&lt; /1/; sup n E/[kM/(n/)k /2 /] /&lt; /1/: The claim then follows from familiar arguments using the Bellman Gronwall Lemma exactly as in the proof of Lemma /4/./6/. u t Proof of Theorem /2/./3 To prove /(i/) we apply Theorem /2/./1 which allows us to choose an R /&gt; /0 such that sup n P/(kX/(n/)k /&gt; R/) /&lt; //: Let B/(c/) denote the ball centered at x / of radius c /&gt; /0/, and let /0 /&lt; / /&lt; //=/2 be such that if a solution x/( / /) of /(/2/) satis/es x/(/0/) /2 B/(//)/, then x/(t/) /2 B/(//=/2/) for t / /0/. Pick T /&gt; /0 such that if a solution x/( / /) of /(/2/) satis/es kx/(/0/)k / R/, then x/(t/) /2 B/(//=/2/) for t /2 /[T/; T /+ /1/]/. Then for all j / /0/, P/(e/(m/(j /+ /1/)/) / //) /= P/(e/(m/(j /+ /1/)/) / //; kX/(m/(j/)/)k /&gt; R/) /+P/(e/(m/(j /+ /1/)/) / //; kX/(m/)k / R/)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The result follows from this and Lemma /4/./7 /(ii/)/.

<!-- formula-not-decoded -->

u

t

X/(n/) for m/(j/) / n /&lt; m/(j /+ /1/)/. Here X/(n /+ /1/) was computed in terms of X/(n/) and the /`noise/' M/(n /+ /1/)/. In the asynchronous case/, however/, the evaluation of X j /(n /+ /1/) can involve Xj /(n/) for n /? / / m / n/, j /6/= i/. Therefore the argument leading to Lemma /4/./6 calls for a slight modi/cation/. While computing X/(n/)/; m/(j/) / n /&lt; m/(j /+ /1/)/, we plug into /2/7

Proof of Theorem /2/./5 The details of the proof/, though pedestrian in the light of the foregoing and /[/6/]/, are quite lengthy/, not to mention the considerable overhead of additional notation/, and are therefore omitted/. We brie/y sketch below a single point of departure in the proof/. In Lemma /4/./6 we compare two functions //( / /) and b //( / /) on the interval /[T /(j/)/; T /(j /+ /1/)/]/. The former in turn involved the iterates e X/(n/) for m/(j/) / n /&lt; m/(j /+ /1/)/, or/, equivalently/,

Xi/(m/) also features in the computation of X k /(l/) for m/(q/) / /` /&lt; m/(q /+ /1/)/, say/, with q /6/= j/, then e Xi/(m/) should be rede/ned there as X i /(m/)/=r/(q/)/. Thus the de/nition of e Xi/(m/) now becomes context/-dependent/. With this minor change/, the proofs of /[/6/] can be easily combined with the arguments used in the proofs of Theorems /2/./1 and /2/./2 to draw the desired conclusions/. u t /4/./3 The Markov model for any measurable A / I R d /. Under the assumptions /(/1/1/) and /(/1/2/) we show below that every compact subset of I R d is petite/, so that / is a / /-irreducible T /-chain/. We refer the reader to /[/1/6/] for further terminology and notation/. Lemma /4/./9 Suppose that /(A/1/)/, /(A/2/)/, /(/1/1/) and /(/1/2/) hold/, and that / / / / /. Then all com/pact subsets of I R d are petite for the Markov chain X/, and hence the chain is / /-irreducible/. Proof/: The conclusions of the theorem will be satis/ed if we can /nd a function s which

The bounds that we obtain for the Markov model /(/9/) are based upon the theory of / /irreducible Markov chains/.

A subset S / I R d is called petite if there exists a probability measure / on I R d and / /&gt; /0 such that the resolvent kernel K satis/es

<!-- formula-not-decoded -->

is bounded from below on compact sets/, and a probability / such that the resolvent kernel K satis/es the bound K/(x/; A/) / s/(x/)//(A/) for every x /2 I R d /, and any measurable subset A / I R d /. This bound is written succinctly as K / s /
 //. The /rst step of the proof is to apply the implicit function theorem together with /(/1/1/) and /(/1/2/) to obtain a bound of the form P d /(x/; A/) /= P/(X/(d/) /2 A j X/(/0/) /= x/) / ///(A/)/; x /2 O/;

/2/8

set O can be chosen independent of //, but the constant / may depend on //. For details on this construction see Chapter /7 of /[/1/6/]/. To complete the proof it is enough to show that K/(x/; O/) /&gt; /0/. To see this/, suppose

that / / / / /, and suppose that W/(n/) /= w / for all n/. Then the foregoing stability analysis shows that X/(n/) /2 O for all n su/ciently large/. Since w / is in the support of the marginal distribution of fW/(n/)g it then follows that K/(x/; O/) /&gt; /0/. From these two bounds/, we then have K/(x/; A/) / /2 /?d Z K/(x/; dy/)P d /(y/; A/) / /2 /?d /K/(x/; O/)//(A/)/: This is of the form K / s /
 / with s lower semicontinuous/, and positive everywhere/. The function s is therefore bounded from below on compact sets/, which proves the claim/. u t The previous lemma together with Theorem /2/./1 allows us to establish a strong form of ergodicity for the model/: Lemma /4/./1/0 Suppose that /(A/1/)/, /(A/2/)/, /(/1/1/) and /(/1/2/) hold/, and that / / / / /.

/2

/+ /1/)/;

<!-- formula-not-decoded -->

such that kxk / L/;

- /(i/) There exists a function V / /: I R d /! /[/1/; /1/) and constants b/; L /&lt; /1 and / /0 /&gt; /0 indepen/dent of / such that PV/ /(x/) / exp/(/?/ /0 //)V / /(x/) /+ bI C /(x/) where C /= fx /: kxk / Lg/. While the function V / will depend upon //, it is uniformly bounded as follows/, / /?/1 /(kxk /2 /+ /1/) / V / /(x/) / //(kxk /2 /+ /1/) where / / /1 does not depend upon //. /(ii/) The chain is V /-uniformly ergodic/, with V /(x/) /= kxk /2 /+ /1/. Proof/: Using /(/3/1/) we may construct T and L independent of / / / / E/[kX/(k /0 /)k /2 /+ /1 j X/(/0/) /= x/] / /(/1/=/2/)/(kxk where k/0 /= /[T/=//] /+ /1/. We now set

/ /0 /= log/(/2/)/=T /. Lipschitz continuity of the model gives the bounds on V / /. This proves /(i/)/. The V /-uniform ergodicity then follows from Lemma /4/./9 and Theorem /1/6/./0/./1 of /[/1/6/]/. u t We note that for small / and large x/, the Lyapunov function V / approximates V /1 plus a constant/, where V/1 /(x/) /= Z T /0 /(kx/(s/)k /2 /+ /1/)/2 s/=T ds/; x/(/0/) /= x/; and x/( / /) is a solution to /(/5/)/. If this o/.d/.e/. is asymptotically stable then the function V /1 is in fact a Lyapunov function for /(/5/)/, provided T /&gt; /0 is chosen su/ciently large/. In /[/1/7/] a bound is obtained on the rate of convergence / given in /(/1/0/) for a chain satisfying the drift condition PV/ /(x/) / /V /(x/) /+ bI C /(x/) The bound depends on the /\petiteness/" of the set C/; and the constants b /&lt; /1 and / /&lt; /1/. The bound on / obtained in /[/1/7/] also tends to unity with vanishing / since in the preceding lemma we have / /= exp/(/?/ /0 //) /! /1 as / /! /0/. From the structure of the algorithm this is not surprising/, but this underlines the fact that care must be taken in the choice of the stepsize //. References /[/1/] ABOUNADI/, J/./, BERTSEKAS/, D/./, BORKAR/, V/.S/./, Learning algorithms for Markov

- decision processes with average cost/, Lab/. for Info/. and Decision Systems/, M/.I/.T/./, /1/9/9/6/, /(Draft report/)/. /[/2/] BARTO/, A/.G/./, SUTTON/, R/.S/./, ANDERSON/, C/.W/./, Neuron/-like elements that can
- /3/0
- solve di/cult learning control problems/, IEEE Trans/. on Systems/, Man and Cybernetics /1/3 /(/1/9/8/3/)/, /8/3/5/-/8/4/6/. /[/3/] BENVENISTE/, A/./, METIVIER/, M/./, PRIOURET/, P/./, Adaptive Algorithms and Stochastic Approximations/, Springer Verlag/, Berlin/-Heidelberg/, /1/9/9/0/. /[/4/] BERTSEKAS/, D/./, TSITSIKLIS/, J/./, Neuro/-Dynamic Programming/, Athena Scienti/c/, Belmont/, MA/, /1/9/9/6/.
- Letters /2/9 /(/1/9/9/7/)/, /2/9/1/-/2/9/4/. /[/6/] BORKAR/, V/.S/./, Asynchronous stochastic approximation/, to appear in SIAM J/. Con/trol and Optim/. /1/9/9/8/.
- /[/7/] BORKAR/, V/.S/./, Recursive self/-tuning control of /nite Markov chains/, Applicationes Mathematicae /2/4 /(/1/9/9/6/)/, /1/6/9/-/1/8/8/.
- /[/1/3/] KONDA/, V/.R/./, BORKAR/, V/.S/./, Actor/-critic type learning algorithms for Markov de/cision processes/, submitted/. /[/1/4/] KUSHNER/, H/./, YIN/, G/./, Stochastic Approximation Algorithms and Applications/, Springer Verlag/, New York/, NY/, /1/9/9/7/.
- /[/8/] BORKAR/, V/.S/./, SOUMYANATH/, K/./, An analog scheme for /xed point computation/, Part I/: Theory/, IEEE Trans/. Circuits and Systems I/. Fundamental Theory and Appl/. /4/4 /(/1/9/9/7/)/, /3/5/1/-/3/5/4/. /[/9/] DAI/, J/.G/./, On the positive Harris recurrence for multiclass queueing networks/: a uni/ed approach via /uid limit models/, Ann/. Appl/. Prob/. /5 /(/1/9/9/5/)/, /4/9/-/7/7/. /[/1/0/] DAI/, J/.G/./, MEYN/, S/.P/./, Stability and convergence of moments for multiclass queueing networks via /uid limit models/, IEEE Trans/. Automatic Control /4/0 /(/1/9/9/5/)/, /1/8/8/9/-/1/9/0/4/. /[/1/1/] HIRSCH/, M/.W/./, Convergent activation dynamics in continuous time networks/, Neural Networks /2 /(/1/9/8/9/)/, /3/3/1/-/3/4/9/. /[/1/2/] JAAKOLA/, T/./, JORDAN/, M/.I/./, SINGH/, S/.P/./, On the convergence of stochastic itera/tive dynamic programming algorithms/, Neural Computation /6/, /(/1/9/9/4/)/, /1/1/8/5/-/1/2/0/1/.
- /3/1
- /[/1/5/] MALYSHEV/, V/.A/./, MEN/'SIKOV/, M/.V/./, Ergodicity/, continuity and analyticity of countable Markov chains/, Trans/. Moscow Math/. Soc/. /1 /(/1/9/8/2/)/, /1/-/4/8/. /[/1/6/] MEYN/, S/.P/./, TWEEDIE/, R/.L/./, Markov Chains and Stochastic Stability/, Springer Ver/lag/, London/, /1/9/9/3/.
- chains/, Annals of Applied Probability/, /4/, /1/9/9/4/. /[/1/8/] NEVEU/, J/./, Discrete Parameter Martingales/, North Holland/, Amsterdam/, /1/9/7/5/. /[/1/9/] SARGENT/, T/./, Bounded Rationality in Macroeconomics/, Clarendon Press/, Oxford/, /1/9/9/3/.
- /[/2/0/] TSITSIKLIS/, J/./, Asynchronous stochastic approximation and Q/-learning/, Machine Learning /1/6 /(/1/9/9/4/)/, /1/9/5/-/2/0/2/. /[/2/1/] WATKINS/, C/.J/.C/.H/./, DAYAN/, P/./, Q/-learning/, Machine Learning /8 /(/1/9/9/2/) /2/7/9/-/2/9/2/.

/3/2