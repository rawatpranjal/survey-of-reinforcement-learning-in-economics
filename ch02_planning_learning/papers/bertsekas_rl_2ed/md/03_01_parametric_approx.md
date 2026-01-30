# 3.1: Parametric Approximation Architectures

**Source:** Bertsekas, "A Course in Reinforcement Learning" (2nd ed., 2025)
**Pages:** 349-369
**Topics:** parametric approximation, cost function approximation, feature-based, linear architecture, nonlinear architecture, training

---

on the work by Gundawar, Li, and Bertsekas [GLB24]. A key fact is that for computer chess there are several programs (commonly called engines ), which supply a move selection policy [this is the maximizer's policy approximation ν in Eq. (2.117)]. Engines can also be used to supply board position evaluations [this is the cost function approximation ˜ J in Eq. (2.117)]. ‡

For a brief summary, our scheme is built around two components:

- (a) The position evaluator . This is implemented with one of the many publicly available chess engines, such as the popular and freely available champion program Stockfish, which produces an evaluation of any given position (normalized in a way that is standard in computer chess).
- (b) The nominal opponent . This is an approximation to the true opponent engine or human, whom we intend to play with. It outputs deterministically a move at each given position, against which we expect to play. In the absence of a known opponent, a reasonable choice is to use a competent chess engine as nominal opponent, such as for example the one used to provide position evaluations (e.g., Stockfish). In any case, it is important not to use a relative poor nominal opponent, which would lead us to underestimate the true opponent.

The nominal opponent and the position evaluator may be implemented with di ff erent chess engines. Moreover the nominal opponent may be changed from game to game to adapt to the real opponent at hand. An important fact is that stored knowledge of the opponent and evaluator engines, such as an opening book or an endgame database, are indirectly incorporated into the approximation in value space scheme.

To make the connection with our mathematical framework for minimax problems, we use the following notation:

- ÷ x k is the chess position of our player at time k .
- ÷ u k is the move choice of our player at time k in response to position x k .
- ÷ w k is the move choice of the nominal opponent at time k in response to position x k followed by move u k .

The resulting position at time k +1 is given by

<!-- formula-not-decoded -->

The ideas of the present section apply more broadly to any two-person antagonistic game, which uses computer engines to supply cost function approximation and move selection policies.

‡ We view the approach of this section as a meta algorithm , which is a broad term that describes an algorithm that 'provides a framework or strategy to develop, combine, or enhance other algorithms' (according to ChatGPT).

Current Position

Xk

Move Uk

All Logal moves

88!

88

888

888

$ 4&amp;

Stockfish

888

Nominal Opponent

Stockfish

Position

Evaluation

<!-- image -->

Position Evaluation

Figure 2.12.1 Illustration of the sequence of calculations of a one-step lookahead scheme for computer chess. Here we use the chess engine Stockfish for both opponent move generation and for position evaluation. At the current position x k , we generate all legal moves u k , and for each pair ( x k ↪ u k ), the opponent engine generates a single best move w k , resulting in the position

<!-- formula-not-decoded -->

We then use the position evaluation engine to evaluate each of the possible x k +1 .

where f is a known function. This corresponds to the dynamic system, where x k is viewed as the state, u k is viewed as the control, and w k is viewed as a known or random disturbance.

At the current position x k , our scheme operates as follows:

- (1) We generate all legal moves u k .
- (2) For each pair ( x k ↪ u k ), the nominal opponent generates a single best move w k , resulting in the position

<!-- formula-not-decoded -->

- (3) We evaluate each of the possible positions x k +1 .

Thus, there are a total of at most m position evaluations and m nominal opponent move generations, where m is an upper bound to the number of legal moves at any position.

Next Position

Xk+1

8 ₴

Figure 2.12.1 describes our scheme for the case of one-step lookahead, with Stockfish used for nominal opponent moves and position evaluations. Note that the scheme is well suited for parallel computation. In particular, the moves of the nominal opponent can be computed in parallel, and the positions following the generation of the opponent's moves can also be evaluated in parallel. Thus, with su ffi cient parallel computing resources, our scheme requires roughly twice as much computation as the underlying engines to generate a move.

The paper [GLB24] presents test results using strong chess engines, such as Stockfish, as well as other weaker engines. These results show that, similar to rollout, our scheme improves the performance of the position evaluation engine on which it is based, and for relatively weak engines, dramatically so. This finding generally assumes a reasonable choice for opponent move selection, such as for example the engine that is used for position evaluation. In our tests, a scheme that uses Stockfish of various strength levels as the engine for both nominal opponent and position evaluator has beaten the engine itself by a significant margin, although the winning margin diminishes as the strength of the engine increases. This is to be expected since Stockfish plays near-perfect chess at its highest levels. Qualitatively, this is similar to the performance of approximation in value space and rollout methods, which emulate a Newton step and attain a superlinear convergence, but with the amount of cost improvement diminishing near J * .

## Multistep Lookahead and Other Extensions

It is possible to introduce two-step and multistep lookahead in the preceding algorithm: the lookahead stage of Fig. 2.12.1 is replicated over multiple stages; see Fig. 2.12.2. For an /lscript -step lookahead scheme, the computation needed per move is m + m 2 + · · · + m /lscript opponent move generations plus m /lscript position evaluations. This computation time can be reduced if the lookahead tree is suitably pruned at the second stage. Moreover, it can be seen that given su ffi cient parallel computing resources, to generate a move, an /lscript -step lookahead scheme requires roughly /lscript + 1 times as much computation as the underlying engines. For more details, extensions, and computational testing, see the paper [GLB24].

Let us also note the possibility of replacing the single nominal opponent move with multiple (presumably good) moves. This creates a reduced minimax tree, where the moves at each stage, after the first one, are pruned, except for a few top moves, selected using some form of position evaluation. The player's move at the current position is then found with a minimax search over the reduced tree that consists of the moves that have not been pruned. Note that it is important not to prune any moves of the first stage in order to preserve the Newton step character of the approximation in value space scheme. On the other hand, the (pruned) multistep minimax

Current Position

Tk

Move uk

All Legal moves

888

888

888

888

Stockfish

83

Next Position

Xk+1

888

Nominal Opponent

18.2

82

Figure 2.12.2 Illustration of the two-step lookahead version of the computer chess scheme of Fig. 2.12.1. Again we use Stockfish for both opponent move generation and for position evaluation.

<!-- image -->

search can be costly, and may require a lot of position evaluations and move generations, so for the scheme to be viable, the pruning should be aggressive and the length of the lookahead should be limited.

## 2.12.5 Combined Approximation in Value and Policy Space for Sequential Noncooperative Games

Noncooperative games (also called nonzero-sum games, or Nash games) represent a significant extension of the zero-sum games that we discussed earlier. Here there are a finite number of agents who choose distinct controls, interact with each other, and aim to optimize their private cost functions. The analysis of the agents' choices in the absence of complete or partial information regarding the choices of the other agents has fascinated engineers, mathematicians, and social scientists since the seminal (26 pages!) Ph.D. dissertation of J. Nash [Nas50].

In this section we will discuss a relatively simple special class of nonzero-sum games, which involves sequential choices of the controls of the agents, with complete communication of these controls to the other agents. Still their treatment by DP is very challenging in general. In fact,

Uk+1

Wk+1

Stockfish

Xk+2

Position Evaluation

contrary to the minimax problem, where there is a Bellman equation (albeit complicated by components that are neither convex nor concave; cf. the discussion at the end of Section 2.12.2), for the class of problems of this section there may not be a Bellman equation that can be used as a basis for approximation in value space.

We will now focus on an important example of sequential noncooperative game, a leader-follower problem , also known as a Stackelberg game in economics and as a bilevel optimization problem in mathematical programming. It involves a dominant decision maker, called the leader , and m other decision makers, called the followers . Once the followers observe the decision of the leader, they optimize their choices according to their private cost functions. We make no assumption on the state and control spaces, other than that the constraints of the followers are decoupled.

Let us first consider for simplicity a two-stage stochastic environment with perfect state information, and the special case where both leader and followers act cooperatively in the sense that they minimize a common cost function. Thus, we assume that the leader knowing an initial state x 0 , makes a decision u 0 ∈ U 0 , and a state transition

<!-- formula-not-decoded -->

occurs with cost g 0 ( x 0 ↪ u 0 ↪ w 0 ) ↪ where w 0 is a random disturbance. Then the followers, knowing x 1 , choose decisions u /lscript 1 ∈ U /lscript 1 , /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , and a terminal state

<!-- formula-not-decoded -->

is generated with cost

<!-- formula-not-decoded -->

where w 1 is a random disturbance, and g 2 is the terminal cost function. The problem is to select u 0 and follower policies θ /lscript 1 , /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , so as to minimize the total expected cost.

Clearly this is a two-stage multiagent DP problem, which can be very hard to solve for large m , because of the large size of the control space of the second stage and the coupling of the followers' actions through the terminal cost. This two-stage problem can be solved approximately with multiagent rollout; see Section 2.9. Moreover, the extension from a two-stage to a multistage framework is straightforward.

However, the multiagent formulation just described fails when the followers do not share with the leader the same cost function. In this case it seems very di ffi cult to apply approximation in value space methods. On the other hand, a convenient approach is to convert the problem to a single

Approximation in policy space approaches are possible, and have been suggested in the literature, but have met with mixed success so far.

agent type of problem, involving just the leader and nominal policies of the followers , similar to the approach of Section 2.12.3, and the computer chess paradigm of Section 2.12.4. These policies, can be obtained using some form of policy training, and produce follower actions u /lscript 1 ∈ U /lscript 1 , /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m↪ that depend predictably on the current state x 1 and possibly the preceding action u 0 of the leader. The advantage of this approach is that the Newton step framework fully applies to the leader's problem, and the corresponding on-line implementation of the leader's policy is tractable, based on oneplayer approximation in value space schemes. Limited preliminary results using this approach have been encouraging, but more research along this line is needed.

## 2.13 NOTES, SOURCES, AND EXERCISES

In this chapter, we have first considered deterministic problems, then stochastic problems, and finally minimax problems. We have emphasized the e ff ectiveness of approximation in value space, which derives from its connection to Newton's method. We have discussed several types of approximation in value space schemes, with an emphasis on rollout and its variations, which are very e ff ective, reliable, and easily implementable.

Section 2.1: Our focus in this section has been on finite horizon problems, possibly involving a nonstationary system and cost per stage. However, the insights that can be obtained from the infinite horizon/stationary context fully apply. These include the interpretation of approximation in value space as a Newton step, and of rollout as a single step of the policy iteration method. The reason is that an N -step finite horizon/nonstationary problem can be converted to an infinite horizon/stationary problem with a termination state to which the system moves at the N th stage; see Section 1.6.4.

Section 2.2: Approximation in value space has been considered in an ad hoc manner since the early days of DP, motivated by the curse of dimensionality. Moreover, the idea of /lscript -step lookahead minimization with horizon truncation beyond the /lscript steps has a long history and is often referred to as 'rolling horizon' or 'receding horizon' optimization. Approximation in value space was reframed in the late 80s and was coupled with model-free simulation methods that originated in artificial intelligence.

Section 2.3: The main idea of rollout algorithms, obtaining an improved policy starting from some other suboptimal policy, has appeared in several DP contexts, including games; see e.g., Abramson [Abr90], and Tesauro and Galperin [TeG96]. The name 'rollout' was coined in [TeG96] in the context of backgammon; see Example 2.7.4. The use of the name 'rollout' has gradually expanded beyond its original context; for example samples

collected through trajectory simulation are referred to as 'rollouts' by some authors.

In this book, we will adopt the original intended meaning: rollout is an algorithm that provides policy improvement starting from a base policy, which is evaluated with some form of Monte Carlo simulation, perhaps augmented by some other calculation that may include a terminal cost function approximation. The author's rollout book [Ber20a] provides a more extensive discussion of rollout algorithms and their applications.

Following the original works on rollout for discrete deterministic and stochastic optimization (Bertsekas, Tsitsiklis, and Wu [BTW97], Bertsekas and Casta˜ non [BeC99], and the neuro-dynamic programming book [BeT96]), there has been a lot of research on rollout algorithms, which we list selectively in chronological order: Christodouleas [Chr97], Duin and Voss [DuV99], Secomandi [Sec00], [Sec01], [Sec03], Ferris and Voelker [FeV02], [FeV04], McGovern, Moss, and Barto [MMB02], Savagaonkar, Givan, and Chong [SGC02], Wu, Chong, and Givan [WCG02], [WCG03], Bertsimas and Popescu [BeP03], Guerriero and Mancini [GuM03], Tu and Pattipati [TuP03], Meloni, Pacciarelli, and Pranzo [MPP04], Yan et al. [YDR04], Nedich, Schneider, and Washburn [NSW05], Han, Lai, and Spivakovsky [HLS06], Lee et al. [LSG07], An et al. [ASP08], Berger et al. [BAP08], Besse and Chaib-draa [BeC08], Patek, Breton, and Kovatchev [PBK08], Sun et al. [SZL08], Tian, Bar-Shalom, and Pattipati [TBP08], Novoa and Storer [NoS09], Mishra et al. [MCT10], Malikopoulos [Mal10], Bertazzi et al. [BBG13], Sun et al. [SLJ13], Tesauro et al. [TGL13], Antunes and Heemels [AnH14], Beyme and Leung [BeL14], Goodson, Thomas, and Ohlmann [GTO15], [GTO17], Khashooei, Antunes, Heemels [KAH15], Li and Womer [LiW15], Mastin and Jaillet [MaJ15], Simroth, Holfeld, and Brunsch [SHB15], Huang, Jia, and Guan [HJG16], Lan, Guan, and Wu [LGW16], Lam, Willcox, and Wolpert [LWW16], Gommans et al. [GTA17], Lam and Willcox [LaW17], Ulmer [Ulm17], Bertazzi and Secomandi [BeS18], Sarkale et al. [SNC18], Ulmer at al. [UGM18], Zhang, Ohlmann, and Thomas [ZOT18], Arcari, Hewing, and Zeilinger [AHZ19], Chu, Xu, and Li [CXL19], Goodson, Bertazzi, and Levary [GBL19], Guerriero, Di Puglia, and Macrina [GDM19], Ho, Liu, and Zabinsky [HLZ19], Liu et al. [LLL19], Nozhati et al. [NSE19], Singh and Kumar [SiK19], Yu et al. [YYM19], Yuanhong [Yua19], Andersen, Stidsen, and Reinhardt [ASR20], Durasevic and Jakobovic [DuJ20], Issakkimuthu, Fern, and Tadepalli [IFT20], Lee et al. [LEC20], Li et al. [LZS20], Lee [Lee20], Montenegro et al. [MLM20], Meshram and Kaza [MeK20], Schope, Driessen, and Yarovoy [SDY20], Yan, Wang, and Xu [YWX20], Yue and Kontar [YuK20], Zhang, Kafouros, and Yu [ZKY20], Ho ff man et al. [HCR21], Houy and Flaig [HoF21], Li, Krakow, and Gopalswamy [LKG21], Liu et al. [LPS21], Nozhati [Noz21], Rim­ el­ e et al. [RGG21], Tuncel et al. [TBP21], Xie, Li, and Xu [XLX21], Bertsekas [Ber22d], Paulson, Sonouifar, and Chakrabarty [PSC22], Bai et al. [BLJ23], Rusmevichientong et al. [RST23], Wu and Zeng

[WuZ23], Gerlach and Piatkowski [GeP24], Samani, Hammar, and Stadler [SHS24], Samani et al. [SLD24], Yilmaz, Xiang, and Klein [YXK24], Zhang et al. [ZLZ24], Wang et al. [WTL25].

These references collectively include a large number of computational studies, discuss variants and problem-specific adaptations of rollout algorithms for a broad variety of practical problems, and consistently report favorable computational experience. The size of the cost improvement over the base policy is often impressive, evidently owing to the fast convergence rate of Newton's method that underlies rollout. Moreover these works illustrate some of the other important advantages of rollout: reliability, simplicity, suitability for on-line replanning, and the ability to interface with other RL techniques, such as neural network training, which can be used to provide suitable base policies and/or approximations to their cost functions.

The adaptation of rollout algorithms to discrete deterministic optimization problems, the notions of sequential consistency, sequential improvement, fortified rollout, and the use of multiple heuristics for parallel rollout were first given in the paper by Bertsekas, Tsitsiklis, and Wu [BTW97], and were also discussed in the neuro-dynamic programming book [BeT96]. Rollout algorithms for stochastic problems were further formalized in the papers by Bertsekas [Ber97b], and Bertsekas and Casta˜ non [BeC99]. Extensions to constrained rollout were first given in the author's papers [Ber05a], [Ber05b]. A survey of rollout in discrete optimization was given by the author in [Ber13a].

The model-free rollout algorithm, in the form given here, was first discussed in the RL book [Ber19a]. It is related to the method of comparison training, proposed by Tesauro [Tes89a], [Tes89b], [Tes01], and discussed by several other authors (see [DNW16], [TCW19]). This is a general method for training an approximation architecture to choose between two alternatives, using a dataset of expert choices in place of an explicit cost function.

The material on most likely sequence generation for n -grams, HMMs, and Markov Chains is recent, and was developed in the paper by Li and Bertsekas [LiB24]. As we have noted, our rollout-based most likely sequence generation algorithm can be useful to all contexts where the Viterbi algorithm is used. This includes algorithms for HMM parameter inference, which use the Viterbi algorithm as a subroutine.

Section 2.4: Our discussion of rollout, iterative deepening, and pruning in the context of multistep approximation in value space for deterministic problems contains some original ideas. In particular, the incremental multistep rollout algorithm and variations of Section 2.4.2 are presented here for the first time.

Note also that the multistep lookahead approximations described in Section 2.4 can be used more broadly within algorithms that employ forms of multistep lookahead search as subroutines. In particular, local search

algorithms, such as tabu search, genetic algorithms, and others, which are commonly used for discrete and combinatorial optimization, may be modified along the lines of Section 2.4 to incorporate RL and approximate DP ideas.

Section 2.5: Constrained forms of rollout were introduced in the author's papers [Ber05a] and [Ber05b]. The paper [Ber05a] also discusses rollout and approximation in value space for stochastic problems in the context of so-called restricted structure policies . The idea here is to simplify the problem by selectively restricting the information and/or the controls available to the controller, thereby obtaining a restricted but more tractable problem structure, which can be used conveniently in a one-step lookahead context. An example of such a structure is one where fewer observations are obtained, or one where the control constraint set is restricted to a single or a small number of given controls at each state.

Section 2.6: Rollout for continuous-time optimal control was first discussed in the author's rollout book [Ber20a]. A related discussion of policy iteration, including the motivation for approximating the gradient of the optimal cost-to-go ∇ x J t rather than the optimal cost-to-go J t , has been given in Section 6.11 of the neuro-dynamic programming book [BeT96]. This discussion also includes the use of value and policy networks for approximate policy evaluation and policy improvement for continuous-time optimal control. The underlying ideas have long historical roots, which are recounted in detail in the book [BeT96].

Section 2.7: The idea of the certainty equivalence approximation in the context of rollout for stochastic systems (Section 2.7.2) was proposed in the paper by Bertsekas and Casta˜ non [BeC99], together with extensive empirical justification. However, the associated theoretical insight into this idea was established more recently, through the interpretation of approximation in value space as a Newton step, which suggests that the lookahead minimization after the first step can be approximated with small degradation of performance. This point is emphasized in the author's book [Ber22a], and the papers [Ber22c], [Ber24].

Markov jump problems (Example 2.7.3) have an interesting theory and several diverse applications (see e.g., Sworder [Swo69], Wonham [Won70], Chizeck, Willsky, and Casta˜ non [CWC86], Abou-Kandil, Freiling, and Jank [AKFJ95], Costa and Do Val [CDV02], and Li and Bertsekas [LiB25a]. For related monographs, see Mariton [Mar90], Costa, Fragoso, and Marques [CFM05]). They seem to be particularly well suited for the application of approximation in value space, rollout schemes, and the use of certainty equivalence.

The idea of variance reduction in the context of rollout (Section 2.7.4) was proposed by the author in the paper [Ber97b]. See also the DP textbook [Ber17a], Section 6.5.2.

The paper by Chang, Hu, Fu, and Marcus [CHF05], and the 2007 first edition of their monograph proposed and analyzed adaptive sampling in connection with DP, as well as early forms of Monte Carlo tree search, including statistical tests to control the sampling process (a second edition, [CHF13], appeared in 2013). The name 'Monte Carlo tree search' has become popular, and in its current use, it encompasses a variety of methods that involve adaptive sampling, rollout, and extensions to sequential games. We refer to the papers by Coulom [Cou06], and Chang et al. [CHF13], the discussion by Fu [Fu17], and the survey by Browne et al. [BPW12].

Statistical tests for adaptive sampling has been inspired by works on multiarmed bandit problems; see Lai and Robbins [LaR85], Agrawal [Agr95], Burnetas and Katehakis [BuK97], Meuleau and Bourgine [MeB99], Auer, Cesa-Bianchi, and Fischer [ACF02], Kocsis and Szepesvari [KoS06], Dimitrakakis and Lagoudakis [DiL08], Audibert, Munos, and Szepesvari [AMS09], and Munos [Mun14]. The book by Lattimore and Szepesvari [LaS20] focuses on multiarmed bandit methods, and provides an extensive account of the UCB rule. For recent work on the theoretical properties of the UCB and UCT rules, see Shah, Xie, and Xu [SXX22], and Chang [Cha24].

Adaptive sampling and MCTS may be viewed within the context of a broader class of on-line lookahead minimization techniques, sometimes called on-line search methods. These techniques are based on a variety of ideas, such as random search and intelligent pruning of the lookahead tree. One may naturally combine them with approximation in value space and (possibly) rollout, although it is not necessary to do so (the multistep minimization horizon may extend to the terminal time N ). For representative works, some of which apply to continuous spaces problems, including POMDP, see Hansen and Zilberstein [HaZ01], Kearns, Mansour, and Ng [KMN02], Peret and Garcia [PeG04], Ross et al. [RPP08], Silver and Veness [SiV10], Hostetler, Fern, and Dietterich [HFD17], and Ye et al. [YSH17]. The multistep lookahead approximation ideas of Section 2.4 may also be viewed within the context of on-line search methods.

Another rollout idea for stochastic problems, which we have not discussed in this book, is the open-loop feedback controller (OLFC), a suboptimal control scheme that dates to the 60s; see Dreyfus [Dre65]. The OLFC applies to POMDP as well, and uses an open-loop optimization over the future evolution of the system. In particular, it uses the current information vector I k to determine the belief state b k . It then solves the open-loop problem of minimizing

<!-- formula-not-decoded -->

subject to the constraints

<!-- formula-not-decoded -->

and applies the first control u k in the optimal open-loop control sequence ¶ u k ↪ u k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ♦ . It is easily seen that the OLFC is a rollout algorithm that uses as base policy the optimal open-loop policy for the problem (the one that ignores any state or observation feedback).

For a detailed discussion of the OLFC, we refer to the author's survey paper [Ber05a] (Section 4) and DP textbook [Ber17a] (Section 6.4.4). The survey [Ber05a] discusses also a generalization of the OLFC, called partial open-loop-feedback-control , which calculates the control input on the basis that some (but not necessarily all) of the observations will in fact be taken in the future, and the remaining observations will not be taken. This method often allows one to deal with those observations that are troublesome and complicate the solution, while taking into account the future availability of other observations that can be reasonably dealt with. A computational case study for hydrothermal power system scheduling is given by Martinez and Soares [MaS02]. A variant of the OLFC, which also applies to minimax control problems, is given in the author's paper [Ber72b], together with a proof of a cost improvement property over the optimal open-loop policy.

Section 2.8: The role of stochastic programming in providing a link between stochastic DP and continuous spaces deterministic optimization is well known; see the texts by Birge and Louveaux [BiL97], Kall and Wallace [KaW94], and Prekopa [Pre95], and the survey by Ruszczynski and Shapiro [RuS03]. Stochastic programming has been applied widely, and there is much to be gained from its combination with RL. The material of this section comes from the author's rollout book [Ber20a], Section 2.5.2. For a computational study that has tested the ideas of this section on a problem of maintenance scheduling, see Hu et al. [HWP22]. For a related application, see Gioia, Fadda, and Brandimarte [GFB24].

Section 2.9: The multiagent rollout algorithm was proposed in the author's papers [Ber19c], [Ber20b]. The paper [Ber21a] provides an extensive overview of this research. For followup work, see the sources given for Section 1.6 of Chapter 1.

Section 2.10: The material on rollout for Bayesian optimization and sequential estimation comes from a recent paper by the author [Ber22d]. This work is also the basis for the adaptive control material of Section 2.11, and has been included in the book [Ber22a]. The paper by Bhambri, Bhattacharjee, and Bertsekas [BBB22] discusses this material for the case of a deterministic system, applies rollout to sequential decoding in the context of the challenging Wordle puzzle, and provides an implementation using some popular base heuristics, with performance that is very close to optimal. For related work see Loxley and Cheung [LoC23].

Section 2.11: The POMDP framework for adaptive control dates to the 60s, and has stimulated substantial theoretical investigations; see Mandl

[Man74], Borkar and Varaiya [BoV79], Doshi and Shreve [DoS80], Kumar and Lin [KuL82], and the survey by Kumar [Kum85]. Some of the pitfalls of performing parameter estimation while simultaneously applying adaptive control have been described by Borkar and Varaiya [BoV79], and by Kumar [Kum83]; see [Ber17a], Section 6.8 for a related discussion.

The papers just mentioned have proposed on-line estimation of the unknown parameter θ and the use at each time period of a policy that is optimal for the current estimate. The papers provide nontrivial analyses that assert asymptotic optimality of the resulting adaptive control schemes under appropriate conditions. If parameter estimation schemes are similarly used in conjunction with rollout, as suggested in Section 2.11 [cf. Eq. (2.105)], one may conjecture that an asymptotic cost improvement property can be proved for the rollout policy, again under appropriate conditions.

Section 2.12: The treatment of sequential minimax problems by DP has a long history. For some early influential works, see Blackwell and Girshick [BlG54], Shapley [Sha53], and Witsenhausen [Wit66]. In minimax control problems, the maximizer is assumed to make choices with perfect knowledge of the minimizer's policy. If the roles of maximizer and minimizer are reversed, i.e., the maximizer has a policy (a sequence of functions of the current state) and the minimizer makes choices with perfect knowledge of that policy, the minimizer gains an advantage, the problem may genuinely change, and the optimal value may be reduced. Thus 'min-max' and 'max-min' are generally two di ff erent problems. In classical two-person zero-sum game theory, however, the main focus is on situations where the min-max and max-min are equal. By contrast, in engineering worst case design, the min-max and max-min values are typically unequal.

There is substantial literature on sequential zero-sum games in the context of DP, often called Markov games . The classical paper by Shapley [Sha53] addresses discounted infinite horizon games. A PI algorithm for finite-state Markov games was proposed by Pollatschek and Avi-Itzhak [PoA69], and was interpreted as a Newton method for solving the associated Bellman equation. They have also shown that the algorithm may not converge to the optimal cost function. Computational studies have verified that the Pollatschek and Avi-Itzhak algorithm converges much faster than its competitors, when it converges (see Breton et al. [BFH86], and also Filar and Tolwinski [FiT91], who proposed a modification of the algorithm). Related methods have been discussed for Markov games by van der Wal [Van78] and Tolwinski [Tol89]. Raghavan and Filar [RaF91], and Filar and Vrieze [FiV96] provide extensive surveys of the research up to 1996.

The author's paper [Ber21b] has explained the reason behind the unreliable behavior of the Pollatschek and Avi-Itzhak algorithm. This explanation relies on the Newton step interpretation of PI given in Chapter 1: in the case of Markov games, the Bellman operator does not have the concavity property that is typical of one-player games. The paper [Ber21b] has

also provided a modified algorithm with solid convergence properties, which applies to very general types of sequential zero-sum games and minimax control. The algorithm is based on the idea of constructing a special type of uniform contraction mapping, whose fixed point is the solution of the underlying Bellman equation; this idea was first suggested for discounted and stochastic shortest path problems by Bertsekas and Yu [BeY12], [YuB13]. Because the uniform contraction property is with respect to a sup-norm, the corresponding PI algorithm is convergent even under a totally asynchronous implementation. The algorithm, its variations, and the analysis of the paper [Ber21b] were incorporated as Chapter 5 in the 3rd edition of the author's abstract DP book [Ber22b].

The paper by Yu [Yu14] provides an analysis of stochastic shortest path games, where the termination state may not be reachable under some policies, following earlier work by Patek and Bertsekas [PaB99]. The paper [Yu14] also includes a rigorous analysis of the Q-learning algorithm for stochastic shortest path games (without any cost function approximation). The papers by Perolat et al. [PSP15], [PPG16], and the survey by Zhang, Yang, and Basar [ZYB21] discuss alternative RL methods for games.

The author's paper [Ber19b] develops VI, PI, and Dijkstra-like finitely terminating algorithms for exact solution of 'robust shortest path planning' problems, which involve finding a shortest path assuming adversarial uncertainty in the state transitions. The paper also discusses related rollout algorithms for approximate solution.

Combining approximation in value and policy space for minimax control (Section 2.12.3) and noncooperative games (Section 2.12.5) is formally presented here for the first time, but it is a natural idea, which undoubtedly has occurred to several researchers. The computer chess methodology of Section 2.12.4 is based on this idea, and was introduced in the paper by Gundawar, Li, and Bertsekas [GLB24], as an application of model predictive control to computer chess.

but from AC it generates

## E X E R C I S E S

## 2.1 (A Traveling Salesman Rollout Example with a Sequentially Improving Heuristic)

Consider the traveling salesman problem of Example 1.2.3 and Fig. 1.2.11, and the rollout algorithm starting from city A.

- (a) Assume that the base heuristic is chosen to be the farthest neighbor heuristic, which completes a partial tour by successively moving to the farthest neighbor city not visited thus far. Show that this base heuristic is sequentially consistent. What are the tours produced by this base heuristic and the corresponding rollout algorithm? Answer : The base heuristic will produce the tour A → AD → ADB → ADBC → A with cost 45. The rollout algorithm will produce the tour A → AB → ABD → ABDC → A with cost 13.
- (b) Assume that the base heuristic at city A is the nearest neighbor heuristic, while at the partial tours AB, AC, and AD it is the farthest neighbor heuristic. Show that this base heuristic is sequentially improving but not sequentially consistent. Compute the final tour generated by rollout.

Solution of part (b): Clearly the base heuristic is not sequentially consistent, since from A it generates

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

However, it is seen that the sequential improvement criterion (2.13) holds at each of the states A, AB, AC, and AD (and also trivially for the remaining states).

The base heuristic at A is the nearest neighbor heuristic so it generates

<!-- formula-not-decoded -->

The rollout algorithm at state A looks at the three successor states AB, AC, AD, and runs the farthest neighbor heuristic from each, and generates:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so the rollout algorithm will move from A to AB.

Then the rollout algorithm looks at the two successor states ABC, ABD, and runs the base heuristic (whatever that may be; it does not matter) from each. The paths generated are:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so the rollout algorithm will move from AB to ABD.

Thus the final tour generated by the rollout algorithm is

<!-- formula-not-decoded -->

## 2.2 (A Generic Example of a Base Heuristic that is not Sequentially Improving)

Consider a rollout algorithm for a deterministic problem with a base heuristic that produces an optimal control sequence at the initial state x 0 , and uses the (optimal) first control u 0 of this sequence to move to the (optimal) next state x 1 . Suppose that the base heuristic produces a strictly suboptimal sequence from every successor state x 2 = f 1 ( x 1 ↪ u 1 ) ↪ u 1 ∈ U 1 ( x 1 ) ↪ so that the rollout yields a control u 1 that is strictly suboptimal. Show that the trajectory produced by the rollout algorithm starting from the initial state x 0 is strictly inferior to the one produced by the base heuristic starting from x 0 , while the sequential improvement condition does not hold.

## 2.3 (Computational Exercise - Parking with Problem Approximation and Rollout)

In this computational exercise we consider a more complex, imperfect state information version of the one-directional parking problem of Example 1.6.1. Recall that in this problem a driver is looking for a free parking space in an area consisting of N spaces arranged in a line, with a garage at the end of the line (space N ). The driver starts at space 0 and traverses the parking spaces sequentially, i.e., from each space he/she goes to the next space, up to when he/she decides to park in space k at cost c ( k ), if space k is free. Upon reaching the garage, parking is mandatory at cost C .

In Example 1.6.1, we assumed that the driver knows the probabilities p ( k + 1) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ p ( N -1) of the parking spaces ( k + 1) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ( N -1), respectively, being free. Under this assumption, the state at stage k is either the termination state t (if already parked), or it is F (location k free), or it is F (location k taken), and the DP algorithm has the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for the states other than the termination state t , while for t we have J ∗ k ( t ) = 0 for all k .

We will now consider the more complex variant of the problem where the probabilities p (0) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ p ( N -1) do not change over time, but are unknown to

the driver, so that he/she cannot use the exact DP algorithm (2.119)-(2.120). Instead, the driver considers a one-step lookahead approximation in value space scheme, which uses empirical estimates of these probabilities that are based on the ratio f k k +1 ↪ where f k is the number of free spaces seen up to space k , after the free/taken status of spaces 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ k has been observed. In particular, these empirical estimates are given by

<!-- formula-not-decoded -->

where f k is the number of free spaces seen up to space k , and γ and ¯ p ( m ) are fixed numbers between 0 and 1. Of course the values f k observed by the driver evolve according to the true (and unknown) probabilities p (0) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ p ( N -1) according to

<!-- formula-not-decoded -->

For the solution of this exercise you may assume any reasonable values you wish for N , p ( m ), ¯ p ( m ), and γ . Recommended values are N ≥ 100, and probabilities p ( m ) and ¯ p ( m ) that are nonincreasing with m .

The decision made by the approximation in value space scheme is to park at space k if and only if it is free and in addition

<!-- formula-not-decoded -->

where ˜ J k +1 ( F ) and ˜ J k +1 ( F ) are the cost-to-go approximations from stage k +1. Consider the following two di ff erent methods to compute ˜ J k +1 ( F ) and ˜ J k +1 ( F ) for use in Eq. (2.123):

- (1) Here the approximate cost function values ˜ J k +1 ( F ) and ˜ J k +1 ( F ) are obtained by using problem approximation, whereby at time k it is assumed that the probabilities of free/taken status at the future spaces m = k + 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N 1 are b k ( m↪f k ), m = k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N 1, as given by Eq. (2.121).
2. --

More specifically, ˜ J k +1 ( F ) and ˜ J k +1 ( F ) are obtained by solving optimally the problem whereby we use the probabilities b k ( m↪f k ) of Eq. (2.121) in place of the unknown p ( m ) in the DP algorithm (2.119)-(2.120):

<!-- formula-not-decoded -->

where ˆ J k +1 ( F ) and ˆ J k +1 ( F ) are given at the last step of the DP algorithm

<!-- formula-not-decoded -->

- (2) Here for each k , the approximate cost function values ˜ J k +1 ( F ) and ˜ J k +1 ( F ) are obtained by using rollout with a greedy base heuristic (park as soon as possible), and Monte Carlo simulation. In particular, according to this greedy heuristic, we have ˜ J k +1 ( F ) = c ( k + 1). To compute ˜ J k +1 ( F ) we generate many random trajectories by running the greedy heuristic forward from space k +1 assuming the probabilities b k ( m +1 ↪ f k ) of Eq. (2.121) in place of the unknown p ( m +1), m = k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, and we average the cost results obtained.
2. (a) Use Monte Carlo simulation to compute the expected cost from spaces 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, when using each of the two schemes (1) and (2).
3. (b) Compare the performance of the schemes of part (a) with the following:
4. (i) The optimal expected costs J ∗ k ( F ) and J ∗ k ( F ) from k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, using the DP algorithm (2.119)-(2.120), and the probabilities p ( m ), m = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, that you used for the random generation of the numbers of free spaces f k [cf. Eq. (2.122)].
5. (ii) The expected costs ˆ J k ( F ) and ˆ J k ( F ) from k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1 that are attained by using the greedy base heuristic. Argue that these are given by

<!-- formula-not-decoded -->

- (c) Argue that scheme (1) becomes superior to scheme (2) in terms of cost attained as γ ≈ 1 and ¯ p ( m ) ≈ p ( m ). Are your computational results in rough agreement with this assertion?
- (d) Argue that as γ ≈ 0 and N &gt;&gt; 1, scheme (1) becomes superior to scheme (2) in terms of cost attained from parking spaces k &gt;&gt; 1.
- (e) What happens if the probabilities p ( m ) do not change much with m ?

## 2.4 (Breakthrough Problem with a Random Base Heuristic)

Consider the breakthrough problem of Example 2.3.2 with the di ff erence that instead of the greedy heuristic, we use the random heuristic, which at a given node selects one of the two outgoing arcs with equal probability. Denote by

<!-- formula-not-decoded -->

the probability of success of the random heuristic in a graph of k stages, and by R k the probability of success of the corresponding rollout algorithm. Show that for all k

<!-- formula-not-decoded -->

and that

<!-- formula-not-decoded -->

Conclude that R k glyph[triangleleft]D k increases exponentially with k .

## 2.5 (Breakthrough Problem with Truncated Rollout)

Consider the breakthrough problem of Example 2.3.2 and consider a truncated rollout algorithm that uses a greedy base heuristic with /lscript -step lookahead. This is the same algorithm as the one described in Example 2.3.2, except that if both outgoing arcs of the current node at stage k are free, the rollout algorithm considers the two end nodes of these arcs, and from each of them it runs the greedy algorithm for min ¶ l↪ N -k -1 ♦ steps. Consider a Markov chain with l + 1 states, where states i = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ l -1 correspond to the path generated by the greedy algorithm being blocked after i arcs. State /lscript corresponds to the path generated by the greedy algorithm being unblocked after /lscript arcs.

- (a) Derive the transition probabilities for this Markov chain so that it models the operation of the rollout algorithm.
- (b) Use computer simulation to generate the probability of a breakthrough, and to demonstrate that for large values of N , the optimal value of /lscript is roughly constant and much smaller than N (this can also be justified analytically, by using properties of Markov chains).

## 2.6 (Incremental Truncated Rollout Algorithm for Constraint Programming)

Consider a discrete N -stage optimization problem, involving a tree with a root node s that plays the role of an artificial initial state, and N layers of states x 1 = ( u 0 ) ↪ x 2 = ( u 0 ↪ u 1 ) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x N = ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ), as shown in Fig. 2.1.4. We allow deadend nodes in the graph of this problem, i.e., states that have no successor states, and thus cannot be part of any feasible solution. We also assume that all stage costs as well as the terminal cost are 0. The problem is to find a feasible solution, i.e., a sequence of N transitions through the graph that starts at the initial state s and ends at some node of the last layer of states x N .

- (a) Argue that this is a discrete spaces formulation of a constraint programming problem, such as the one described in Section 2.1.
- (b) Describe in detail an incremental rollout algorithm with /lscript -step lookahead minimization and m -step rollout truncation, which is similar to the IMR algorithm of Section 2.4.2 and operates as follows: The algorithm maintains a connected subtree S that contains the initial state s . The base policy at a state x k either generates a feasible sequence of m arcs starting at x k , where m is an integer that satisfies 1 ≤ m ≤ min ¶ m↪N -k ♦ , or determines that such a sequence does not exist. In the former case the node x k is expanded by adding all of its neighbor nodes to S . In the latter case, the node x k is deleted from S . The algorithm terminates once a state x N of the last layer is added to S .
- (c) Argue that since the algorithm cannot keep deleting nodes indefinitely, one of two things will eventually happen:
- (1) The graph S will be reduced to just the root node s , proving that there is no feasible solution.
- (2) The algorithm will terminate with a feasible solution.

- (d) Suppose that the algorithm is operated so that the selected node x k at each iteration is a leaf node of S , which is at maximum arc distance from s (the number of arcs of the path connecting s and x k is maximized). Show that the subtree S always consists of just a path of nodes, together with all the neighbor nodes of the nodes of the path. Conclude that in this case, the algorithm can be implemented so that it requires O ( Nd ) memory storage, where d is the maximum node degree. How does this algorithm compare with a depth-first search algorithm for finding a feasible solution?
- (e) Describe an adaptation of the algorithm of part (d) for the case where most of the arcs have cost 0 but there are some arcs with positive cost.

## 2.7 (Purchasing Over Time with Multiagent Rollout)

Consider a market that makes available for purchase m products over N time periods, and a buyer that may or may not buy any one of these products subject to cash availability. For each product i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m and time period k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, we denote:

- a i k : The asking price of product i at time k (the case where product i is unavailable for purchase is modeled by setting a i k to ∞ ).
- v i k : The value to the buyer of product i at time k .
- u i k : The decision to buy ( u i k = 1) or not to buy ( u i k = 0) product i at time k .

The conditional distributions P ( a i k +1 ♣ a i k ↪ u i k = 1) and P ( a i k +1 ♣ a i k ↪ u i k = 0) are given. (Thus when u i k = 1, product i will be made available at the next time period at a possibly di ff erent price a i k +1 ; however, it may also be unavailable, i.e., a i k +1 = ∞ .)

The amount of cash available to the buyer at time k is denoted by c k , and evolves according to

<!-- formula-not-decoded -->

The initially available cash c 0 is a given positive number. Moreover, we have the constraint

<!-- formula-not-decoded -->

i.e., the buyer may not borrow to buy products. The buyer aims to maximize the total value obtained over the N time periods, plus the remaining cash at time N :

<!-- formula-not-decoded -->

- (a) Formulate the problem as a finite horizon DP problem by identifying the state, control, and disturbance spaces, the system equation, the cost function, and the probability distribution of the disturbance. Write the corresponding exact DP algorithm.

- (b) Introduce a suitable base policy and formulate a corresponding multiagent rollout algorithm for addressing the problem.

## 2.8 (Treasure Hunting Using Adaptive Control and Rollout)

Consider a problem of sequentially searching for a treasure of known value v among n given locations. At each time period we may either select a location i to search at cost c i &gt; 0, or we may stop searching. Moreover, if the search location i is di ff erent from the location j where we currently are, we incur an additional switching cost s ij ≥ 0. If the treasure is at location i , a search at that location will find it with known probability β i &lt; 1. Our initial location is given, and the a priori probabilities p i , i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n , that the treasure is at location i are also given. We assume that ∑ n i =1 p i &lt; 0, so there is positive probability that there is no treasure at any one of the n locations.

- (a) Formulate the problem as a special case of the adaptive control problem of Section 2.11, with the parameter θ taking one of ( n + 1) values, θ 0 ↪ θ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ n , where θ 0 corresponds to the case where there is no treasure at any location, and θ i , i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n , corresponds to the case where the treasure is at location i . Use as state the current location together with the current probability distribution of θ , and use as control the choice between stopping the search or continuing the search at one of the n locations.
- (b) Consider the special case where there is only one location. Show that the optimal policy is to continue searching up to the point where the conditional expected benefit of the search falls below a certain threshold, and that the optimal cost can be computed very simply. (The proof is given in Example 4.3.1 of the DP textbook [Ber17a].)
- (c) Formulate the rollout algorithm of Section 2.11 with two di ff erent base policies:
- (1) A policy that is optimal among the policies that never switch to another location (they continue to search the same location up to stopping).
- (2) A policy that is optimal among the policies that may stay at the current location or may switch to the location that is most likely to contain the treasure according to the current probability distribution of θ .
- (d) Implement the preceding two rollout algorithms using reasonable problem data of your choice.

## Learning Values and Policies

| Contents                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 3.1. Parametric Approximation Architectures . . . . . . . p. 351 3.1.1. Cost Function Approximation . . . . . . . . . p. 352 3.1.2. Feature-Based Architectures . . . . . . . . . p. 353 3.1.3. Training of Linear and Nonlinear Architectures . p. 364 3.2. Neural Networks . . . . . . . . . . . . . . . . . p. 372 3.2.1. Training of Neural Networks . . . . . . . . . p. 377 3.2.2. Multilayer and Deep Neural Networks . . . . . p. 378 3.3. Learning Cost Functions in Approximate DP . . . . . p. 379 3.3.1. Fitted Value Iteration . . . . . . . . . . . . p. 380 3.3.2. Q-Factor Parametric Approximation - Model-Free . . Implementation . . . . . . . . . . . . . . . p. 382 3.3.3. Parametric Approximation in Infinite Horizon . . . . Problems - Approximate Policy Iteration . . . . p. 384 3.3.4. Optimistic Policy Iteration with Parametric Q-Factor . Approximation - SARSA and DQN . . . . . . p. 388 3.3.5. Approximate Policy Iteration for Infinite Horizon . . . POMDP . . . . . . . . . . . . . . . . . . p. 390 3.3.6. Advantage Updating - Approximating Q-Factor . . . Di ff erences . . . . . . . . . . . . . . . . . p. 395 3.3.7. Di ff erential Training of Cost Di ff erences for Rollout p. 397 3.4. Learning a Policy in Approximate DP . . . . . . . . p. 399 3.4.1. The Use of Classifiers for Approximation in Policy . . Space . . . . . . . . . . . . . . . . . . . p. 400 3.4.2. Policy Iteration with Value and Policy Networks p. 404 3.4.3. Why Use On-Line Play and not Just Train a Policy . . Network to Emulate the Lookahead Minimization? p. 406 |