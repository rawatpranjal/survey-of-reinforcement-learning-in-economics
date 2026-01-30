## DYNAMIC GAMES WITH ASYMMETRIC INFORMATION: A FRAMEWORK FOR EMPIRICAL WORK*

## Chaim Fershtman and Ariel Pakes

We develop a framework for the analysis of dynamic oligopolies with persistant sources of asymmetric information that enables applied analysis of situations of empirical importance that have been difficult to deal with. The framework generates policies that are ''relatively'' easy for agents to use while still being optimal in a meaningful sense, and is amenable to empirical research in that its equilibrium conditions can be tested and equilibrium policies are relatively easy to compute. We conclude with an example that endogenizes the maintenance decisions of electricity generators when the costs states of the generators are private information. JEL Codes: L13, C73, D82.

## I. Introduction

This article develops a framework for the analysis of dynamic oligopolies with persistent sources of asymmetric information that can be used in a variety of situations which are both of empirical importance and have not been adequately dealt with in prior applied work. These situations include competition between producers when there is a producer attribute which is unknown to its competitors and serially correlated over time, investment games where the outcome of the investment is unobserved, or repeated auctions for primary products (e.g., timber) where the capacity available to process the quantity acquired at the auction is private information. Less obviously, but probably more empirically important, the framework also allows us to analyze markets in which the decisions of both producers and consumers have dynamic implications, but consumers make decisions with different information sets then producers do. As will be discussed, this enables applied analysis of dyanmics in durable, experience, storeable, and network good industries.

In building the framework we have two goals. First we want a framework that generates policies that are ''relatively'' easy for agents to use while still being optimal in some meaningful sense

*We thank two referees; the editor Elhanan Helpman; John Asker, Susan Athey, Adam Brandenburger, Eddie Dekel, Liran Einav, Drew Fudenberg, Phil Haile, Robin Lee, Greg Lewis, and Michael Ostrovsky for constructive comments; and Niyati Ahuja for superb research assistance. We would like to thank the Binational Science Foundation (BSF 2008020) for financial support.

! The Author(s) 2012. Published by Oxford University Press, on behalf of President and Fellows of Harvard College. All rights reserved. For Permissions, please email: journals .permissions@oup.com

The Quarterly Journal of Economics (2012), 1-51. doi:10.1093/qje/qjs025.

of the word. In particular the framework should not require the specification and updating of players' beliefs about their opponents types, as in perfect Bayesian equilibrium, and should not require agents to retain information that it is impractical for them to acquire. Second we want the framework to be usable by empirical researchers; its conditions should be defined in terms of observable magnitudes and it should generate policies which can be computed with relative ease (even when there are many underlying variables that effect the returns to different choices). The twin goals of ease of use to agents and ease of analysis by the applied research work out, perhaps not surprisingly, to have strong complimentarities.

To accomplish these tasks we extend the framework in Ericson and Pakes (1995) to allow for asymmetric information. 1 Each agent's returns in a given period are determined by all agents' ''payoff relevant'' state variables and their actions. The payoff relevant random variables of producers would typically include indexes of their cost function, qualities of the goods they market, and so on, whereas in a durable good market those of consumers would include their current holdings of various goods and the household's demographic characteristics. Neither a player's ''payoff relevant'' state variables nor its actions are necessarily observed by other agents. Thus producers might not know either the cost positions or the details of supplier contacts of their competitors, and in the durable goods example neither consumers nor producers need know the entire distribution of holdings crossed with household characteristics (even though this will determine the distribution of future demand and prices).

The fact that not all state variables are observed by all agents and that the unobserved states may be correlated over time implies that variables that are not currently payoff relevant but are related to the unobserved past states of other agents will help predict other agent's behavior. Consequently, they will help predict the returns from a given agent's current actions. So in addition to payoff relevant state variables agents have ''informationally relevant'' state variables. For example, in many markets past prices will be known to agents and will contain information on likely future prices.

1. Indeed our assumptions nest the generalizations to Ericson and Pakes (1995) reviewed in Doraszelski and Pakes (2007).

The ''types'' of the agents, which are defined by their state variables, are only partially observed by other agents and evolve over time. In the durable goods example, the joint distribution of household holdings and characteristics will evolve with household purchases, and the distribution of producer costs and goods marketed will evolve with the outcomes of investment decisions. As a result each agent continually changes its perceptions of the likely returns from its own possible actions. 2

Recall that we wanted our equilibrium concept to be testable. This, in itself, rules out basing these perceptions on Bayesian posteriors, as these posteriors are not observed. Instead we assume that the agents use the outcomes they experienced in past periods that had conditions similar to the conditions the agent is currently faced with to form an estimate of expected returns from the actions they can currently take. Agents act so as to maximize the discounted value of future returns implied by these expectations. So in the durable goods example a consumer will know its own demographics and may have kept track of past prices, while the firms might know past sales and prices. Each agent would then choose the action that maximized its estimate of the expected discounted value of its returns conditional on the information at its disposal. We base our equilibrium conditions on the consistency of each agents' estimates with the expectation of the outcomes generated by the agents' decisions.

More formally we define a state of the game to be the information sets of all of the players (each information set contains both public and private information). An experience-based equilibrium (EBE) for our game is a triple which satisfies three conditions. The triple consists of (1) a subset of the set of possible states, (2) a vector of strategies defined for every possible information set of each agent, and (3) a vector of values for every state that provides each agent's expected discounted value of net cash flow conditional on the possible actions that agent can take at that state. The conditions we impose on this triple are as follows. The first condition is that the equilibrium policies ensure that once we visit a state in our subset we stay within that subset in all future periods, visiting each point in that subset repeatedly. That is, the subset of states is a recurrent class of the Markov

2. Dynamic games with asymmetric information have not been used extensively to date, a fact that attests (at least in part) to their complexity. Notable exceptions are Athey and Bagwell (2008), and Cole and Kocherlakota (2001).

process generated by the equilibrium strategies. The second condition is that the strategies are optimal given the evaluations of outcomes. The final condition is that optimal behavior given these evaluations actually generates expected discounted value of future net cash flows that are consistent with these evaluations in the recurrent subset of states.

The conditions defining an EBE do not restrict the players' perceptions of the value of outcomes from nonequilibrium strategies. This is consistent with our focus on equilibrium conditions that use information that could be learned from the outcomes of equilibrium play, as without further restrictions neither the players nor researchers would have access to the experience which would be required to obtain consistent estimates of the outcomes from nonequilibrium strategies. Consequently the EBE conditions, in and of themselves, do not restrict agents' behavior; any profile of strategies can be rationalized as EBE. There are two approaches to choosing restrictions on the value of outcomes from nonequilibrium strategies, and they will be appropriate in different applied problems. One approach is to impose further conditions on the equilibrium concept, an alternative we explore by introducing a restricted EBE. A restricted EBE imposes conditions on the outcome of all feasible strategies from points in the recurrent class. In a restricted EBE any outcome from a feasible strategy that is in the recurrent class is consistent with the expected discounted value of future net cash that would be generated from equilibrium play from that outcome. We provide familiar examples of models used in applied work in which the players can construct consistent estimates of all the values required to satisfy the equilibrium conditions of a restricted EBE from knowledge of the primitives of the game and the outcomes from equilibrium play. Alternatively (or additionally), if data are available on either equilibrium play or the outcomes from that play, then that data and assumptions on the primitives of the game will suffice to restrict nonequilibrium play.

Weshowthat a restricted EBE that is consistent with a given set of primitives can be computed using a simple (reinforcement) learning algorithm. Moreover the equilibrium conditions are testable, and the testing procedure does not require computation of posterior distributions. Neither the iterative procedure which defines the computational algorithm nor the test of the equilibrium conditions have computational burdens which increase at a particular rate as we increase the number of variables that

impact on returns; neither is subject to a curse of dimensionality. At least in principle this should lead to an ability to analyze models that contain many more state variables, and hence are likely to be much more realistic, then could be computed using standard Markov perfect equilibrium concepts. 3

One could view our reinforcement learning algorithm as a description of how players learn the implications of their actions in a changing environment. This provides an alternative reason for interest in the output of the algorithm. However the learning rule would not, by iteself, restrict behavior without either repeated play or prior information on initial conditions. Also the fact that the equilibrium policies from our model can be learned from past outcomes accentuates the fact that those policies are most likely to provide an adequate approximation to the evolution of a game in which it is reasonable to assume that agents' perceptions of the likely returns to their actions can be learned from the outcomes of previous play. Since the states of the game evolve over time and the possible outcomes from each action differ by state, if agents are to learn to evaluate these outcomes from prior play the game needs to be confined to a finite space.

When all the state variables are observed by all the agents, our equilibrium notion is similar to but weaker than the familiar notion of Markov perfect equilibrium as used in Maskin and Tirole (1988, 2001). This is because we only require that the evaluations of outcomes used to form strategies be consistent with competitors' play when that play results in outcomes that are in the recurrent subset of points, and hence are observed repeatedly. We allow for feasible outcomes that are not in the recurrent class, but the conditions we place on the evaluations of those outcomes are weaker; they need only satsify inequalities that ensure they are not observed repeatedly. In this sense our notion of equilibrium is akin to the notion of self-confirming

3. For alternative computational procedures seethereviewinDoraszelskiand Pakes (2007). Pakes and McGuire (2001), show that reinforcement learning has significant computational advanatages when applied to full information dynamic games, a fact which has been used in several applied papers; for example, Goettler, Parlour, and Rajan (2005) and Beresteanu and Ellickson (2006). Goettler, Parlour, and Rajan (2008), use it to approximate optimal behavior in finance applications. Weshowthat a similar algorithm can be used in games with asymmetric information and provide a test of the equilibrum conditions which is not subject to a curse of dimensionality. The test in the original Pakes and McGuire article was subject to such a curse and it made their algorithm impractical for large problems.

equilibrium, as defined by Fudenberg and Levine (1993) (though our application is to dynamic games). An implication of using the weaker equilibrium conditions is that we might admit more equilibria than the Markov perfect concept would. The restrictions used in the restricted EBE reduce the number of equilibria.

The original Maskin and Tirole (1988) article and the framework for the analysis of dynamic oligopolies in Ericson and Pakes (1995) laid the groundwork for the applied analysis of dynamic oligopolies with symmetric information. This generated large empirical and numerical literatures on an assortment of applied problems (see Benkard 2004 or Gowrisankaran and Town 1997 for empirical examples and Doraszelski and Markovich 2007 or Besanko et al. 2010 for examples of numerical analysis). None of these models have allowed for asymmetric information. Our hope is that the introduction of asymmetric information in conjunction with our equilibrium concept helps the analysis in two ways. It enables the applied researcher to use more realistic behavioral assumptions and hence provide a better approximation to actual behavior, and it simplifies the process of analyzing such equilibria by reducing its computational burden.

As noted, this approach comes with its own costs. First, it is most likely to provide an adequate approximation to behavior in situations for which there is a relevant history to learn from. Second, our equilibrium conditions enhance the possiblity for multiple equilibria over more standard notions of equilibria. With additional assumptions one might be able to select out the appropriate equilibria from data on the industry of interest, but there will remain the problem of chosing the equilibria for counterfactual analysis.

Weconclude with an example that endogenizes the maintenance decisions of electricity generators. We take an admittedly simplified set of primitives and compute and compare equilibria based on alternative institutional constraints. These include asymmetric information equilibria where there are no bounds on agent memory, asymmetric information equilibria where there are such bounds, symmetric information equilibria, and the solutions to the social planner problem in two environments-one with more capacity relative to demand than the other. We show that in this environment the extent of excess capacity relative to demand has economically significant effects on equilibrium outcomes.

The next section describes the primitives of the game. Section II provides a definition of and sufficient conditions for our notion of an EBE. Section III provides an algorithm to compute and test for this equilibrium, and Section IV contains our example.

## II. Dynamic Oligopolies with Asymmetric Information

We extend the framework in Ericson and Pakes (1995) to allow for asymmetric information. 4 In each period there are nt potentially active firms, and we assume that with probability one nt /C20 /C22 n &lt; 1 (for every t ). Each firm has payoff relevant characteristics. Typically these will be characteristics of the products marketed by the firm or determinants of their costs. The profits of each firm in every period are determined by their payoff relevant random variables, a subset of the actions of all the firms, and a set of variables which are common to all agents and account for common movements in factor costs and demand conditions, say, d 2 D , where D is a finite set. For simplicity we assume that dt is observable and evolves as an exogenous first-order Markov process.

The payoff relevant characteristics of firm i , denoted by ! i 2 /C10 i , take values on a finite set of points for all i . There will be two types of actions-actions that will be observed by the firm's competitors m o i , and those that are unobserved m u i . For simplicity we assume that both take values on a finite state space, so mi ¼ ð m o i , m u i Þ 2 M i . 5 Notice that, also for simplicity, we limit ourselves to the case where an agent's actions are either known only to itself (they are ''private'' information), or to all agents (they are ''public'' information). For example, in an investment game the prices the firm sets are typically observed, but the investments a firm makes in the development of its products may not be. Though both controls could affect current profits or the

4. Indeed our assumptions nest the generalizations to Ericson and Pakes (1995), and the amendments to it introduced in Doraszelski and Satterthwaite (2010), and reviewed in Doraszelski and Pakes (2007). The latter paper also provide more details on the underlying model.

5. As in Ericson and Pakes (1995), we could have derived the assumption that /C10 and M are boundedsetsfrommoreprimitiveconditions.Alsotheoriginalversion of this paper (which is available on request) included both continuous and discrete controls, where investment was the continuous control. It was not observed by agent's opponents and affected the game only through its impact on the transition probabilities for ! .

probability distribution of payoff relevant random variables, they need not. A firm might simply decide to disclose information or send a signal of some other form.

Letting i index firms, realized profits for firm i in period t are given by

<!-- formula-not-decoded -->

where /C25 ð/C1Þ : /C2 n i ¼ 1 /C10 i /C2 n i ¼ 1 Mi /C2 D ! R . ! i , t evolves over time and its conditional distribution may depend on the actions of all competitors, that is

<!-- formula-not-decoded -->

Some examples will illustrate the usefulness of this structure.

A familiar special case occurs when the probability distribution of ! i , t þ 1, or P ! ð : j mi , m /C0 i , ! Þ , does not depend on the actions of a firm's competitors, or m /C0 i . Then we have a ''capital accumulation'' game. For example in the original Ericson and Pakes (1995) model, m had two components, price and investment, and ! consisted of characteristics of the firm's product or its cost function that the firm was investing to improve. Their ! i , t þ 1 ¼ ! i , t þ /C17 i , t /C0 dt , where /C17 i , t was a random outcome of the firm's investment whose distribution was determined by P ! ð/C1j mi , t , ! i , t Þ , and dt was determined by aggregate costs or demand conditions.

Now consider a sequence of timber auctions with capacity constraints for processing the harvested timber. Each period there is a new lot up for auction, firms submits bids (a component of our mi ), and the firm that submits the highest bid wins. The quantity of timber on the lot auctionned may be unknown at the time of the auction but is revealed to the firm that wins the lot. The firm's state (our ! i ) is the amount of unharvested timber on the lots the firm owns. Each period each firm decides how much to bid on the current auction (our first component of mi ) and how much of its unharvested capacity to harvest (a second component of mi which is constrained to be less than ! i ). The timber that is harvested and processed is sold on an international market which has a price that evolves exogenously (our f dt g process), and revenues equal the amount of harvested timber times this price. Then the firm's stock of unharvested timber in t +1, our ! i , t þ 1 is ! i , t minus the harvest during period t plus the amount on lots for

which the firm won the auction. The latter, the amount won at auction, depends on m /C0 i , t , that is, the bids of the other firms, as well as on mi , t .

Finally, consider a market for durable goods. Here we must explicitly consider both consumers and producers. Consumers are differentiated by the type and vintage of the good they own and their characteristics, which jointly define their ! i , and possibly by information they have access to which might help predict future prices and product qualities. Each period the consumer decides whether to buy a new vintage and if so which one (a consumer's mi )-a choice that is a determinant of the evolution of their ! i . Producers determine the price of the product marketed and the amount to invest in improving their product's quality (the components of the producer's mi ). These decisions are a function of current product quality and its own past sales (both components of the firm's ! i ), as well as other variables that affect the firm's perceptions about demand conditions. Since the price of a firm's competitors will be a determinant of the firm's sales, this is another example where the evolution of the firm's ! i , t þ 1 depends on m /C0 i , t as well as on mi , t .

The information set of each player at period t is, in principle, the history of variables that the player has observed up to that period. We restrict ourselves to a class of games in which each agent's strategies are a mapping from a subset of these variables, in particular from the variables that are observed by the agent and are either ''payoff'' or ''informationally'' relevant, where these two terms are defined as follows. The ''payoff relevant'' variables are defined, as in Maskin and Tirole (2001), to be those variables that are not current controls and affect the current profits of at least one of the firms. In terms of equation (1), all components of ð ! i , t , ! /C0 i , t , dt Þ that are observed are payoff relevant. Observable variables that are not payoff relevant will be informationally relevant if and only if either (1) even if no other agent's strategy depends on the variable player i can improve its expected discounted value of net cash flows by conditioning on it, or (2) even if player i 's strategy does not condition on the variable there is at least one player j whose strategy will depend on the variable. For example, say all players know ! j , t /C0 1 but player i does not know ! j , t . Then even if player j does not condition its strategy on ! j , t /C0 1, since ! j , t /C0 1 can contain information on the distribution of the payoff relevant ! j , t which, in turn, will affect /C25 i , t ð/C1Þ through

its impact on mj , t , player i will generally be able to gain by conditioning its strategy on that variable. 6

As before, we limit ourselves to the case where information is either known only to a single agent (''private'') or to all agents (''public''). The publicly observed component will be denoted by /C24 t 2 /C10 ð /C24 Þ , while the privately observed component will be zi , t 2 /C10 ð z Þ . For example ! j , t /C0 1 may or may not be known to agent i at time t ; if it is known ! j , t /C0 1 2 /C24 t , otherwise ! j , t /C0 1 2 zj , t . Since the agent's information at the time actions are taken consists of Ji , t ¼ ð /C24 t , zi , t Þ 2 J i , we assume strategies are functions of Ji , t , that is,

<!-- formula-not-decoded -->

Notice that if ! j , t is private information and affects the profits of firm i then we will typically have /C25 i , t 2 zi , t .

Weuseour examples to illustrate. We can embed asymmetric information into the original Ericson and Pakes (1995) model by assuming that ! i , t has a product quality and a cost component. Typically quality would be publicly observed, but the cost would not be and so becomes part of the firm's private information. Current and past prices are also part of public information set and contain information on the firms' likely costs, while investment may be public or private. In the timber auction example, the stock of unharvested timber is private information, but the winning bids (and possibly all bids), the published characteristics of the lots auctioned, and the marketed quantities of lumber are public information. In the durable good example the public information is the history of prices, but we need to differentiate between the private information of consumers and that of producers. The private information of consumers consists of the vintage and type of the good it owns and its own characteristics, whereas the firm's private information includes the quantities it sold in prior periods and typically additional information whose contents will depend on the appropriate institutional structure.

Throughout we only consider games where both #/C10 ð /C24 Þ and #/C10 ð z Þ are finite . This will require us to impose restrictions on the

6. Note that these defintions will imply that an equilibrium in our restricted strategyspacewillalsobeanequilibriuminthegeneralhistory-dependentstrategy space.

structure of informationally relevant random variables, and we come back to a discussion of situations in which these restrictions are appropriate later. To see why we require these restrictions, recall that we want to let agents base decisions on past experience. For the experience to provide an accurate indication of the outcomes of policies we will need a visit a particular state repeatedly; a condition we can only insure when there is a finite state space.

## III. Experience-Based Equilibrium

This section is in two parts. We first consider our basic equilibrium notion and then consider further restrictions on equilibrium conditions that will sometimes be appropriate.

For simplicity we assume all decisions are made simultaneously so there is no subgame that occurs within a period. In particular we assume that at the beginning of each period there is a realization of random variables and players update their information sets. Then the players decide simultaneously on their policies. The extension to multiple decisions nodes within a period is straightforward.

Let s combine the information sets of all agents active in a particular period, that is, s ¼ ð J 1 , . . . , Jn Þ when each Ji has the same public component /C24 . We say that Ji ¼ ð zi , /C24 Þ is a component of s if it contains the information set of one of the firms whose information is combined in s . We can write s more compactly as s ¼ ð z 1 , . . . , zn , /C24 Þ . So S ={ s : z 2 /C10 ( z ) n , x 2 /C10 ( x ), for 0 /C20 n /C20 /C22 n } lists the possible states of the world.

Firms' strategies in any period are a function of their information sets, so they are a function of a component of that period's s . From equation (2) the strategies of the firms determine the distribution of each firm's information set in the next period, and hence together the firms' strategies determine the distribution of the next period's s . As a result any set of strategies for all agents at each s 2 S , together with an initial condition, defines a Markov process on S .

We have assumed that S is a finite set. As a result each possible sample path of any such Markov process will, in finite time, wander into a subset of the states in S , say R /C26 S , and once in R stay within it forever. R could equal S but typically will not, as the strategies the agents choose will often ensure that some

states will not be visited repeatedly, a point we return to later. 7 R is referred to as a recurrent class of the Markov process as each point in R will be visited repeatedly.

Note that this implies that the empirical distribution of next period's state given any current s 2 R will eventually converge to a distribution, and this distribtuion can be constructed from actual outcomes. This will also be true of the relevant marginal distributions, for example, the joint distribution of the Ji components of s that belong to different firms, or that belong to the same firm in adjacent time periods. We use a superscript e to designate these limiting empirical distributions, so p e ð J 0 i j Ji Þ for Ji /C26 s 2 R provides the limit of the empirical frequency that firm i 0 s next period information set is J 0 i conditional on its current information being Ji 2 R and so on. 8

We now turn to our notion of EBE. It is based on the notion that at equlibrium players expected value of the outcomes from their strategies at states which are visited repeatedly are consistent with the actual distribution of outcomes at those states. Accordingly the equilibrium conditions are designed to ensure that at such states (1) strategies are optimal given participants' evaluations, and (2) that these evaluations are consistent with the empirical distribution of outcomes and the primitives of the model.

Notice that this implies that our equilibrium conditions could, at least in principle, be consistently tested. 9 To obtain a consistent test of a condition at a point we must, at least potentially, observe that point repeatedly. So we could only consistently test for conditions at points in a recurrent class. As we shall see this implies that our conditions are weaker than ''traditional'' equilibrium conditions. We come back to these issues, and

7. Freedman (1983) provides a precise and elegant explanation of the properties of Markov chains used here. Though there may be more than one recurrent class associated with any set of policies, if a sample path enters a particular R , a point, s , will be visited infinitely often if and only if s 2 R .

8. Formally the empirical distribution of transitions in R will converge to a Markovtransitionmatrix,say, p e , T /C17 f p e ð s 0 j s Þ : ð s 0 , s Þ 2 R 2 g . Similarly the empirical distribution of visits on R will converge to an invariant measure, say, p e , I /C17 f p e ð s Þ : s 2 Rg . Both p e , T and p e , I are indexed by a set of policies and a particular choice of a recurrent class associated with those policies. Marginal distributions for components of s are derived from these objects.

9. We say ''in principle'' here because this presumes that the researcher doing the testing can access the union of the information sets available to the agents that played the game.

their relationship to past work, after we provide our definition of equilibrium.

DEFINITION 1. An EBE consists of

- . a subset R/C26 S ;
- . strategies m /C3 ð Ji Þ for every Ji which is a component of any s 2 S ;
- . expected discounted value of current and future net cash flow conditional on the decision mi , say, W ð mi j Ji Þ , for each mi 2 M i and every Ji which is a component of any s 2 S ,

such that

CONDITION 1 ( R is a recurrent class). The Markov process generated by any initial condition s 0 2 R , and the transition kernel generated by f m /C3 g , has R as a recurrent class (so, with probability 1, any subgame starting from an s 2 R will generate sample paths that are within R forever).

CONDITION 2 (Optimality of strategies on R ). For every Ji which is a component of an s 2 R , strategies are optimal given W ð/C1Þ , that is m /C3 ð Ji Þ solves

<!-- formula-not-decoded -->

CONDITION 3 (Consistency of values on R ). Take every Ji which is a component of an s 2 R . Then

X

<!-- formula-not-decoded -->

where

/C26

X

/C0

/C1

<!-- formula-not-decoded -->

and

/C27

/C26

/C27

<!-- formula-not-decoded -->

Note that the evaluations f W ð mi j Ji Þg need not be correct for Ji not a component of an s 2 R . Nor do we require correctness of

the evaluations for the W ð mi j Ji Þ 's associated with points in R but at policies that differ from those in m /C3 i . The only conditions on these evaluations are that chosing an mi 6¼ m /C3 i would lead to a perceived evaluation which is less than that from the optimal policy (this is ensured by condition C 2). 10 On the other hand, the fact that our equilibrium conditions are limited to conditions on points that are played repeatedly implies that agents are able to learn the values of the outcomes from equilibrium play, and weprovide an algorithm that would allow them to form consistent estimates of those outcomes. Further comments on our equilibrium notion follow.

Beliefs on Types. Note also that our conditions are not formulated in terms of beliefs about either the play or the ''types'' of opponents. There are three reasons for this to be appealing. First, as beliefs are not observed, they can not be directly tested. Second, as we will show presently, it implies that we can compute equilibria without ever explicitly calculating posterior distributions. Finally (and relatedly) we will show that an implication of the equilibrium conditions is that agent's can chose optimal strategies based on the agent's own observable experience; indeed the agents need not even know all the primitive parameters of the game they are playing.

Relationship to Self-Confirming Equilibria. EBE, though formulated for dynamic games, is akin to the notion of self-confirming equilibria (Fudenberg and Levine 1993), which has been used in other contexts. 11 Self-confirming equilibria weaken the standard Nash equilibrium conditions. It requires that each player has beliefs about opponents' actions and that the player's actions are best responses to those beliefs. However the players' beliefs need only be correct along the equilibrium path. This ensures that no players observes actions which contradicts its beliefs. Our equilibrium conditions explicitly introduce the evaluations that the agents use to determine their actions. They are similar to the conditions of self-confirming equilibria in that the most they ensure is that these evaluations are consistent with the opponents actions along the equilibrium path. However, we

10. The fact that our conditions do not apply to points outside of R or to mi 6¼ m /C3 i implies that the conditional probabilities in equation (3) are well defined.

11. See also Dekel, Fudenberg, and Levine (2004) for an anlysis of self-confirming equilibrium in games with asymmetric information.

distinguish between states that are repeated infinitely often and those that are not, and we do not require the evaluations which determine actions at transitory states to be consistent with the play of a firm's opponents.

Boundary Points. It is useful to introduce a distinction made by Pakes and McGuire (2001). They partition the points in R into interior and boundary points. Points in R at which there are feasible (though inoptimal) strategies which can lead to a point outside of R are labeled boundary points. Interior points are points that can only transit to other points in R no matter which of the feasible policies are chosen (equilibrium or not). At boundary points there are actions which lead to outcomes which can not be consistently evaluated by the information generated by equilibrium play. This because our EBE notion does not restrict perceptions of returns from actions m 6¼ m /C3 for Ji /C26 s 2 R .

Multiplicity. Notice that Bayesian perfect equilibria will satisfy our equilibrium conditions, and typically there will be a multiplicity of such equilibria. Since our EBE notion does not restrict perceptions of returns from actions not played repeatedly, it will admit an even greater multiplicity of equilibria. There are at least two ways to select out a subset of these equilibria. One is to impose further conditions on the definition of equilibrium, an alternative that we explore in the next subsection. As explained there, this requires a game form which enables agents to acquire information on outcomes from nonequilibrium play.

Alternatively (or additionally) if data are available we could use it to restrict the set of equilibria. If we observe or can estimate a subset of either f W ð/C1Þg or f m /C3 ð/C1Þg we can restrict any subsequent analysis to be consistent with their values. In particular since there are (generically) unique equilibrium strategies associated with any given equilibrium f W ð/C1Þg , if we were able to determine the f W ð/C1Þg associated with a point (say, through observations on sample paths of profits) we could determine m /C3 i at that point, and conversely if we know m /C3 i at a point we can restrict equilibrium f W ð/C1Þg at that point. Similarly we can direct the computational algorithm we are about to introduce to compute an equilibria that is consistent with whatever data are observed. On the other hand were we to change a primitive of the model we could not single out the equlibria that is likely to result without further

assumptions (though one could analyze likely counterfactual outcomes if one is willing to assume a learning rule and an initial condition; see Lee and Pakes 2009).

## III.A. Restricted EBE

Our condition (3) only requires correct evaluations of outcomes from equilibrium actions that are observed repeatedly; that is, for W ð mi j Ji Þ at mi ¼ m /C3 i and Ji /C26 s 2 R . There are circumstances when imposing restrictions on equilibrim evaluations of actions off the equilibrium path for states that are observed repeatedly, that is, at mi 6¼ m /C3 i for Ji /C26 s 2 R , might be natural, and this subsection explores them.

Barring compensating errors, for agents to have correct evaluations of outcomes from an mi 6¼ m /C3 i they will need to know (1) expected profits and the distribution of future states that result from playing mi , and (2) the continuation values from the states that have positive probability when mi is chosen. Whether or not agents can obtain the information required to compute expected profits and the distribution of future states when an mi 6¼ m /C3 i is played depends on the details of the game, and we discuss this further later. For now we assume that they can, and consider what this implies for restricting the evaluations of outcomes from nonoptimal actions.

Consider strengthening the condition (3) to make it apply to all mi 2 M i at any Ji /C26 s 2 R . Then, at equilibrium, all outcomes that are in the recurrent class are evaluated in a way that would be consistent with the expected discounted value of returns that the action would yield were all agents (including itself) to continue playing their equilibrium strategies; this regardless of whether the action that generated the outcome was an equilibrium action. As in an unrestricted EBE outcomes that are not in the recurrent class are evaluated by perceptions which are not required to be consistent with any observed outcome. 12 As a result the restricted EBE ensures that in equilibrium when agents are at interior points they evalute all feasible actions in a

12. We note that there are cases where it would be natural to require outcomes not in the recurrent class to be consistent with publicly available information on primitives. Forexample,evenifafirmneverexitedfromaparticularstatetheagent might know its selloff value (or a bound on that value), and then it would be reasonabletorequirethattheactionofexitingbeevaluatedinawaythatisconsistentwith that information. It is straightforward to impose such constraints on the computational algorithm introduced in the next section.

way that is consistent with expected returns given equilibrium play. However at boundary points only those actions whose outcomes are in the recurrent class with probability 1 are evaluated in this manner.

DEFINITION 2 (Restricted EBE). Let /C25 E ð mi , Ji Þ be expected profits and f p ð J 0 i j Ji , mi Þg J 0 i be the distribution of J 0 , both conditional on ð mi , Ji Þ and m /C3 /C0 i . A restricted EBE requires, in addition to C1 and C2, that

X

<!-- formula-not-decoded -->

for all mi 2 M i and Ji /C26 s 2 R .

Weshowhowtocompute and test for a restricted EBE in the next section. We now point out one of the implications of this definition and then consider situations that enable agents to acquire the information required to consistently evaluate W ð mi j Ji Þ , for mi 6¼ m /C3 i , and Ji /C26 s 2 R .

Note that in some cases this equilibrium concept imposes a strong restriction on how agents react to nonequilibrium play by their competitors. To see this recall that the outcome is J 0 i ¼ ð /C24 0 , z 0 i Þ , where /C24 0 contains new public, and z 0 i new private, information. Competitors observe /C24 0 and /C24 . Were an agent to play an mi 6¼ m /C3 i it may generate a /C24 0 which is not in the support of the distribution of /C24 0 generated by ð /C24 , m /C3 i Þ . 13 Then if we impose the restrictions in equation (4) we impose constraints on the agent's evaluations of outcomes of actions which the agent's competitors would see as inconsistent with their experience from previous play. For the agent to believe such estimates are correct, the agent would have to believe that the competitor's play would not change were the competitor to observe an action off the equilibrium path. An alternative would be to assume that in equilibrium, agents only need to have correct evaluations for the outcomes of actions that competitor's could see as consistent with equilibrium play; that is, actions that generate a support for /C24 0 which is contained in the support /C24 0 conditional on ð /C24 , m /C3 i Þ . Then we would only restrict equilibrium beliefs about outcomes from

13. As an example consider the case where mi is observable. Then were the agent to play ~ mi 6¼ m /C3 i , ~ mi would be in /C24 0 and, provided there does not exist a ~ Ji ¼ ð /C24 , ~ zi Þ such that m /C3 ð /C24 , ~ zi Þ ¼ ~ mi , the support of /C24 0 given ð /C24 , ~ mi Þ will differ from that given ð /C24 , m /C3 i Þ .

actions that no agent perceives as inconsistent with equilibrium play. We do not pursue this further here, but one could modify the computational algorithm to accomodate this definition of a restricted EBE rather than the one in equation (4).

As noted for agents to be able to evaluate actions in a manner consistent with the restricted EBE they must know /C25 E ð mi , Ji Þ and f p ð J 0 i j Ji , mi Þg J 0 i for mi 6¼ m /C3 i at all Ji /C26 s 2 R . We now consider situations in which these objects can be computed from the information generated by equilibrium play or knowledge of the primitives of the problem. 14 The reader who is not interested in these details can proceed directly to the next section.

We consider a case where /C25 E ð mi , Ji Þ can be consistently estimated 15 , and investigate situations in which the agent can calculate W ð mi j Ji Þ , 8 mi 2 M i . To compute W ð mi j Ji Þ the agent has to be able to evaluate p ð J 0 i j Ji , mi Þ /C17 p ð /C24 0 j z 0 i , Ji , mi Þ p ð z 0 i j Ji , mi Þ , 8 J 0 i in the support of ð mi , Ji Þ , mi 2 M i and Ji /C26 s 2 R . When the required probabilities can be evaluated, the W ð m j Ji Þ calculated need only be ''correct'' if the support of f p ð J 0 i j Ji , mi Þg is in the recurrent class.

Consider a capital accumulation game in which the investment component of mi , say, mI , i is not observed but the pricing component, say, mP , i is observed, and assume prices are set before the outcome of the current investments is known. If zi represent costs which is private information then p ð /C24 0 j Ji , mi Þ ¼ p ð /C24 0 j Ji , mP , i Þ . Assume also that { z } evolves as a controled Markov process, so that p ð z 0 i j Ji , mi Þ ¼ p ð z 0 i j zi , mI , i Þ , and is known from the primitives of the cost-reducing process. Since costs are not observed and are a determinant of prices, past prices are informationally relevant (they contain information on current costs).

14. Note that even if agents can access the required information, to evaluate actions in the way assumed in a restricted EBE they will have to incur the cost of storing additional information and making additional computations-a cost we return to in the context of the computational algorithm discussed in the next section.

15. Whether /C25 E ð mi 6¼ m /C3 i , Ji Þ can be consistently estimated depends on the specifics of the problem, but it frequently can be. For a simple example consider an investment game where the profit function is additively separable in the cost of investment or mi , so that /C25 E ð mi , Ji Þ ¼ /C25 E ð m /C3 i , Ji Þ þ m /C3 i /C0 mi . If profits are not additively separable in mi but mi is observed then it suffices that agents be able to compute profits as a function of ð Ji , mi , m /C0 i Þ , as in the computational example below and in differentiated product markets in which the source of assymetric information is costs, equilibrium is Nash in prices, and agents know the demand function. In auctions the agent can compute /C25 E ð mi , Ji Þ if the agent can learn the distribution of the winning bid.

In this model p ð J 0 i j Ji , mi Þ ¼ p ð /C24 0 j Ji , mP , i Þ p ð z 0 i j zi , mI , i Þ . Since /C24 0 is set by the firm's decision on mP , i and p ð z 0 i j zi , mI , i Þ is known, the agent will always be able to evalute W ð mi j Ji Þ , 8 mi 2 M i . If mP , i ¼ m /C3 P , i then these evaluations will be correct if the support z 0 i given ð zi , mI , i Þ is in the support of ð zi , m /C3 I , i Þ , since then all J 0 with positive probability will be in the recurrent class. If the support condition is met but mP , i 6¼ m /C3 P , i then W ð mI , i , mP , i 6¼ m /C3 P , i j Ji Þ will be correct if there is a ð ~ zi , /C24 Þ /C26 s 2 R with the property that the optimal price at that point is mP , i , i.e. m /C3 P , i ð ~ zi , /C24 Þ ¼ mP , i . 16

## III.B. The Finite State Space Condition

Our framework is restricted to finite state games. We now consider this restriction in more detail. We have already assumed that there was (1) an upper bound to the number of firms simultaneously active, and (2) each firm's physical states (our ! ) could only take on a finite set of values. These restrictions ensure that the payoff relevant random variables are finite dimensional, but they do not guarantee this for the informationally relevant random variables, so optimal strategies could still depend on an infinite history. 17 We can ensure that the informationally relevant random variables are finite dimensional either through restrictions on the form of the game, or by imposing constraints on the cognitive abilities of the decision makers.

One example of a game form that can result in a finite dimensional space for the informationally relevant state variables is when there is periodic simultaneous revelation of all variables which are payoff relevant to all agents. Claim 1 of Appendix 1 shows that in this case an equilibrium with strategies restricted to depend on only a finite history is an equilibrium to the game with unrestricted strategies. Claim 2 of Appendix 1 shows that there is indeed a restricted EBE for the game with periodic revelation of information. The numerical analysis in Section IV includes an example in which regulation generates such a structure. Periodic revelation of all information can also result from

16. If the agents did not know the form of the underlying controlled Markov process a priori, it may be estimable using the data generated by the equilibrium process.

17. The conditions would however ensure finiteness in a game with asymmetric information where the sources of asymmetric information are distributed independently over time (as in Bajari, Benkard, and Levin 2007, or Pakes, Ostrovsky, and

Berry 2007).

situations in which private information can seep out of firms (say, through labor mobility) and will periodically do so for all firms at the same time, or when the equilibrium has one state which is visited repeatedly at which the states of all players are revealed.

There are other game forms which ensure finiteness. One example is when the institutional structure ensures that each agent only has access to a finite history. For example, consider a sequence of Internet auctions, one every period, for different units of a particular product. Potential bidders enter the auction site randomly and can only bid at finite increments. Their valuation of the object is private information, and the only additional information they observe are the sequence of prices that the product sold at while the bidder was online. If, with probability 1, no bidder remains on the site for more than L auctions, prices more than L auctions in the past are not in any bidder's information set, and hence cannot effect bids. 18 Alternatively a combination of assumptions on the functional forms for the primitives of the problem and the form of the interactions in the market that yield finite dimensional sufficient statistics for all unknown variables could also generate our finite state space condition.

A different way to ensure finiteness is through bounded cognitive abilities, say, through a direct bound on memory (e.g., agents cannot remember what occured more than a finite number of periods prior), or through bounds on complexity, or perceptions. There are a number of reasons such a restriction may be appealing to empirical researchers. First it might be thought to be a realistic approximation to the actual institutions in the market. Second in most applications the available data is truncated so the researcher does not have too long a history to condition on. Moreover in any given application one could investigate the extent to which policies or outcomes depended on particular variables either empirically or computationally.

To illustrate, our computational example computes equilibria to finite state games generated by both types of assumptions. One of the questions we address is whether the different assumptions we use to obtain finiteness, all of which seem a priori reasonable, generate equilibria with noticeably different policies.

18. Formally this example requires an extension of our framework to allows for state variables that are known to two or more, but not to all, agents.

## IV. An Algorithm for Computing an EBE

This section shows that we can use a reinforcement learning algorithm to compute an EBE. As a result our equilibria can be motivated as the outcome of a learning process. In the reinforcement learning algorithm players form expectations on the value that is likely to result from the different actions available to them and choose their actions optimally given those expectations. From a given state, those actions, together with realizations of random variables whose distributions are determined by them, lead to a current profit and a new state. Players use this profit together with their expectations of the value they assign to the new state to update their expectation of the continuation values from the starting state. They then proceed to chose an optimal policy for the new state, a policy that maximizes its expectations of the values from that state. This process continues iteratively.

Note that the players' evaluations at any iteration need not be correct. However we would expect that if policies converge and we visit a point repeatedly we will eventually learn the correct continuation value of the outcomes from the policies at that point. Our computational mimic of this process includes a test of whether our equilibrium conditions, conditions that ensure that continuation evaluations are in fact consistent with subsequent play, are satisfied. We note that since our algorithm is a simple reinforcement learning algorithm, an alternative approach would have been to view the algorithm itself as the way players learn the values needed to choose their policies, and justify the output of the algorithm in that way. A reader who subscribes to the latter approach may be less interested in the testing subsection. 19

We begin with the iterative algorithm for an EBE, then note the modifications required for a restricted EBE, and then move on to the test statistic for both equilibrium concepts. A discussion of the properties of the algorithm, together with its relationship to the previous literature and additional details that can make implementation easier, is deferred until Appendix 2.

The algorithm consists of an iterative procedure and subroutines for calculating initial values and profits. We begin with the

19. On the other hand, there are several issues that arise were one to take the learning approach as an approximation to behavior, among them the question of whether(andhow)anagentcanlearnfromtheexperienceofotheragents,and how much information an agent gains about its value in a particular state from experience in related states.

iterative procedure. Each iteration, indexed by k , starts with a location that is a state of the game (the information sets of the players) say, L k ¼ ½ J k 1 , . . . , J k n ð k Þ /C138 , and the objects in memory, say, M k ¼ f M k ð J Þ : J 2 J g . The iteration updates both these objects. We start with the updates for an unrestricted EBE, and then come back to how the iterative procedure is modified when computing a restricted EBE. The rule for when to stop the iterations consists of a test of whether the equilibrium conditions defined in the last section are satisfied, and we describe the test immediately after presenting the iterative scheme.

Memory. The elements of M k ð J Þ specify the objects in memory at iteration k for information set J , and hence the memory requirements of the algorithm. Often there will be more than one way to structure the memory with different ways having different advantages. Here we focus on a simple structure that will always be available (though not necessarily always efficient); alternatives are considered in Appendix 2.

M k ð Ji Þ contains

- . a counter, h k ð Ji Þ , which keeps track of the number of times we have visited Ji prior to iteration k , and if h k ð Ji Þ &gt; 0 it contains
- . W k ð mi j Ji Þ for mi 2 M i , i ¼ 1 , . . . , n .

If h k ð Ji Þ ¼ 0 there is nothing in memory at location Ji . If we require W ð/C1j Ji Þ at a Ji at which h k ð Ji Þ ¼ 0 we have an initiation procedure that sets W k ð mi j Ji Þ ¼ W 0 ð mi j Ji Þ . Appendix 2 considers choices of f W 0 ð/C1Þg . For now we simply note that high initial values tend to ensure that all policies will be explored.

Policies and Random Draws for Iteration k . For each J k i which is a component of L k call up W k ð/C1j J k i Þ from memory and choose m k ð J k i Þ to

<!-- formula-not-decoded -->

With this f m k ð J k i Þg use equation (1) to calculate the realization of profits for each active agent at iteration k (if d is random, then the algorithm has to take a random draw on it before calculating profits). These same policies, f m k ð J k i Þg , are then substituted into the conditioning sets for the distributions of the next period's state variables (the distributions in equation (2) for payoff relevant random variables and the update of informationally relevant

state variables if the action causes such an update), and they, in conjunction with the information in memory at L k , determine a distribution for future states (for f J k þ 1 i g ). A pseudo random number generator is then used to obtain a draw on the next period's payoff relevant states.

Updating. Use ð J k i , m k ð J k i Þ , ! k þ 1 i , d k þ 1 Þ to obtain the updated location of the algorithm

h

i

<!-- formula-not-decoded -->

To update the W it is helpful to define a ''perceived realization'' of the value of play at iteration k (that is, the perceived value after profits and the random draws are realized), or

<!-- formula-not-decoded -->

To calculate V k þ 1 ð J k i Þ we need to first find and call up the information in memory at locations f J k þ 1 i g nk þ 1 i ¼ 1 . 20 Once these locations are found we keep a pointer to them, as we will return to them in the next iteration.

For the intuition behind the update for W k ð/C1j J k i Þ note that were we to substitute the equilibrium W /C3 ð/C1j J k þ 1 i Þ and /C25 E ð/C1j J k i Þ for the W k ð/C1j J k þ 1 i Þ and /C25 k ð/C1j J k i Þ in equation (5) and use equilibrium policies to calculate expectations, then W /C3 ð/C1j J k i Þ would be the expectation of V /C3 ð/C1j J k i Þ . Consequently we treat V k þ 1 ð J k i Þ as a random draw from the integral determining W /C3 ð/C1j J k i Þ and update the value of W k ð/C1j J k i Þ as we do an average, for example,

!

<!-- formula-not-decoded -->

where m k i is the policy perceived to be optimal for agent i at iteration k . This makes W k ð J k i Þ the simple average of the V r ð J r i Þ over the iterations at which J r i ¼ J k i . Though use of this simple average will satisfy Robbins and Monroe's (1951) convergence conditions, we will typically be able to improve the precision of our estimates of the W ð/C1Þ by using a weighting scheme that

20. The burden of the search for these states depends on how the memory is structured, and the efficiency of the alternative possiblities depend on the properties of the problem analyzed. As a result we come back to this question when discussing our example.

downweights the early values of V r ð/C1Þ as they are estimated with more error than the later values. 21

Completing the Iteration. We now replace the W k ð/C1j J k i Þ in memory at location J k i with W k þ 1 ð/C1j J i k Þ (for i ¼ 1 , . . . , nk ) and use the pointers obtained above to find the information stored in memory at L k þ 1 . This completes the iteration as we are now ready to compute policies for the next iteration. The iterative process is periodically stopped to run a test of whether the policies and values the algorithm outputs are equilbirium policies and values. We come back to that test presently.

Updating When Computing a Restricted EBE. The algorithm just described only updates W k ð mi j Ji Þ for mi ¼ m k i , the policy that is optimal given iteration k 's evaluations. So this algorithm is unlikely to provide correct evaluations of outcomes from actions off the equilibrium path, and a restricted EBE requires correct evaluations of some of those outcomes (the outcomes in R ). To compute a restricted EBE we modify this algorithm to update all the f W k ð m j J k i Þg m 2M i , that is, the continuation values for all possible actions from a state whenever that state is reached. This ensures that whenever a nonequilibrium action has a possible outcome that is in the recurrent class, it will be evaluated correctly provided all recurrent class points are evaluated correctly.

To update W k ð mi j J k i Þ when mi 6¼ m k i we take a random draw from the distribution of outcomes conditional on that mi , use it and the random draws from the competing agent's optimal policies to form what the perceived value realization would have been had the agent implemented policy mi 6¼ m /C3 i (substitute mi for m k i in the defintion V k þ 1 ð J k i Þ in equation (5)), and use it to form W k þ 1 ð mi j J k i Þ (as in equation (6)). The rest of the algorithm is as above; in particular we update the location using the draws from

21. One simple and surprisingly effective way of doing so is to restart the algorithm using as starting values the values outputted from the first several million draws. The Robbins and Monroe (1951) article is often considered to have initiated thestochastic approximationliteratureofwhichreinforcementlearningisaspecial case. Their conditions on the weighting function are that the sum of the weights of eachpointvisited infinitely often must increase without bound while the sum of the weights squarred must remain bounded.

the optimal policy. Note that the algorithm to compute a restricted EBE is significantly more computationally burdensome then that for the unrestricted EBE (the computational burden at each point goes up by a factor of /C2 nk i ¼ 1 # M i nk ), and is likely to also increase the memory requirements.

## IV.A. Testing Whether the Output of the Algorithm Constitues an EBE or a Restricted EBE

Assume we have a W vector in memory at some iteration of the algorithm, say, W k ¼ ~ W , and we want to test whether ~ W generates an EBE on a recurrent subset of S . To perform the test we need to check our equilibrium conditions and this requires: (1) a candidate for a recurrent subset determined by ~ W , say, Rð ~ W Þ , and checks for both, (2) the optimality of policies and, (3) the consistency of ~ W , on Rð ~ W Þ .

To obtain a candidate for Rð ~ W ), start at any s 0 and use the policies implied by ~ W to simulate a sample path f s j g J 1 þ J 2 j ¼ 1 . Let Rð J 1 , J 2 , /C1Þ be the set of states visited at least once between j ¼ J 1 and j ¼ J 2. Provided J 1 , J 2, and J 1 /C0 J 2 grow large, R will become a recurrent class of the process generated by ~ W . In practice to determine whether any finite ð J 1 , J 2 Þ are large enough, one generates a second sample path starting at J 2 and continuing for another J 2 /C0 J 1 iterations. We then check to see that the set of points visited on the second sample path are the same as those in Rð J 1 , J 2 , /C1Þ .

The second equilibrium condition specifies that the policies must be optimal given ~ W . This is satisfied by construction as we chose the policies that maximize ~ W ð mi j Ji Þ at each Ji .

To check the third equilibrium condtion we have to check for the consistency of ~ W with outcomes from the policies generated by ~ W on the points in R . Formally we have to check for the equality in

X

<!-- formula-not-decoded -->

In principle we could check this by direct summation for the points in R . However this is computationally burdensome, and the burden increases exponentially with the number of possible states (generating a curse of dimensionality). So proceeding in this way would limit the types of empirical problems that could be analyzed.

A far less burdensome alternative, and one that does not involve a curse of dimensionality, is to use simulated sample paths for the test. To do this we start at an s 0 2 R and forward simulate. Each time we visit a state we compute perceived values, the V k þ 1 ð/C1Þ in equation (5), for each Ji at that state, and keep track of the average and the sample variance of those simulated perceived values across visits to the same state, say,

<!-- formula-not-decoded -->

Anestimate of the mean square error of ^ /C22 ð/C1Þ as an estimate of ~ W ð/C1Þ can be computed as ð ^ /C22 ð/C1Þ /C0 ~ W Þ 2 . The difference between this mean square error and the sampling variance, or ^ /C27 2 ð W ð m /C3 ð Ji Þj Ji ÞÞ , is an unbiased estimate of the bias squarred of ^ /C22 ð/C1Þ as an estimate of ~ W ð/C1Þ . We base our test of the third EBE condition on these bias estimates.

More formally if we let E ð/C1Þ take expectations over simulated random draws, l index information sets, and do all computations as percentages for each ~ Wl ð/C1Þ value, the expectation of our estimate of the percentage mean square of ^ /C22 ð Wl Þ as an estimate of ~ Wl is

!

<!-- formula-not-decoded -->

Let ð d MSEs , /C27 2 s , ð Biass Þ 2 Þ be the average of ð d MSEl , /C27 2 l , ð Biasl Þ 2 Þ over the information sets (the l ) of the agents active at state s , and ^ /C27 2 s be the analogous average of ^ /C27 2 ð Wl Þ ~ W 2 l . Then since ^ /C27 2 s is an unbiased estimate of /C27 2 s , the law of large numbers ensures that an average of the ^ /C27 2 s at different s converges to the same average of /C27 2 s . Let hs be the number of times we visit point s . We use as our test statistic, say, T , an hs weighted average of the difference between the estimates of the mean square and that of the variance, so if ! indicates (almost sure) covergence, the argument above implies that

X

X

X

<!-- formula-not-decoded -->

a weighted average of the sum of squares of the percentage bias. If T is sufficiently small we stop the algorithm; otherwise we continue. 22

Testing for a Restricted EBE. Our test for a restricted EBE is similar except that in the restricted case we simulate the mean and the variance of outcomes for every mi 2 M i for each information set l , say, ð ^ /C22 mi , l , ^ /C27 2 mi , l Þ , for each Jl /C26 s and s 2 R . We then use the analogue of equation (7) to derive estimates of f d MSEl , mi g and average over mi 2 M i to obtain new estimates of ð d MSEl , ^ /C27 2 l Þ . The test statistic is obtained by substituting these new estimates into the formula for T in equation (8) and will be labeled T R .

## V. Example: Maintenance Decisions in An Electricity Market

The restructuring of electricity markets has focused attention on the design of markets for electricity generation. One issue in this literature is whether the market design would allow generators to make super-normal profits during periods of high demand. In particular the worry is that the facts that currently electricity is not storable and has extremely inelastic demand might lead to sharp price increases in periods of high demand (for a review of the literature on price hikes and an empirical analysis of their sources in California during the summer of 2000, see Borenstein, Bushnell, and Wolak 2002). The analysis of the sources of price increases during periods of high demand typically conditions on whether generators are bid into or withheld from the market, though some of the literature have tried to incorporate the possiblity of ''forced,'' in constrast to ''scheduled,'' outages (see Borenstein, Bushnell, and Wolak 2002). Scheduled outages are largely for maintenance and maintenance decisions are difficult to incorporate into an equilibrium analysis because, as many authors have noted, they are endogenous. 23

22. Formally T is an L 2 ðPRÞ norm in the percentage bias, where PR is the invariant measure associated with ðR , ~ W Þ . Appendix 2 comments on alternative possible testing procedures, some of which may be more powerful than the test provided here.

23. There has, however, been an extensive empirical literature on when firms do maintenance (see, for example, Harvey, Hogan, and Schatzki 2004 and the literature reviewed their). Of particular interest are empirical investigations of

Since the benefits from incuring maintenance costs today depend on the returns from bidding the generator in the future, and the latter depend on what the firms' competitors bid at future dates, an equilibrium framework for analyzing maintenance decisions requires a dynamic game with strategic interaction. To the best of our knowledge maintenance decisions of electric utilities have not been analyzed within such a framework to date. Here we provide the details of a simple example that endogenizes maintenance decisions and then compute a restricted EBE for that example.

Overview of the Model. In our model the level of costs of a generator evolve on a discrete space in a nondecreasing random way until a maintenance decision is made. In the full information model each firm knows the current cost state of its own generators as well as those of its competitors. In the model with asymmetric information the firm knows the cost position of its own generators, but not those of its competitors.

In any given period firms can hold their generators off the market. Whether they do so is public information. They can, but need not, use the period they are shut down to do maintenance. If they do maintenance, the cost level of the generator reverts to a base state (to be designated as the zero state). If they do not do maintenance the cost state of the generator is unchanged. In the asymmetric information model whether a firm maintains a generator that is not bid into the market is private information.

If they bid the generator into the market, they submit a supply function and compete in the market. If the generator is bid in and operated its costs are incremented by a stochastic shock. There is a regulatory rule ensuring that the firms do maintenance on each of their generators at least once every six periods.

For simplicity we assume that if a firm submits a bid function for producing electricity from a given generator, it always submits the same function (so in the asymmetric information environment the only cost signals sent by the firm is whether it bids in each of its generators). We do, however, allow for heterogeneity in both cost and bidding functions across generators. In particular we allow for one firm that owns only big generators, Firm B, and one firm that only owns small generators, Firm S. Doing

the co-ordination of maintenance decsions, see, for example, Patrick and Wolak

(1997).

maintenance on a large generator and then starting it up is more costly than doing maintenance on a small generator and starting it up, but once operating the large generator operates at a lower marginal cost. The demand function facing the industry distinguishes between the five days of the work week and the two-day weekend, with demand higher in the work week.

In the full information case the firm's strategy are a function of; the cost positions of its own generators, those of its competitors, and the day of the week. In the asymmetric information case the firm does not know the cost position of its competitors' generators, though it does realize that its competitors' strategy will depend on those costs. As a result any variable that helps predict the costs of a competitors' generators will be informationally relevant.

In the asymmetric information model Firm B's perceptions of the cost states of Firm S's generators will depend on the last time each of Firm S's generators shut down. So the time of the last shutdown decision on each of Firm S's generators are informationally relevant for Firm B. Firm S's last shutdown and maintenance decisions depended on what it thought Firm B's cost states were at the time those decisions were made, and hence on the timing of Firm B's prior shutdown decisions. Consequently Firm B's last shutdown decisions will generally be informationally relevant for itself. As noted in the theory section, without further restrictions this recurrence relationship between one firm's actions at a point in time and the prior actions of the firm's competitors at that time can make the entire past history of shutdown decisions of both firms informationally relevant. Below we consider alternative restrictions each of which have the effect of truncating the relevant past history in a different way.

Social Planner and Full Information Problem. To facilitate efficiency comparisons we also present the results generated by the same primitives when (1) maintenance decisions are made by a social planner that knows the cost states of all generators, and (2) a duopoly in which both firms have access to the cost states of all generators (their own as well as their competitors, our ''full information'' problem). The planner maximizes the sum of the discounted value of consumer surplus and net cash flows to the firms. However since we want to compare maintenance decisions holding other aspects of the environment constant, when the planner decides to bid a generator into the market, we

constrain it to use the same bidding functions used in the competitive environments.

Since the social planner problem is a single-agent problem, we compute it using a standard contraction mapping. The equilibrium concept for the full information duopoly is Markov perfect and an equilibrium can be computed for it using techniques analogous to those used for the asymmetric information duopoly (see Pakes and McGuire 2001).

## V.A. Details and Parameterization of the Model

Firm B has two generators at its disposal. Each of them can produce up to 25 megawatts of electricity at a constant marginal cost which depends on their cost state ( mcB ð ! Þ ) and can produce higher levels of electricity at increasing marginal cost. Firm S has three generators at its disposal each of which can produce 15 megawatts of electricity at a constant marginal cost which depends on their cost state ( mcS ð ! Þ ) and higher levels at increasing marginal cost. So the marginal cost function of a generator of type k 2 f B , S g is as follows:

<!-- formula-not-decoded -->

where /C22 qB ¼ 25 and /C22 qS ¼ 15 and the slope parameter /C12 ¼ 10. For a given ! and level of production, Firm B's generator's marginal cost is smaller than those of Firm S at any cost state, but the cost of maintaining and restarting Firm B's generators is two and a half times that of Firm S's generators (see Table I).

The firms bid just prior to the production period and they know the cost of their own generators before they bid. If a generator is bid, it bids a supply curve that is identical to its highest marginal cost at which it can operate. The market supply curve is obtained by the horizontal summation of the individual supply curves. For the parameter values indicated in Table I, if Firm B bids in Nb number of generators and Firm S bids in Ns number of generators, the resultant market supply curve is:

8

<!-- formula-not-decoded -->

TABLE I PRIMITIVES THAT DIFFER AMONG FIRMS

| Parameter                                         | Firm B         | Firm S           |
|---------------------------------------------------|----------------|------------------|
| Number of generators                              | 2              | 3                |
| Range of !                                        | 0-4            | 0-4              |
| Marginal cost constant ( ! ¼ ð 0 , 1 , 2 , 3 Þ )* | (20,60,80,100) | (50,100,130,170) |
| Maximum capacity at constant MC                   | 25             | 15               |
| Costs of maintenance                              | 5,000          | 2,000            |

* At ! ¼ 4 the generator must shut down.

and supply is infinitely elastic at p =600. The $600 price cap is meant to mimic the ability of the independent system operator to import electricity when local market prices are too high.

The market maker runs a uniform price auction; it horizontally sums the generators' bid functions and intersects the resultant aggregate supply curve with the demand curve. This determines the price per megawatt hour and the quantities the two firms are told to produce. The market maker then allocates production across generators in accordance with the bid functions and the equilibrium price.

The demand curve is log-linear

<!-- formula-not-decoded -->

with a price elasticity of /C11 ¼ : 3. In our base case the intercept term Dd ¼ weekday ¼ 7 and Dd ¼ weekend ¼ 6 : 25. We later compare this to a case where demand is lower, Dd ¼ weekday ¼ 5 : 3 and Dd ¼ weekend ¼ 5 : 05, as we found different behavioral patterns when the ratio of production capacity to demand was higher.

As noted if the generator does maintenance then it can be operated in the next period at the low cost base state ( ! ¼ 0). If the generator is shut down but does not do maintenance its cost state does not change during the period. If the generator is operated the state of the generator stochastically decays. Formally if ! i , j , t 2 /C10 ¼ f 0 , 1 , . . . , 4 g is the cost state of firm i 's j th generator and it is operated in period t , then

<!-- formula-not-decoded -->

where /C17 i , j , t 2 f 0 , 1 g with each outcome having probability.5.

The information at the firm's disposal when it makes its shutdown and maintenance decisions, say, Ji , t , always includes the vector of states of its own generators, say, ! i , t ¼ f ! i , j , t ; j ¼ 1 . . . ni g 2 /C10 ni , and the day of the week (denoted by d 2 D ). In the full information model it also includes the cost states of its competitors' generators. In the asymmetric information case firms do not know their competitors' cost states and so keep in memory public information sources which may help them predict their competitors' actions. The specification for the public information used differs for the different asymmetric information models we run, so we come back to it when we introduce those models.

The strategy of firm i 2 f S : B g is a choice of

<!-- formula-not-decoded -->

where m =0 indicates the generator is shut down and not doing maintenance, m =1 indicates the generator is shut down and doing maintenance, and m =2 indicates the firm bids the generator into the market. The cost of maintenance is denoted by cmi , and if the firm bids into the market the bid function is the highest marginal cost curve for that type of generator. We imposed the constraint that the firm must do maintenance on a generator whose ! ¼ 4.

If p ð m 1 , t , m 2 , t , dt Þ is the market clearing price while yi , j , t ð mB , t , mS , t , dt Þ is the output alocated by the market maker to the j th generator of the i th firm, the firm's profits ( /C25 i ð/C1Þ ) are X

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where I f/C1g is the indicator function which is 1 if the condition inside the brackets is satisfied and 0 elsewhere, c ð ! i , j , t , yi , j , t ð/C1ÞÞ is the cost of producting output yi , j , t at a generator whose cost state is given by ! i , j , t , and cmj , i is the cost of maintenance (our ''investment'').

## V.B. Alternative Informational Assumptions for the Assymmetric Information Model

Wehavejust described the primitives and the payoff relevant random variables of the models we compute. We now consider the

different information sets that we allow the firm to condition on in those models. As noted the public information that is informationally relevant could, in principle, include all past shutdown decisions of all generators-those owned by the firm as well as those owned by the firms' competitors. To apply our framework we have to ensure that the state space is finite. We present the results from three different assumptions on the information structure of the asymmetric information model, each of which have the effect of ensuring finiteness. In addition we compare these results to both a full information model in which all generator's states are public information, and to those generated by a social planner that maximizes the sum of discounted consumer and producer surplus.

All three asymmetric information (AsI) models that we compute assume ð ! i , t , dt Þ 2 Ji , t . The only factor that differentiates the three is the public information kept in memory to help the firm assess the likely outcomes of its actions. In one case there is periodic full revelation of information; it is assumed that a regulator inspects all generators every T periods and announces the states of all generators just before period T +1. In this case we know that if one agent uses strategies that depend only on the information it has accumulated since the states of all generators were revealed, the other agent can do no better than doing so also. We computed the equilibria for this model for T =3, 4, 5, 6 to see the sensitivity of the results to the choice of T. The other two cases restrict the memory used in the first case; in one a firm partitions the history it uses more finely than in the other. In these cases it maywell be that the agents would have profitable deviations if we allowed them to condition their strategies on more information.

The public information kept in memory in the three AsI models is as follows.

- (1) In the model with periodic full revelation of information the public information is the state of all generators at the last date information was revealed, and the shutdown decisions of all generators since that date (since full revelation occurs every T periods, no more than T periods of shutdown decisions are ever kept in memory).
- (2) In finite history s the public information is just the shutdown decisions made in each of the last T periods on each generator.

- (3) In finite history /C28 the public information is only the time since the last shutdown decision of each generator.

The information kept in memory in each period in the third modelisafunctionofthatinthesecond;acomparisonoftheresults fromthesetwomodelsprovidesanindication onwhethertheextra informationkeptinmemoryinthesecondmodelhasanyimpacton behavior. The first model, with full revelation every six periods, is the only one whose equilibrium is ensured to be an equilibrium to the game where agents can condition their actions on the indefinite past. That is, there may be unexploited profit opportunties whenemploying the equilibrium strategies of the last two models. On the other hand the cardinality of the state space in the model with periodic full revelation of information is an order of magnitude larger than in either of the other two models. 24

## V.C. Computational Details

We compute a restricted EBE using the algorithm provided in Section III. The full information (FI) equilibrium is computed using analogous reinforcement learning algorithms (see Pakes and McGuire 2001), and the social planner is computed using a standard iterative technqiue (as it is a contraction mapping with a small state space). This section describes two model-specific details needed for the computation: (1) starting values for the W ð/C1j/C1Þ 's and the /C25 E ð/C1j/C1Þ , and (2) the information storage procedures.

To ensure experimentation with alternative strategies we used starting values which, for profits, were guaranteed to be higher than their true equilibrium values, and for continuation values, that we were quite sure would be higher. Our intitial values for expected profits are the actual profits the agent would receive were its competitor not bidding at all, or

<!-- formula-not-decoded -->

For the intial condition for the expected discounted values of outcomes given different strategies we assumed that the profits were the other competitor not producing at all could be

24. However, there is no necessary relationship between the size of the recurrent classes in the alternative models, and as a result no necessary relationship between either the computational burdens or the memory requirements of those models. The memory requirements and computational burdens generated by the different assumptions have to be analyzed numerically.

obtained forever with zero maintenance costs and no depreciation, that is,

<!-- formula-not-decoded -->

The memory was structured first by public information, and then for each given public information node, by the private information of each agent. We used a tree structure to order the public information and a hash table to allocate the private information conditional on the public information. To keep the memory manageable, every 50 million iterations we performed a ''clean-up'' operation that dropped all those points that were not visited at all in the last 10 million iterations.

## V.D. Computational Properties of the Results

The results reported below are from runs in which we ran the model 500 million iterations and then printed out test statistics for each firm. The test statistics gave us an R 2 /C25 1 (to five significant digits; for example, for T =5, the R 2 was .9995 and .9996 for Firms B and S, respectively). 25

Table II considers the sensitivity of the output from the AsI model with full revelation every T periods to the choice of T . As we increase T the difference in these variables' values becomes progressively smaller, with the difference between T =6 and T =5 not large enough to impact any of our conclusions. Consequently we focus on the T =5 case for all the rest of our calculations.

Next we asked how well we approximate the AsI model with periodic full revelation with our AsI models with restricted state spaces. Table III compares summary statistics from the full revelation model to models in which all a firm remembers about its competitors is (1) whether the competitors' generators were bid into the market in each of the last five periods (the finite history s information sturcture), or (2) the last time each of its competitors' generators was shut down (our finite history /C28 model). The table shows that the finite history /C28 information stucture does not approximate the periodic full revelation model well, but the finite history s structure does much better. Indeed, it would be

25. As a check of our programs, we also checked to see that our results from the program that computed the restricted EBE model also consituted an equilibrium for the program that ran the unrestricted EBE model, which it did.

TABLE II PERIODIC FULL REVELATION DIFFERENT T

| Summary statistics                        |   T=3 |   T=4 |   T=5 |   T=6 |
|-------------------------------------------|-------|-------|-------|-------|
| Consumer surplus ( /C2 10 /C0 3 ) 58,000+ | 550   | 572   | 581   | 580   |
| Profit B ( /C2 10 /C0 3 )                 | 393   | 389   | 384   | 383   |
| Profit S ( /C2 10 /C0 3 )                 | 334   | 324   | 322   | 324   |
| Maintenance cost B ( /C2 10 /C0 3 )       |  25.9 |  21.6 |  20.2 |  19.4 |
| Maintenance cost S ( /C2 10 /C0 3 )       |  12.1 |  11.8 |  11.8 |  11.8 |
| Production cost B ( /C2 10 /C0 3 )        | 230.2 | 235.3 | 235.1 | 234.3 |
| Production cost S ( /C2 10 /C0 3 )        | 230.4 | 226.9 | 228.1 | 229.2 |

hard to tell the difference from this and the periodic full revelation model with the kind of data sets we usually have. We use the output from the model with periodic full revelation in our analysis of results in the next subsection, but if we were to compute models with larger state spaces the finite history s model would become increasingly attractive.

We are interested in the models with the restricted information structures because they generate smaller state spaces and hence are likely to impose less of a computational burden both on the researcher, and perhaps more important, on the agents' actual decision-making process. For both these reasons the restricted information structures may be more suitable for applied work. Table IV provides the sizes of the recurrent classes and compute times for these models (including the test time). 26 We note that the compute times for the AsI models are the compute times for the restricted EBE. If one were to suffice with the weaker notion of an (unrestricted) EBE, the compute time would go down dramatically (e.g., its compute time per 100 million iterations for the periodic full revelation model when T =5 was just under two hours).

The first thing to notice from the table is that the compute time per 100 million iterations increases with the size of the recurrent class, though at a decreasing rate. The size of the

26. All computations in this article were run on the Odyssey cluster supported by the FAS Science Division Research Computing Group at Harvard University. For a description of the machine used see http//rc.fas.havard.edu. However the memory requirements for all runs was well within 1 GB, so the runs could be done on a laptop.

TABLE III

## THREE ASYMMETRIC INFORMATION MODELS (T=5)

|                                           | Finite history   | Finite history   | Periodic   |
|-------------------------------------------|------------------|------------------|------------|
| Summary statistics                        | /C28             | s                | Revelation |
| Consumer surplus ( /C2 10 /C0 3 ) 58,000+ | 270              | 580              | 581.5      |
| Profit B ( /C2 10 /C0 3 )                 | 414              | 384.7            | 384.5      |
| Profit S ( /C2 10 /C0 3 )                 | 439              | 323.5            | 322.8      |
| Maintenance cost B ( /C2 10 /C0 3 )       | 28.5             | 20.0             | 20.2       |
| Maintenance cost S ( /C2 10 /C0 3 )       | 18.0             | 11.7             | 11.8       |
| Production cost B ( /C2 10 /C0 3 )        | 226.8            | 235.5            | 235.1      |
| Production cost S ( /C2 10 /C0 3 )        | 254.6            | 228.4            | 228.1      |

TABLE IV COMPUTATIONAL COMPARISONS

|                                                                 | AsI; finite hist. /C28                                          | AsI; finite hist. s                                             | AsI; full revel.                                                | Full info.                                                      |
|-----------------------------------------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------|
| Compute times per 100 million iterations (hours; includes test) | Compute times per 100 million iterations (hours; includes test) | Compute times per 100 million iterations (hours; includes test) | Compute times per 100 million iterations (hours; includes test) | Compute times per 100 million iterations (hours; includes test) |
| Hours                                                           | 3:04                                                            | 11:08                                                           | 17:14                                                           | 1:05                                                            |
| Cardinality of recurrent class                                  | Cardinality of recurrent class                                  | Cardinality of recurrent class                                  | Cardinality of recurrent class                                  | Cardinality of recurrent class                                  |
| Firm B                                                          | 5650                                                            | 38,202                                                          | 67,258                                                          | 3,553                                                           |
| Firm S                                                          | 5519                                                            | 47,304                                                          | 137,489                                                         | 3,553                                                           |

recurrent class for the finite history /C28 model is only 5% of that for the periodic full revelation model, and apparently this is not a rich enough partition of the state space to provide an adequate approximation. The size of the recurrent class from the finite history s model that does approximate quite well is about 42% of that of the periodic full revelation model. The relative simplicity of the FI model is reflected in its much smaller recurrent class. As a result it computes much quicker than any of the other models.

Finally, to figure out compute times one needs to know how the test statistic behaves as we increase the number of iterations. To ensure that the R 2 statistic was above .99 we needed as much as 200 million iterations, and in all our runs the R 2 flattened out between 250 and 350 million iterations at values that were one to at least four significant digits (see Figure I for an example).

<!-- image -->

## V.E. The Economics of the Alternative Environments

The output of the algorithm includes; strategies, quantities produced and prices by day of the week, realized costs (both operational and maintenance), profits, and consumer welfare. We start with a comparison of the base case AsI model with the base case social planner (the second and third columns in Table V).

Strategies. Panel A of Table V is rather striking. The social planner never shuts down without doing maintenance, and does more maintenance on both its big and its small generators than do the AsI competitors. During the week, when demand is high, the planner operates both its large generators and its small generators at almost full capacity. On average it typically does maintenance on one, and sometimes on both, large generators on Sunday, ensuring that those generators are at a low-cost state when going into the work week. The planner typically does maintenance on one small generator on Saturday and another on

TABLE V

QUANTITIES AND COSTS

|                                            | Base case   | Base case   | Base case   | Excess capacity   | Excess capacity   |
|--------------------------------------------|-------------|-------------|-------------|-------------------|-------------------|
|                                            | Planner     | AsI         | FI          | AsI               | FI                |
| Panel A: Strategies                        |             |             |             |                   |                   |
| Firm B: Shutdown and maintenance           |             |             |             |                   |                   |
| Shutdown percentage                        | 14.52       | 19.96       | 12.31       | 41.97             | 43.75             |
| Maintenance percentage                     | 14.52       | 10.10       | 10.90       | 6.47              | 6.25              |
| Firm S: Shutdwon and maintenance           |             |             |             |                   |                   |
| Shutdown Percentage                        | 16.85       | 21.48       | 20.74       | 53.1              | 56.4              |
| Maintenance Percentge                      | 16.85       | 9.83        | 9.91        | 5.22              | 4.84              |
| Firm B: Operating generators by day of the |             | week        |             |                   |                   |
| Saturday                                   | 1.41        | 1.08        | 1.72        | 1.03              | 1.0               |
| Sunday                                     | 0.88        | 1.21        | 1.65        | 1.03              | 1.0               |
| Weekday ave.                               | 1.93        | 1.78        | 1.78        | 1.03              | 1.0               |
| Firm S: Operating generators by day of the |             | week        |             |                   |                   |
| Saturday                                   | 1.55        | 1.56        | 2.03        | 1.21              | 0.48              |
| Sunday                                     | 1.89        | 1.75        | 1.86        | 1.20              | 0.44              |
| Weekday ave.                               | 2.80        | 2.64        | 2.55        | 1.25              | 1.44              |
| Panel B: Costs                             |             |             |             |                   |                   |
| Maintenance cost B ( /C2 10 /C0 3 )        | 29          | 20.2        | 21.95       | 12.9              | 12.5              |
| Maintenance cost S ( /C2 10 /C0 3 )        | 20.2        | 11.8        | 11.9        | 6.3               | 5.8               |
| Production cost B ( /C2 10 /C0 3 )         | 211.1       | 235.1       | 240.4       | 48.3              | 48.4              |
| Production cost S ( /C2 10 /C0 3 )         | 174.8       | 228.1       | 215.9       | 13.6              | 11.8              |
| Total cost/quantity                        | 0.389       | 0.452       | 0.444       | 0.290             | 0.282             |
| Panel C: Quantities and prices             |             |             |             |                   |                   |
| Average quantity wkend                     | 93.5        | 92.0        | 98.6        | 33.6              | 33.1              |
| Average price wkend                        | 303         | 325         | 260         | 168               | 175.6             |
| Average quantity wkday                     | 185.7       | 181.8       | 181.2       | 42.50             | 42.43             |
| Average price wkday                        | 374         | 401         | 411         | 177               | 177               |

Sunday, and if it requires more maintenance of small generators than that it will maintain two small generators on Saturday.

The AsI equilibrium generates about 30% more shutdown of large generators and 25% more shutdown of small generators than the social planner, but actually does about 30% less maintenance on both types of generators than does the social planner. That is, about half the time generators are not operating in the

AsI equilibrium they are shut down without doing maintenance. The shutdown decision results in higher prices for the firm's operating generator(s). The number of generators operated on weekends by the AsI equilibrium is about the same as the social planner operates, so the AsI equilibrium is operating fewer generators than the social planner on the high-demand weekdays (though it still operates more generators on weekdays than on weekends).

The social planner does more maintenance than the AsI equilibrium generates, and almost all its maintenace is done during the low-demand weekends. This enables the planner to operate more generators on the high-demand weekdays, pushing down price on those days and adding to consumer surplus. The social planner internalizes this increase in consumer surplus, while the firms operating in the AsI equilibrium would not.

Costs. The fact that the social planner does more maintenance, and that the planner can optimize maintenance jointly over the large and small generators, results in much lower production costs for the planner than is generated by the AsI equilibrium. Indeed, the planner has lower total (maintenance plus production) costs per unit quantity. This is despite the fact that our model has increasing costs, and the social planner produces more quantity (particularly on the high demand weekdays).

Prices and Quantities. Recall that we model an electricity market with relatively inelastic demand. So the fact that the planner produces about 2% more output than the AsI equilibrium on weekdays causes the planners' prices to be about 10% lower on those days. This implies that the AsI equilibrium produces a larger difference in prices between weekdays and weekends than does the social planner. However even the social planner's weekday prices are 20% higher than weekend prices; that is, prices ''spike'' on high-demand days. Apparently we need to change the institutional setting to get the price discrepancy between weekdays and weekends to under 20%.

Consumer Surplus and Profits. It is not a surprise that the planner generates higher total surplus (see Table VI) than does the AsI equilibrium, but it is somewhat surprising that the planner also generates more profits. This is largely because the planner does more maintenace than either of the duopolies, and this reduces total costs.

TABLE VI

## CONSUMER SURPLUS AND PROFITS

|                                           | Base Case   | Base Case   | Base Case   | Excess Capacity   | Excess Capacity   |
|-------------------------------------------|-------------|-------------|-------------|-------------------|-------------------|
|                                           | Planner     | AsI         | FI          | AsI               | FI                |
| Cons. surplus ( /C2 10 /C0 3 ) 58,000+    | 662         | 581.5       | 595         | 1,316             | 1,311             |
| Total profits (Firm B+S) ( /C2 10 /C0 3 ) | 716.2       | 707.3       | 706.7       | 58.1              | 61.9              |
| Firm B profits ( /C2 10 /C0 3 )           | 385.3       | 384.5       | 388.1       | 53.2              | 54.5              |
| Firm S profits ( /C2 10 /C0 3 )           | 331.0       | 322.9       | 318.8       | 4.9               | 7.4               |
| Total surplus ( /C2 10 /C0 3 ) 590,000+   | 378.1       | 288.9       | 301.4       | 1,374             | 1,373             |

AsI versus FI Equilibria: Base Case and ''Excess'' Capacity. The comparison of the AsI equilibrium to the FI equilibrium strategies depends on the extent of generating capacity relative to demand. In the base case the FI equlibrium generates less shutdown and more maintenance than does the AsI, but when there is more capacity relative to demand the AsI equilibrium does less shutdown and more maintenance.

The differences are most noticeable in the comparative behavior of the firms during weekends. In the base case the AsI equilibrium generates noticeably less operation of both large and small generators during the weekend than does the FI equilibrium. The weekend shutdowns in the AsI equilibrium enables the firms to signal that their generators will be bid in on the weekdays to follow, and in the base case weekday prices are more than 20% higher than weekend prices. There are no signaling incentives in the FI equilibrium and in that equilibrium more output is produced on weekends. When we increase capacity relative to demand the difference between weekday and weekend prices drops dramatically (to 5.4% in the AsI and 1% in the FI equilibrium) and now both firms operate more on weekends in the AsI equilibrium than in the FI equilibrium (only slightly more for Firm B, but noticeably more for Firm S).

The second noticeable change when we add capacity relative to demand is that the average cost (maintenance plus production cost divided by quantity) is quite a bit lower when there is relatively more capacity. In both capacity environments the average cost in the AsI equilibrium is similar to that in the FI equilibrium, but average costs falls by over 30% when the ratio of capacity to demand increases. Of course firms would have to

weigh any reduction in average costs against the cost of installing the capacity before engaging in capacity-expanding investments, and we cannot compute the private value of capacity expansion without a more complicated dynamic model than the one used here.

Moving to Table VI, we see that the differences in consumer and total surplus between the AsI and the FI equilibrium are always small and differ in sign in the two environments. The major difference between the environment with more capacity relative to demand is that with a higher capacity to demand ratio we see a large increase in consumer surplus and a large (but smaller in absolute value) decrease in producer surplus. As a result total surplus is noticeably larger when the ratio of capacity to demand is higher. This is largely a consequence of prices falling when the ratio of capacity to demand increases, particularly weekday prices. Indeed, it seems that one way to decrease the weekend/weekday price differential is to increase the ratio of capacity to demand.

## VI. Concluding Remark

We have presented a simple framework for analyzing finite state dynamic games with asymmetric information. It consists of a set of equilbrium conditions which, at least in principle, are empirically testable, and an algorithm capable of computing policies that satisfy those conditions for a given set of primitives. Its advantages are twofold. First by chosing alternative information structures we can approximate behavior by agents in complex institutional settings without requiring those agents to have unrealistically excessive information retention and computational abilities. Second the algorithm we use for analyzing the equilibria is relatively efficient in that it does not require storage and updating of posterior distributions, explicit integration over possible future states to determine continuation values, or storage and updating of information at all possible points in the state space. The hope is that this will enable us to approximate behavior and analyze outcomes in markets which have been difficult to deal with to date. This includes markets with dynamic consumers as well as dynamic producers, and markets where accounting for persistent sources of asymmetric information is crucial to the analysis of ourcomes.

## Appendix I: Claims for Periodic Revelation

CLAIM 1 (Periodic revelation). If for any initial st 2 R there is a T /C3 &lt; 1 and a random /C28 (whose distribution may depend on st ) which is less than or equal to T /C3 with probability 1, such that all payoff relevant random variables are revealed at t þ /C28 , then if we construct an equilibrium to a game whose strategies are restricted to not depend on information revealed more than /C28 periods prior to t , it is an equilibrium to a game in which strategies are unrestricted functions of the entire history of the game. Moreover there will be optimal strategies for this game which, with probability 1, only take distinct values on a finite state space, so # jRj is finite. #

Sketch of Proof . Let hi , t denote the entire history of variables observed by agent i by time t , and Ji , t denote that history truncated at the last point in time when all information was revealed. Let ð W /C3 ð/C1j Ji Þ , m /C3 ð Ji Þ , p e ð/C1j Ji ÞÞ be EBE (or restricted EBE) valuations, strategies, and resulting probability distributions when agents condition both their play and their evaluations on Ji (so they satisfy C 1, C 2, C 3 of Section II). Fix Ji ¼ Ji , t . What we must show is that

<!-- formula-not-decoded -->

satisfy C 1, C 2, C 3 if the agents' condition their expectations on hi , t .

For this it suffices that if the * strategies are played then for every possible ð J 0 i , J /C0 i Þ ,

<!-- formula-not-decoded -->

If this is the case, strategies which satisfy the optimality conditions with respect to f W /C3 ð/C1j Ji , t Þg will satisfy the the optimality comditions with respect to f W ð/C1j hi , t Þg , where it is understood that the latter equal the expected discounted value of net cash flows conditional on all history.

We prove the second equality by induction (the proof of the first is similar and simpler). For the intial condition of the inductive argument use the period in which all information is revealed. Then p e ð J /C0 i j Ji Þ puts probability 1 at J /C0 i ¼ J /C0 i , t as does Pr ð J /C0 i j hi Þ . For the inductive step, assume Pr ð J /C0 i , t 0 j hi , t 0 Þ ¼ p e ð J /C0 i j Ji , t 0 Þ . What we must show is that if

agents use the * policies then the distribution of J /C0 i , t 0 þ 1 conditional on hi , t 0 þ 1 depends only on Ji , t 0 þ 1.

Let a bar over a set of variables indicate its complement in [ iJi , t for any t , and

<!-- formula-not-decoded -->

so that /C22 i is the new private, and /C15 is the new public, information in Ji , t 0 þ 1. We assume that

<!-- formula-not-decoded -->

so that the distribution of the new private and public information depend only on agents' policies and the information in [ iJi , t . The fact that (A1) allows the distribution of /C15 to depend on policies generates the possiblity of sending signals or revealing information on events that have occured since all information was revealed. What (A1) rules out is models where the intepretation of those signals depends on information that occurred prior to the period when all states were revealed.

Since for any events ð A , B , C Þ , Pr ð A j B , C Þ ¼ Pr ð A , B j C Þ Pr ð B = C Þ

<!-- formula-not-decoded -->

From (A1) and the * policies, the numerator in this expression can be rewritten as

<!-- formula-not-decoded -->

and from the hypothesis of the inductive argument Pr ð J /C0 i , t 0 j hi , t 0 Þ ¼ p e ð J /C0 i , t 0 j Ji , t 0 Þ . A similar calculation for the denominator concludes the proof. #

## CLAIM 2. There exists a restricted EBE if there is periodic revelation of information. #

Sketch of a Proof . In our existence proof we consider only games for which there is a periodic revelation of all

private information. That is, games in which every /C28 periods all private information is revealed. Note that at that period Ji , t ¼ /C24 t and it contains only the payoff relevant variables ! for all the firms. Our proof will follow Maskin and Tirole (2001) that showed the existence of Markov perfect equilibrium (MPE) for complete information dynamic game with finite action space with the proper adjustments. Their proof established first that there is an MPE in the finite period game and then used the fact that the game is continuous at infinity to establish the existence of MPE in the infinite horizon game (see Fudenberg and Levine 1983).

To prove existence we must allow for behavioral strategies. In our main setup we consider only pure strategies because we believe that this is the right framework for empirical analysis, but to guarantee the existence of restricted EBE we extend our framework and allow for mixing as well. Having behavioral strategies will not change our setting and our equilibrium conditions much. Each firm will still have the evaluation W ð m j Ji Þ according to which it determines its optimal actions and our consistency requirement would be the same. We will just allow the firm to mix but we do not need to have a valuation for the mixed strategy, only for the actions m themselves. Note also that whenever the firm mixes between different m 's these m 's should have the same valuations. 27

Our concept of restricted EBE is relevant for an infinite game. We define a cycle as a game that starts at a particular initial conditions ! 0, which is the list of all payoff relevant random variables in that period, and is being played for /C28 periods. A one-cycle game would be an infinite game that starts with the initial conditions ! 0 being played for one cycle and then starts all over with the same initial condition ! 0 and continue to repeat itself. We construct a T -cycle game in the same manner. A game that starts at a particular ! 0 being played for T cycles (or for T /C28 periods) and then starts over with the same initial conditions ! 0 and continues to repeat itself in this manner. Our approach would be to establish a restricted EBE for this T -cycle game and then let T !1 and show that the limit would be a restricted EBE equilibrium for our dynamic game.

27. Clearly whenever the firm mixes, all the m 's in the support of such a mix are chosen with positive probability in the equilibrium play and all resultant outcomes are in the recurrent class.

Consider now a one-cycle game that starts at a particular ! 0. This is a finite game and it has a perfect Bayesian equilibrium (see Kreps and Wilson 1982). That is, for this game we can define (possibly mixed) strategies m /C3 ð Ji , t , ! 0 Þ and beliefs B /C3 ð z /C0 i , t j Ji , t , ! 0 Þ which are probability distributions over z /C0 i , t (the types of the other players at period t , t ¼ 1 , . . . , /C28 ) that specifies the beliefs of player i regarding the types of other players. 28 The conditions that are satisfied are that m /C3 ð Ji , t , ! 0 Þ is optimal given B /C3 ð z /C0 i , t j Ji , t , ! 0 Þ and that the beliefs are consistent with strategies whenever possible. We can now use this PBE to construct a restricted EBE for our one-cycle game. The strategies would be the same strategies as in PBE and the construction of W ð m j Ji , t , ! 0 Þ is straightforward given the equilibrium strategies. Such a constrcution can be done for every ! 0. In a similar way we can construct a restricted EBE for the T -cycle game, that is, constructing the values W ð m j Ji , t , ! 0 , T Þ and the strategies m ð Ji , t , ! 0 , T Þ where t 2 f 1 , . . . , T /C28 g (as afterward the game will replicate itself). Claim 1 implies that we need only ensure that Ji , t includes the values of the payoff relevant random variables at the beginning of the cycle, and all observable variables since that time. Both W ð m j Ji , t , ! 0 , T Þ and m ð Ji , t , ! 0 , T Þ depend on T which defines the number of cycles we are having before restarting the game.

The last stage is to establish existence of restricted EBE for the infinite horizon game. Let's look at the values and the strategies only of the first cycle (the first /C28 periods). Let's define by m 1 ð Ji , t , ! 0 , T Þ as the restricted EBE strategies of the first cycle (the first /C28 periods) when the game is a T -cycle game and when the starting point is ! 0. There is an equilibrium in which these strategies are identical for every first cycle in the T -cycle game. We now let T !1 and examine the strategies m 1 ð Ji , t , ! 0 , T Þ . We can construct a converging sequence (subsequence if needed) such that m 1 ð Ji , t , ! 0 , T Þ ! m 1 ð Ji , t , ! 0 Þ . We can construct such a converging sequence for every possible ! 0 and define the strategies m 1 ð Ji , t Þ for the infinite game. We now claim that m 1 ð Ji , t Þ together with the valuation W ð m j Ji , t Þ that it generates constitute a restricted EBE.

To do that, we follow similar arguments as in Fudenberg and Levine (1983). First note that that for every T 0 there is a

28. Notethat ! 0 is formally part of Ji , t butwewriteitseparatelyheretoindicate the starting point of any cycle.

sufficiently large T 00 such that if we look at a T -cycle game where T &gt; T 00 then m ð Ji , t , ! 0 , T Þ would be sufficiently close to m 1 ð Ji , t , ! 0 Þ for t &lt; T 0 /C28 . That is, the strategies of the first T 0 cycles converge to m 1 ð Ji , t , ! 0 Þ as T !1 (note that we can view each of the first T 0 cycles as the first cycle). Assume now that m 1 ð Ji , t Þ (together with W ð m j Ji , t Þ ) is not a restricted EBE for the infinite game. Then there is a player i and an information set J 0 i , t such that there is a strategy m 0 such that m 0 /C31 i m 1 ð J 0 i , t Þ , that is, W ð m 0 j J 0 i , t Þ &gt; W ð m 1 ð J 0 i , t Þj J 0 i , t Þ (this is with a slight abuse of notation such that W ð m 0 j Ji , t Þ would be the expected evaluation when player i plays the [possibly mixed] strategy m 0 ). We now claim that if W ð m 0 j J 0 i , t Þ &gt; W ð m 1 ð J 0 i , t Þj J 0 i , t Þ then there is a T and restricted EBE for the T -cycle game that starts at ! 0 such that at the first cycle we have W ð m 0 j J 0 i , t , ! 0 , T Þ &gt; W ð m ð J 0 i , t , ! 0 , T Þj J 0 i , t , ! , T Þ which contradicts the fact that m ð J 0 i , t , ! 0 , T Þ is the equilibrium play for the T -cycle game. This inequality exists because W ð m 0 j J 0 i , t , ! 0 , T Þ (respectively W ð m ð J 0 i , t , ! 0 , T Þj J 0 i , t , ! , T Þ ) can be as close as we wish to W ð m 0 j J 0 i , t Þ (respectively W ð m 1 ð J 0 i , t Þj J 0 i , t Þ ) when we let T be sufficiently large (and using the continuity at infinity).

## Appendix II: Algorithmic Details

We begin with a brief review of the properties of the algorithm, and then move to some notes on how one might usefully amend the algorithm to be more effecient when different primitives are appropriate.

The advantages of using a stochastic algorithm to compute equilibria in full information games relative to standard iterative technqiues like those used in Pakes and McGuire (1994) were explored by Pakes and McGuire (2001). These advanatages are even larger in asymmetric information games that use the EBE conditions. This because those conditions do not require us to form beliefs about players' types, and the stochastic algorithm neither computes posterior beliefs nor tests for their consistency with the actual distribution of types.

Pakes and McGuire (2001) noted that, at least formally, their stochastic algorithm does away with all aspects of the curse of dimensionality but the one in computing their test statistic. Accordingly as they increased the dimension of the state space

in their examples the computation of the test statistic quickly became the dominant computational burden. We circumvent this problem by substituting simulation for explicit integration in the construction of the test statistic, thereby eliminating the curse of dimensionality entirely.

However as is typical in algorithms designed to compute equilibria for (nonzero sum) dynamic games, there is no guarantee that our algorithm will converge to equilibrium values and policies; that is, all we can do is test whether the algorithm outputs equilibrium values, we can not guarantee convergence to an equilibrium a priori. Moreover there may be more than one equilibria which is consistent with a given set of primitives. There are, however, both choices in implementation and amendments to the algorithm that will influence which equilibrium is computed.

0

One choice is that for the initial evaluations, that is, our W . High initial values are likely to encourage experimentation and lead to an equilbrium in which players have explored many alternatives. An alternative way of ensuring experimentation is to amend the algorithm as follows. Instead of having agents chose the ''greedy'' policy at each iteration, that is, the policy that maximizes W k , use choice procedure that has an exogenous probability of chosing each possible action at each early iteration, but let that probability go to 0 for all but the greedy policy as the number of iterations grows large. Though both these procedures will insure experimentation, they will also tend to result in longer computational times.

As noted in a particular applied context one may be more interested in directing the algorithm to compute an equilibrium which is consistent with observed data, say, by introducing a penalty function that penalizes deviations from the exogenous information available, than in computing an equilibria that ensures experimentation. Relatedly note that our estimates of the ~ W are sample averages and will be more accurate at a given location the more times we visit that location. If one is particularly interested in policies and values at a given point, for example, at a point that is consistent with the current data on a given industry, one can increase the accuracy of the relevant estimates by restarting the algorithm repeatedly from that point.

Both the structure of memory provided and the test given in the text are always available, but that memory structure need not be computationally efficient, and the test need not be the most

powerful test. A brief discussion of alternative memory structures and testing procedures follows.

Alternative Memory Structures. It is useful to work with the distribution of the increment in ! between two periods, that is, defining /C17 t þ 1 /C17 ! t þ 1 /C0 ! t , we work with

<!-- formula-not-decoded -->

where P /C17 is derived from the family of distributions in equation (2).

We begin with the case where m is observed by the agent's competitors. Then we could hold in memory either estimates of W ð m j Ji Þ or estimates of W ð /C17 , m j Ji Þ . If the latter we would choose m at iteration k to maximize P /C17 W k ð /C17 , m j Ji Þ p ð /C17 j m , m k /C0 1 /C0 i , ! Þ . The trade-off here is clear. By holding estimates of W ð /C17 , m Þ instead of estimates of W ( m ) in memory, we increase both memory requirements and the number of summations we need to do at each iteration. However, we are likely to decrease the number of iterations needed until convergence, as explicit use of the primitive p ð /C17 j/C1Þ allows us to integrate out the variance induced by /C17 conditional ð m , Ji Þ rather than relying on averaging the simulation draws to do so. The W ð /C17 , m j Ji Þ memory structure is particulary easy to use when the probability of /C17 conditional on mi is independent of m /C0 i (i.e., in capital accumulation games), and we used it in our electric utility example.

When m is unobservable there is an even simpler memory structure that can be used in capital accumulation games. We can then hold in memory estimates of W ð /C17 j Ji Þ and chose m at iteration k to maximize P /C17 W k ð /C17 j Ji Þ p ð /C17 j m , ! Þ (we cannot do this when m is observable because then m is a signal and will have an effect on next period's state that is independent of /C17 ). Then the memory requirements may be larger when we hold estimates of W ð m j Ji Þ in memory relative to holding estimates of W ð /C17 j Ji Þ , and will be if the cardinality of the choice set (of M ) is greater than the cardinality of the the support of the family P /C17 . Notice that the model that holds estimates of W ð /C17 j Ji Þ in memory is a natural way of dealing with continuous controls (continuous m ) whose values are unobserved by competitors, and that we may well have some controls observed and some unobserved, in which case hybrids of the structures introduced above would be possible. As for computational burden, the model that holds estimates of W ð /C17 j Ji Þ in memory has the advantage that it explicitly integrates out over

the uncertainty in /C17 and hence should require fewer iterations until convergence.

Alternative Testing Procedures. Several aspects of the test provided in the text can be varied. First the test provided in the text ensures that the ~ W outputted by the algorithm is consistent with the distribution of current profits and the discounted evaluations of the next period's state. We could have considered a test based on the distribution of discounted profits over /C28 periods and the discounted evaluation of states reached in the /C28 th period. We chose /C28 ¼ 1 because it generates the stochastic analogue of the test traditionally used in iterative procedures to determine whether we have converged to a fixed point. It may well be that a different /C28 provides a more discerning test, and with our testing algorithm it is not computational burdensome to increase /C28 .

Second we used an informal stopping rule, stopping the algorithm when the norm of the bias in the estimates of f W ð/C1Þg was sufficiently small. Instead, we could have used a formal statistical test of the null hypothesis that there was no bias (i.e., test the null H 0 : T ¼ 0). Notice that if we did proceed in this way we could, by increasing the number of simulation draws, increase the power of any given alternative to one. This suggests that we would want to formalize the trade-off between size, power, and the number of simulation draws, and explicitly incorporate allowance for imprecision in the computer's calculations. These are tasks we leave to future research.

## TelAviv University and CEPR Harvard University and NBER

## References

- Athey, Susan, and Kyle Bagwell, ''Collusion with Persistent Cost Shocks,''
- Econometrica , 76, no. 3 (2008), 493-540.
- Bajari, Patrick, Lanier Benkard, and Jonathan Levin, ''Estimating Dynamic
- Models of Imperfect Competition,'' Econometrica , 75 (2007), 1331-1370.
- Benkard, Lanier, ''A Dynamic Analysis of the Market for Wide-Bodied Commercial Aircraft,'' Review of Economic Studies , 71, no. 3 (2004), 581-611.
- Beresteanu, Arie, and Paul Ellickson, ''The Dynamics of Retail Oligopoly,'' (Mimeo: Department of Economics, Duke University, 2006).
- Besanko, David, Ulrich Doraszelski, Yaroslav Kryukov, and Mark Satterthwaite, ''Learning by Doing, Organizational Forgetting and Industry Dynamics,'' Econometrica , 78, no. 2 (2010), 453-508.
- Borenstein, Severin, James B. Bushnell, and Frank A. Wolak, ''Measuring Market Inefficiencies in California's Restructured Wholesale Electricity Market,'' American Economic Review , 92 (2002), 1376-1405.

- Cole, Harold L., and Narayana Kocherlakota, ''Dynamic Games with Hidden Actions and Hidden States,'' Journal of Economic Theory , 98 (2001), 114-126.
- Dekel, Eddie, Drew Fudenberg, and David K. Levine, ''Learning to Play Bayesian Games,'' Games and Economic Behavior , 46 (2004), 282-303.
- Doraszelski, Ulrich, and Sarit Markovich, ''Advertising Dynamics and Competetive Advantage,'' RAND Journal of Economics , 38, no. 3 (2007), 1-36.
- Doraszelki, Ulrich, and Ariel Pakes ''A Framework for Applied Dynamic Analysis in IO,'' In The Handbook of Industrial Organization , ed. Armstrong, M., and Porter, R. (Amsterdam: North Holland, 2007), 1889-1966.
- Doraszelki, Ulrich, and Mark Satterthwaite, ''Computable Markov-Perfect Industry Dynamics,'' RAND Journal of Economics , 41 (2010), 215-243.
- Ericson, Richard, and Ariel Pakes, ''Markov-Perfect Industry Dynamics: A Framework for Empirical Work,'' Review of Economic Studies , 62 (1995), 53-82.
- Freedman, David, Markov Chains (New York: Springer Verlag, 1983).
- Fudenberg, Drew, and David K. Levine, ''Subgame Perfect Equilibrium of Finite and Infinite Horizon Games,'' Journal of Economic Theory , 31 (1983), 227-256.
- ---, ''Self Confirming Equilibrium,'' Econometrica , 61, no. 3 (1993), 523-545.
- Goettler, Ronald L., Christine A. Parlour, and Uday Rajan, ''Equilibrium in a Dynamic Limit Order Market,'' Journal of Finance , 60, no. 5 (2005), 2149-2192.
- ---, ''Informed Traders and Limit Order Markets,'' (Mimeo: The of Chicago, the Graduate School of Business, 2008).
- Gowrisankaran, Gautam, and Robert J. Town, ''Dynamic Equilibrium in the Hospital Industry,'' Journal of Economics and Management Strategy , 6, no. 1 (1997), 45-74.
- Harvey, Scott M., William W. Hogan, and Todd Schatzki, ''A Hazard Rate Analysis of Mirantı ´s Generating Plant Outgaes in California,'' (Mimeo: Kennedy School of Government, 2004).
- Kreps, David M., and Robert Wilson, ''Sequential Equilibria,'' Econometrica , 50, no. 4 (1982), 863-894.
- Lee, R., and Ariel Pakes, ''Multiple Equilibria and Selection by Learning in an Applied Setting,'' Economics Letters , 104 (2009), 13-16.
- Maskin, Eric, and Jean Tirole, ''A Theory of Dynamic Oligopoly, I: Overview and Quantity Competition with Large Fixed Costs,'' Econometrica , 56 (1988), 549-570.
- ---, ''Markov Perfect Equilibrium: Observable Actions,'' Journal of Economic Theory , 100 (2001), 191-219.
- Pakes, Ariel, and Paul McGuire, ''Computing Markov Perfect Nash Equilibrium: Numerical Implications of a Dynamic Differentiated Product Model,'' RAND Journal of Economics , 25, no. 4 (1994), 555-589.
- ---, ''Stochastic Algorithms, Symmetric Markov Perfect Equilibrium, and the 'Curse' of Dimensionality,'' Econometrica , 69 (2001), 1261-1281.
- Pakes, Ariel, Michael Ostrovsky, and Steve Berry, ''Simple Estimators for the Parameters of Discrete Dynamic Games with Entry Exit Examples,'' RAND Journal of Economics , 38, no. 2 (2007), 373-399.
- Patrick, Robert H., and Frank A. Wolak, ''The Impact of Market Rules and Market Structure on the Price Determination Process in the England and Wales Electricity Market,'' POWER working paper PWP-047, University of California Energy Institute, 1997.
- Robbins, Herbert, and Sutton Monroe, ''A Stochastic Approximation Technique,'' Annals of Mathematics and Statistics , 22 (1951), 400-407.