## A Computational Framework for Analyzing Dynamic Procurement Auctions: The Market Impact of Information Sharing ∗

John Asker, Chaim Fershtman, Jihye Jeon, and Ariel Pakes (UCLA, Tel-Aviv, Boston, and Harvard Unversities)

June 20, 2018.

## Abstract

This paper develops a computational framework to analyze dynamic auctions. We modify the Experience Based Equilibrium concept to account for dynamic auctions and add, and then operationalize a boundary consistent condition which mitigates the extent of multiple equilibria that can arise in Experience Based Equilibria. Our example shows that allowing for the dynamics implicit in many auction environments is important in that it enables the emergence of equilibrium states that can only be reached when firms are responding to dynamic incentives. It also shows that the impact of information sharing can depend crucially on the extent of dynamics and suggests that information sharing, even of strategically important data, need not be anti-competitive.

Keywords: Experience Based Equilibria, Dynamic Procurement Auctions, Information Sharing.

∗ JEL Codes: D43, K21, L41, C63, C73. We would like to thank numerous seminar audiences for their comments and questions. We are particularly grateful to Gautam Gowrisankaran and Mark Satterthwaite for extensive comments. El Hadi Caoui provided excellent research assistance. Financial Assistance from the US-Israel Binational Science Foundation is greatly appreciated. Contact details: John Asker: johnasker@econ.ucla.edu; Chaim Fershtman: fersht@post.tau.ac.il; Jihye Jeon: jjeon@stern.nyu.edu; and Ariel Pakes: apakes@fas.harvard.edu. The usual caveat applies.

## 1 Introduction

This paper develops a computational framework to analyze dynamic auctions and then applies it to illustrate the possible implications of different rules for information exchange in that setting. A realistic framework for analyzing dynamic auctions requires the framework to allow for serially correlated asymmetric information. The literature on the numerical analysis of dynamic games with serially correlated asymmetric information was considered by Fershtman and Pakes (2012), and we provide the modification required to use it to analyze dynamic auctions. More fundamentally we extend their notion of restricted experience based equilibrium by adding a consistency requirement on the boundary of the recurrent class of states and show how to compute and test for boundary consistent equilibria.

Dynamic auctions are sequential auctions in which the state of the bidders, and therefore their evaluation of the good that is auctioned, change endogenously depending on the history of auction. The value of winning an auction to produce aircraft or ships depends on the backlog or the order book of the firm, and the value of winning a highway repair project or a timber auction depends on whether the inputs currently under the control of the firm are already fully committed for the following period. The fact that the auction is dynamic implies a rich set of strategic incentives. For example, a firm may choose to allow a competitors' state to transition to a point where that competitor becomes a less aggressive market participant in order to win a subsequent auction at a lower bid.

As an illustration we examine how the extent of information sharing impacts competition in a dynamic sequence of procurement auctions. Our goal is to shed light on the extent to which dynamic considerations can color the way antitrust regulators, procurement agencies, and other policy agencies approach the regulation of information sharing. The specific model we investigate is loosely based on the description of timber auctions in Baldwin, Marshall and Richard (1997), although, to keep the model simple, many departures are made from the precise institutional features described therein. Having this specific empirical example in mind eases much of the exposition. In each period, two firms can bid for the right to harvest a lot of timber in a first price sealed bid auction. Each firm has a stock of timber that it already has the right to harvest (its inventory). This stock is private information, and its evolution is the source of dynamics. To compete in the auction, firms must pay a participation fee and simultaneously submit a bid. A firm may also choose to not participate. The winner of the auction, if any, receives the right to harvest the lot, and discovers how much harvestable material it contains. Harvest then occurs, which depletes the stock of timber each firm has. 1

In our benchmark model, once every T periods, there is a full revelation of the state variable. That is, during this revelation period each firm observes the stock of unharvested timber of its competitor. Information sharing is modeled as shrinking the

1 The closest model to ours is that estimated in the innovative contribution of Jofre-Bonet and Pesendorfer (2003). This framework is further extended in Groeger (2014), Saini (2013), Balat (2015) and Jeziorski and Krasnokutskaya (2016). Jofre-Bonet and Pesendorfer's model, and those that follow, has private information that is conditionally independent across states. That is, conditional on (observed) state variables, knowing the private information of a rival last period provides no information as to the private information of the rival this period. This is not the case in our model. In particular, this means that the competitors' prior period bid is a signal on its current state, and that information sharing has persistent value across periods.

time interval between full revelation periods. We also investigate a model in which firms decide whether to share information. Voluntary information sharing involves firms making a choice every T periods as to whether to reveal every period for the next T periods. For voluntary information sharing to occur over the next T periods, all firms must want to share information. Finally we compare the results from these models to those we obtain from a model with myopic firms.

The numerical analysis of this game illustrates how information sharing can, through increases in the precision of the firm's beliefs about its competitors' states, affect bidding behavior at a given state. This, in turn, shapes the desirability, and therefore the likelihood, of being in different states. An important point to bear in mind is that, conditional on the information that they have, firms compete unilaterally (there is no cartel like behavior). However more information induces firms to act so as to spend more time in states where competition is less intense; which are states where both firms have larger stocks of timber. That is an increase in information increases the intensity of bidding and decreases profits in just about all states, but because of the incentives to move to states where bidding intensity is lower, increasing information decreases the average bid and increases average profits. Since increasing information induces firms to spend more time in states with higher inventory it also increases total sales from the auctioned timber.

Interestingly, although more information increases the value of firms, in our voluntary information exchange game firms have difficulty committing to exchange information and most often choose not to share. Finally when we compare to a situation where firms do not care about the future (have a discount factor of zero), the extent of information sharing has negligible effects.

This paper is organized as follows. Subsection 1.1, which follows, discusses the related literature. In subsection 1.2 we provide a brief review of the role of information sharing in antitrust policy. Section 2 describes our baseline model, and then the information sharing and the voluntary information sharing variants of the model. Then, in section 3, computational details are described. A reader not concerned with computational details can skip this section and proceed directly to section 4. Section 4 discusses the numerical analysis, focusing on the competitive impact of information sharing. Section 5 concludes.

## 1.1 Related Literature

Our paper is closely related to the literature on the numerical analysis of dynamic oligopolistic games that uses the Ericson and Pakes framework (1995; for a survey of this literature see Doraszelski and Pakes, 2007). Recent applications of this methodology to questions related to antitrust policy include Besanko, Doraszelski and Kryukov (2014), on predatory pricing, and Mermelstein, Nocke, Satterthwaite and Whinston (2014), on mergers. Within this literature, the closest papers to ours are Saini (2013) and Jeziorski and Krasnokutskaya (2016). Both these papers apply the Markov Perfect equilibrium concept to auction settings, exploring the optimal procurement policy given capacity constrained suppliers and subcontracting, respectively. 2

As noted, our paper differs from this literature in that our focus is on information asymmetry, as in Fershtman and Pakes (2012). While that paper focuses on capital

2 Both these papers build on Jofre-Bonet and Pesendorfer (2003).

accumulation games, we consider a more complex structure where, since we are modeling an auction, the evolution of a firm's state depends not only on its own action (its bid), but also on the bids of its competitors. We also introduce and operationalize a boundary consistency condition that rules out equilibria and can be rationalized either by prior information or experimentation.

Within the auction literature Maskin and Riley (2000) consider asymmetric auctions and show that sealed bidding tends to favor weaker bidders while in open auction the bidder with the highest value win. Athey, Levin, and Seira (2011) extend the framework to a repeated auction. They consider a theoretical model of a repeated auction and then use data on timber auctions to conduct an empirical analysis of the effect of the type of auction (open or sealed bid) on the firms participation and bidding.

Our paper also relates to the empirical literature on bidding collusion. There are several approaches in the literature for examining whether an auction is competetive or collusive. See, for example, Porter and Zona (1993, 1999), Baldwin Marshall and Richard (1997), Pesendorfer (2000), Bajari and Ye (2003), and Asker (2010). Aoyagi (2003) considers collusion in a repeated auction when bidders are allowed to communicate with each other before each auction. In another paper Athey and Bagwell (2008) consider collusion between competitors in a repeated homogenous-good-bertrand market, in which costs (types) are private information and evolve over time according to an exogenous markov process. In contrast to the environment considered here, the evolution of costs (types) in that model is unaffected by the actions of any player. We do not have specific collusion in our setup but we do examine information exchange regarding the firms' inventories on the firms' participation and bidding behavior. The policy implications of our paper relate also to the extensive literature on information sharing in oligopoly see Clarke (1983), Gal-Or (1985, 1986), Shapiro (1986), and Kirby (1988). For a survey of this literature see Kuhn and Vives (1995). More recent empirical work includes Doyle and Snyder (1999) and Luco (2017).

## 1.2 Background: information exchange and policy

Though explicit agreements to fix prices are per se violations of the antitrust laws, the legal treatment of information sharing among competitors is less clear. 3 The legality of an exchange of price information is determined in part by the extent to which the audience is restricted. Clearly, a merchant who posts prices in a public display is communicating price information to competitors but is not in violation of statutes. More problematic is the communication of price information between competitors in a way that consumers do not have access to. 4 U.S. courts apply a rule of reason test to decide whether the exchange of price information constitutes an unreasonable restraint

3 The canonical statement of the per se nature of price fixing under section 1 of the U.S. Sherman Act is United States v. Socony-Vacuum Oil 310 U.S. 150 (1940). Information sharing also tends to fall within the scope of section 1 of the Sherman Act. See the majority decision in United States v. Container Corp. 393 U.S. 333 (1969).

4 In Container Corp the U.S. Supreme Court held that, despite any agreement on pricing, the exchange of information about specific prices offered to specific customers was a violation of the antitrust laws. This case created confusion as to whether per se treatment applied to information sharing. This was clarified in United States v. Citizens &amp; Southern National Bank 422 U.S. 86., which explicitly adopted a rule of reason approach. In doing so the court appealed to the idea that price exchange facilitated price stabilization (a form of price fixing).

of trade. 5 Factors that are taken into account include the level of market concentration, the fungibility of the products, the nature of the information exchanged, its timeliness and specificity, and whether the information is made publicly available. 6

U.S. courts take a sympathetic view of the sharing of non-price information recognizing that efficiencies are more likely from the sharing of information regarding production processes and costs. For instance, the Supreme court in the 1925 Maple Flooring Manufacturers decision, held that:

'... corporations which openly and fairly gather and disseminate information as to the cost of their product, the volume of production, ..., stocks of merchandise on hand, ... without however reaching or attempting to reach any agreement or any concerted action with respect to prices or production or restraining competition do not thereby engage in unlawful restraint of commerce...' 7

Contemporary guidance from the FTC and DoJ states that 'The sharing of information relating to price, cost, output, customers, or strategic planning is more likely to be of competitive concern than the sharing of less competitively sensitive information.' 8 This suggests a somewhat more nuanced view in modern times. The E.U., by contrast, has tended to take a harsher view of both price, and non-price, information sharing agreements. The exchange of information relating to future prices is considered a restriction of competition by object (equivalent to a per se offense in the U.S.). 9 This may include non-price strategic information. Our example illustrates that a harsh approach to the sharing of information can be misguided.

## 2 A Model of a Dynamic Auction

We consider a model in which there are n firms in the market and no entry into and exit from the industry. Each of the firms can harvest and sell a portion of their stock of lumber each year at a fixed price. The actual quantity that can be sold in each period depends on a firm specific random outcome of a harvesting process from a stock of timber that has not yet been harvested, and is private information. The stock will be increased if the firm wins a procurement auction which occurs every period. The procurement auction is a simple first price sealed bid auction. Participation in the procurement auction is costly, and participation decisions are public information observed by all firms. However, the amount of lumber per lot won in the auction is random and observed only by the winning firm.

There are two types of periods. Periods with full information exchange and periods without information sharing. In our baseline model full information exchange occurs

5 In this context, an unreasonable restraint would be one that synthesizes or facilitates a cartel-like pricing structure. Information exchange may also constitute a facilitating practice in inferring the existence of an explicit price fixing conspiracy.

6 A modern discussion of the judicial approach taken can be seen in the decision of Justice Satomayor, while sitting as a judge on the second circuit court of appeal, in Todd v. Exxon Corp 275 F.3d 191 (2001).

7 see Maple Flooring Manufacturers' Assn. v. United States 268 U.S. 563 (1925)

8 See FTC/DoJ's April 2000 Antitrust Guidelines for Collaborations Among Competitors at page 15.

9 See the E.U. 2011 Guidelines on the applicability of Article 101 of the Treaty on the Functioning of the European Union to horizontal co-operation agreements and Dole Food Company et al. v. Commission .

every T periods. There are a number of possible rationals for this and it keeps the information set finite. 10

We begin with the timing of the events that occur within a period. Then, we describe the overall structure of the game. Following that, we define the equilibrium conditions, explain our computational procedure, and then provide and compare results from models with different amounts of information sharing.

## Timing

1. Each firm brings into the period a stock of timber that can be harvested ( ω i,t ).
2. Every period begins with the announcement a first price sealed bid auction.
3. Firms observe the realization of their stochastic participation fee. We assume that F it ∼ U [ F l , F h ]. The realization is not observed by rival firms.
4. Each firm decides whether to participate in the auction. All the firms that decide to participate submit their bids simultaneously. At the time of bidding, participation decisions of rival firms are not observable.
5. The rules of the auction define an increment b . Bids must be multiples of this increment. Hence bids must be elements of the set { b, 2 b, 3 b, ..., b } .
6. The highest bid wins. If high bids are tied, then the winner is decided randomly, with each tied bid having an equal chance of winning. We denote the probability of winning by firm i by p w ( b i , b -i ). The winning bid, the identity of the winner, and the participants in the auction become public information.
7. If there is information exchange it occurs at this point. If it is a period of information exchange (which occurs every T periods), then ω i,t of all the firms is revealed. Otherwise the new public information revealed in the period is; who participated in the auction, denoted as p t , who won the auction at period t , denoted by i w t , and the winning bid b ∗ t . We denote the new public information as ξ n t ≡ [ i w t , b ∗ t , p t ]. In a period of information exchange the new public information is [ i w t , ω t ], the identity of the firm that won the auction and the observed state ω t ≡ { ω i,t } . 11
8. The winner discovers the amount of timber on the plot it won. This is given by θ + η t where θ is the average amount and η t is an i.i.d (across time) discrete random variable. η t is not observed by competing (losing) firms. The timber in stock ( ω i,t ) is updated accordingly. There is a random realization of the ability to extract, e + glyph[epsilon1] i,t where glyph[epsilon1] i,t is a discrete random variable with probabilities p ( glyph[epsilon1] i,t ). The draws on glyph[epsilon1] i,t are independent over agents and not observed by competitors.
9. Harvest is made and each firm sells all its harvested timber at a unit price of $1. Thus a firm's per period revenue is given by min { ω i + I { i = win } ( θ + η ) , e + glyph[epsilon1] i } ,

10 See Fershtman and Pakes (2012) for a list of ways to keep the information set finite. Information revelation every T periods is convenient for us as it allows us to directly compare equilibria based on a sequence of larger information sets. We can justify our structure by; assuming that a regulator imposes mandatory periodic information revelation, or assuming the existence of a trade group that facilitates the sharing of information every T periods.

11 Note that at a period of information revelation the winning bid and the participation decision of that period do not enter the public information because they are payoff and informationally irrelevant. They do not provide any additional signal on the ω of the firms, as these ω 's are revealed at that periods.

where I { i = win } is an indicator function which takes the value of one if i wins the auction and zero elsewhere. 12 The quantity harvested by firm i is not observable by other firms. 13 Note that if b i = ∅ signifies no participation, a firm's expected profit, given ( b i , b -i , F i , ω i ), are glyph[negationslash]

<!-- formula-not-decoded -->

10. All the firms updates their private ω i .

## Agents' Strategy Spaces.

In general, the strategy space could include everything observed from the history of the game. Most of the early applied literature focused on equilibria with strategies that depend only on variables which are either 'payoff' or 'informationally' relevant. The payoff relevant variables are defined, as in Ericson and Pakes (1995) or Maskin and Tirole (2001), to be those variables that are not current controls and affect the current profits of at least one of the firms. In a game with asymmetric information observable variables that are not payoff relevant will affect behavior if they are informationally relevant. A variable is informationally relevant if and only if even if no agents' strategy depended upon the variable some player can improve its expected discounted value of net cash flows by conditioning on it; for more details see Fershtman and Pakes (2012). That paper also shows that in models with periodic revelation of information there exists an equilibrium which only conditions on the revealed information and the information that has become available since the revelation. We focus on this equilibrium in the remainder of the paper. 14

The information set of firm i consists of public and private information. The public information at the beginning of period t , denoted by ξ t consists of; τ t ∈ [1 , . . . , T ], the time since last information exchange, ω t -τ t , the last revealed ω vectors, and the τ t -period history of winning bids, winner identities and participant identities. Formally ξ t = { τ t , ω t -τ t , ξ n t -1 , ..., ξ n t -τ t } . 15 Information revelation occurs when τ t = T (which is period τ t = 0 for the next cycle). The private information at the point in time decisions are made includes ω i,t and F i,t . However since F i,t is i.i.d., it enters the value function linearly, and does not have an independent effect on future values whereas the other state variables do. As a result it will be useful to have notation for J i,t = ( ω i,t , ξ t ) separately from I i,t = ( J i,t , F i,t ).

Strategies. There are two elements of a firm's strategy; the participation strategy and bidding strategy. We denote firm i strategy as b ( J i , F i ) →{B∪ ∅ } where b = ∅ signifies no participation.

12 Here, and in what follows, we drop time subscripts, except where they add clarity.

13 Otherwise the observable harvested quantity may serve as a signal regarding ω i .

14 Equilibrium is defined formally in section 2.2.

15 Note that for a period with information revelation the public information includes only the identity of the winner in the auction and not the winning bid or the participants identity as these variables are not informationally relevant.

## 2.1 The Dynamic System

We let V ( I i ) be the value of the game for a player i given his information set I i . We have

<!-- formula-not-decoded -->

where (i) W ( ∅ | J i ) is the value of the game if the firm decides not to participate in the auction in that period, and (ii) W ( b | J i ) is the value when the firm participates and bids b ∈ B .

Now consider the value of the game when firm i participates in the auction and bids b ∈ B . For every possible J i we define p w ( b | J i ) to be the player's perception about the probability of winning the auction when it bids b and we let i w be the winning firm. Letting β be the discount factor, the firm's expectation of current period revenue (which excludes F i ) is

<!-- formula-not-decoded -->

It follows that, for b ∈ B ,

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

where ω ′ ( ω, η, glyph[epsilon1] i ) is the updated ω i when the firm does win the auction and is a function of the random outcomes of the size of the lot won ( η ) and the harvesting decision ( glyph[epsilon1] i ); i.e. ω ′ ( ω, η, glyph[epsilon1] i ) = max { 0 , ω i -( e + glyph[epsilon1] i ) + θ + η } . When the firm does not win the auction its updated ω is a function of the initial ω and the random outcome of the harvesting process, glyph[epsilon1] i , i.e. ω ′ ( ω i , glyph[epsilon1] i ) = max { 0 , ω i -( e + glyph[epsilon1] i ) } . p ( ξ ′ | ξ, ω i , b, i = i w ) is the probability distribution of future public information given the current public information ξ , the firm's private information ω i and the identity of the firm winning the auction with bid b . Similarly, p ( ξ ′ | ξ, ω i , b, i = i w ) is the probability distribution of future public information given that the firm loses the auction.

Lastly the continuation value when the firm chooses not to participate in the auction, our W ( ∅ | J i ), is

<!-- formula-not-decoded -->

where p ( ξ ′ | ξ, ω i , b = ∅ ) is the probability distribution of future public information given the current public information, the firm's private information, and the choice of not participating in the auction.

<!-- formula-not-decoded -->

## 2.2 The Restricted Experience Based Equilibrium

We now derive the conditions of a restricted experience based equilibrium for this game (see Fershtman and Pakes 2012). We let s be the set consisting of the payoff and informationally relevant states of all the firms, that is s = ( J 1 , ..., J n ) when all the J i have the same public component ξ . We will say that J i = ( ω i , ξ ) is a component of s if it contains the information set of one of the firms whose information is combined in s . Note that we can also write s = ( ω 1 , ..., ω n , ξ ) and define the set of possible states S = { s : ( ω 1 , . . . , ω n ) ∈ Ω n ( ω ) , ξ ∈ Ω( ξ ) } .

Definition of a REBE: A restricted experience based equilibria consists of the following three objects.

1. A set R that is a subset of the state space (i.e. R⊂S ).
2. Bidding and participation strategies, b ∗ ( J i , F i ) for each firm and for every J i which is a component of any s ∈ S and F i ∈ [ F l , F h ].
3. A set of numbers W ≡ { W ∗ ( b | J i ) b ∈B∪ ∅ } that, for every J i that is a component of any s ∈ S , have an interpretation as the firm's perceptions of the expected discounted values of current and future cash flows conditional on its information set should it bid b or not participate in the auction (i.e. where b = ∅ ).

For these objects to define a REBE they must satisfy the following three conditions.

C1: R is a recurrent class. The Markov process generated by any initial condition s 0 ∈ R , and the transition kernel generated by { b ∗ ( J i , F i ) } i =1 ,...,n J i ∈ s ∈S ,F i ∈ [ F l ,F h ] has R as a recurrent class; that is, with probability one, any subgame starting from an s 0 ∈ R will generate sample paths that are within R forever.

C2: Optimality of strategies. Conditional on W ≡ { W ∗ ( b | J i ) b ∈B∪ ∅ } J i ∈ s,s ∈S , the strategies are optimal. That is glyph[negationslash]

<!-- formula-not-decoded -->

C3: Consistency of values on R . Consistency requires that the perception of discounted values, generated by every possible choice at every J i that is a component of an s ∈ R equals the expected discounted value of returns generated by that choice from that J i ; where expectations are taken using the empirical distribution of outcomes from that J i (empirical distributions are denoted by a superscript E ). Formally for every b ∈ B ∪ ∅ , W ∗ ( b | J i ) , the equilibrium evaluations satisfy

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

µ E w ( b | J i ) is the empirical probability of winning if the agent bids b at J i or

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

while glyph[negationslash]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As noted in Fershtman and Pakes (2012) any Markov Perfect Bayes equilibrium will satisfy the conditions of a REBE. In fact a REBE admits more equilibria than does Markov Perfect Bayes. To understand the main reason why, it is helpful to distinguish between two types of points in the recurrent class; interior points and boundary points.

At an interior point an agent will stay in the recurrent class with probability one regardless of which of the feasible policies is chosen. At a boundary point the agent will stay in the recurrent class with probability one if the equilibrium policy is chosen. The agent may move outside of R if a feasible but non-equilibrium policy is chosen. In a restricted experienced based equilibrium the perceived discounted value of all feasible policies from an interior point equals the actual expected discounted value that would arise from all agents playing their equilibrium policies. However at boundary points only the perception of returns from the policies that lead to points in R with probability one are required to equal the actual discounted values were all agents to play their optimal strategies. Policies that lead to points outside of the recurrent class are determined solely by perceptions and different perceptions on boundary points can support different equilibria.

There are situations where it might be reasonable to impose restrictions on off the equilibrium path behavior at boundary points. This would restrict the set of equilibria further. We consider one such restriction in the next section. The reader who is not interested in this refinement should be able to go directly to section 2.4.

## 2.3 Strengthening REBE: Boundary Consistency

If agents have prior knowledge or experiment with off the equilibrium path policies at boundary points then we might expect off the equilibrium path behavior at boundary points to satisfy some restrictions. This section provides one such restriction; that the perceived value of off-equilibrium-path play from a boundary point equals the expected discounted value of profits from that point when all agents use their equilibrium policies (note that those policies are defined on all of S ). We call this a boundary consistency condition as it, together with condition C2, ensures that off the equilibrium path play at boundary points would lead to discounted values that are less than those of optimal play. Note that to impose this condition we need only calculate discounted values for

profits along sample paths before they re-enter the recurrent class (if they do re-enter) as we can use C3 above to evaluate the periods thereafter.

To formalize our condition we need to define the set of actions which could be taken from points in the recurrent class that would generate outcomes which are not in the recurrent class. To this end let supp [ p s ′ ( ·| b i , b ∗ -i , s )] be the support of the probability distribution over next period states, generated by actions ( b i , b ∗ -i ) and initial state s = ( J i , J -i ). The boundary set of couples ( b, s ), which we denote by B , are the set of action-state combinations such that if s = ( J i , J -i ) ∈ R , action b is taken by i and equilibrium actions are taken by the other agents, then a probability distribution for s ′ is generated which has a point in its support which is not in the recurrent class, or

<!-- formula-not-decoded -->

glyph[negationslash]

<!-- formula-not-decoded -->

The additional condition that needs to be satisfied for the one-period deviation to actually yield an outcome which is less than the value of optimal play is C4 below. In this condition we use γ to index periods since the off-equilibrium-path policy is played. Let F = ( F i , F -i ). The probability distribution p ( s γ | b, s, { F τ } γ τ =1 ) is derived recursively, with p ( s 1 | b, b ∗ , s ) = ∑ F -i p ( s 1 | b i = b, b -i = b ∗ ( J -i , F -i ) , s ) p ( F -i ), and for γ &gt; 1 , p ( s γ | b, s γ -1 ) = ∑ F p ( s γ | s γ -1 , b ∗ , F ) p ( F ) .

C4:Boundary Consistency. Let π i ( b ∗ , s, F ) ≡ π ( b ∗ i ( J i , F i ) , b ∗ -i ( F -i , J -i ) , F i , J i ) and π i ( b, b ∗ -i , s, F ) ≡ π ( b, b ∗ -i ( F -i , J -i ) , F i , J i ). Then our condition is ∀ ( b, J i ) component of ( b, s ) ∈ B and for every F i , glyph[negationslash]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where p ( s γ | s γ -1 , b ∗ , F γ ) is the probability of reaching state s γ at time γ given that at time γ -1 the state is s γ -1 , participation fees are F γ and the players play the equilibrium strategies b ∗ .

Definition. We call an equilibrium which satisfies C 1 to C 4 a 'Boundary Consistent' REBE .

Notice that if for any sample path (i.e. any { s γ } ∞ γ =1 ), we define γ R = min γ { γ : ( s γ ) ∈ R} , we can replace

<!-- formula-not-decoded -->

in C4 with β γ R ∑ F i V ( s γ R , F i ) p ( F i ) . We provide a formal test for the existence of boundary consistent policies below. The fact that we can replace the infinite sum in C4 with β γ R ∑ F i V ( s γ R , F i ) p ( F i ) eases the burden of computing of the test.

## 2.4 Information sharing

We study the role of information sharing between firms participating in a sequence of procurement auctions. In our benchmark case information is shared every T periods. Between these periods firms do not observe the evolution of their competitors' states; however they do observe the public information which may help in predicting their competitors' behavior. We then compare our baseline model to two models that allow for information exchange at more frequent intervals. The only difference among them is the extent of information sharing as we do not allow for any additional mechanism which facilitates coordination among firms. We also assume that when information is exchanged firms reveal their true state. 16

## 2.4.1 Information Exchange (IE)

In the first information sharing model there is mandatory information exchange every period. 17 We denote this model as IE .

## 2.4.2 Voluntary Information Exchange (VIE)

In the second information sharing model, we adjust the baseline model such that in the period in which there is a forced information exchange firms also make a decision on whether to share information in every period for the next T -1 periods hence. If one of the firms does not wish to share information, there is no voluntary information sharing over the next T -1 periods, and in the T th period firms' chose whether they wish to share information in the subsequent T periods. We call this model the VIE model and describe it in more detail now.

We have a period index τ = 0 , 1 , . . . T, which designates the time from the period of mandatory information exchange. At τ = 0 each firm also needs to decide if it wishes to be part of an information exchange scheme. The decision of whether to share information, ˜ R i ∈ { 0 , 1 } , is made simultaneously with the participation and bidding decision. ˜ R i = 1 denotes that firm i wishes to share information. Information is actually exchanged, denoted by R = 1, only when ˜ R i = ˜ R -i = 1.

The timing of the game is adjusted so that the sequence described in section 2 changes as follows. If τ = 0, step (4) is replaced with

- 'Each firm decides whether to participate in the auction. Participation is costly, it requires an expenditure of F i,t (a draw from the uniform distribution). If they decide to participate they simultaneously submit their bids and decide whether to reveal information over the next T periods. If both firms agree to reveal information, there is information exchange over the next T periods and the voluntary information exchange state R is set to 1. R is 0 otherwise. At the time of bidding, participation decisions of rival firms are not observable.'

For τ &gt; 0 we replace step (5) with

16 Truthful revelation may require careful design of the incentives surrounding the agreement. For an exploration of this in the context of explicit cartels in auction markets see (for example) Graham and Marshall (1987), McAfee and McMillian (1992) and Mailath and Zemsky (1991).

17 Formally we compute the model already described with the constraint that T = 1.

- 'Information exchange occurs at this point. If R = 1, ω i,t of all the firms is revealed. This is in addition to the new public information (i.e. who won the auction). If R = 0, the new public information revealed in the period is the same as in the baseline model that is ξ n t = [ i w t , b ∗ t , p t ] .'

In the V IE game the agents' information set is different than in the B game in that the public information also includes the most recent information sharing indicator, or R ∈ { 0 , 1 } .

The information exchange decision: At periods when τ = 0 firms need to decide if they wish to exchange information in the next T periods. In those periods we let ˜ R ∈ [0 , 1] indicate the decision over whether to exchange information ( ˜ R = 1) or not ( ˜ R = 0) and define

<!-- formula-not-decoded -->

where W ( b, ˜ R = 1 | J i ) ( W ( b, ˜ R = 0 | J i )) is the firm's perceptions of the expected discounted value of current and future cash flows, given the choice of bid and the choice to reveal information in the next T periods, conditional on its information set.

The actual exchange state, our R , has R = 1 if and only if ˜ R i = ˜ R -i = 1. When τ = 0, W ( b, ˜ R = 0 , J i ) is analogous to W ( b, J i ) in equation (3). When τ = 0 and ˜ R = 1 there is a probability of moving into different R states that depends on the perceptions of whether the competitor will chose to reveal. We let p ( R = 1 | J i , ˜ R = 1) be the firm's perception of that probability given ˜ R i = 1 and J i . We use this perception combined with equation (3) to form W ( b, ˜ R i = 1 | J i ). For τ &gt; 0 the dynamics are similar to the B case when R = 0, and are similar to the dynamics of the IE case when R = 1.

Definition of a REBE for the VIE case: The definition of a REBE for the VIE case is analogous to that for the Baseline and IE cases but with the differences we now consider. In the VIE in periods with τ &gt; 0 the public information ξ includes the outcome of the last voluntary information exchange, i.e. R ∈ { 0 , 1 } . At τ = 0 the optimal policies are given by glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

Finally since the agent needs a perception of the probability that R = 1 when he evaluates the returns from ˜ R = 1, there is a an additional consistency requirement that the perception of this probability is, in equilibrium, equal to its empirical probability, or

<!-- formula-not-decoded -->

Before going to our results we explain the computational algorithm we used to obtain them. A reader who is not interested in the computational algorithm should be able to go straight to section 4.

## 3 Computation, relationship to learning, and testing.

This section provides a reinforcement learning algorithm that computes a REBE for our baseline model. We then provide a test for boundary consistency of a computed REBE.

The algorithm models players as having perceptions on the value that is likely to result from the different actions available to them at each state. The players choose the actions that is optimal given those perceptions and the realized participation fees. The realizations of random variables whose distributions are determined by the chosen actions and the current state lead to a current profit and a new state. Players use this profit, together with their perceptions of the continuation values they assign to the new state, to update their perceptions of the values of the starting state. They then proceed to choose an optimal policy for the new state which maximizes the perception of the value from that state. This process continues iteratively.

As is explained in Fershtman and Pakes (2012) the reinforcement learning algorithm described above is an algorithm that agents could actually use to learn the values associated with various actions. If the game is a capital accumulation game, i.e. a game where the transition probabilities for an agent's state depend only on the given agent's policies, then the agent would learn the distribution of future states conditional on all of its possible action. This is not necessarily the case when the game is not a capital accumulation game, such as the sequence of procurement auctions we consider here. The reason is that in a general game an agent might never know what the evolution of its state would have been if it played an action off the equilibrium path even if that action, had it been played, would keep the agent in the recurrent class with probability one. For example in the auction game we consider here, an agent that wins the auction at an optimal bid, will not learn from repeated equilibrium play what would have happened if it bid a lower value (since in this auction game agents do not observe non-winning bids of competitors).

We could perturb the algorithm to maintain the analogy with learning by forcing agents to experiment with different policies at each state (as in Fudenberg and Levine (1998)). This would, however, increase the complexity of the algorithm. A less computationally burdensome way of proceeding to compute a REBE is to use knowledge that the computer has in its memory but the agent does not have to update the values associated with all policies (even those the agent does not take). Indeed from a computational point of view the fact that we can compute an equilibrium for a non-capital accumulation game without explicitly calculating the impact of one firm's policies on the evolution of its competitors' states is an advantage of our algorithm relative to algorithms which require explicit computation of all continuation values (see, for e.g., Besanko et al. (2014)).

We begin this section by outlining the computational algorithm for an arbitrary set of initial conditions and providing a test of whether the output of the algorithm constitutes a REBE. We then discuss how one can test whether the output of the algorithm is consistent with the stronger notion of equilibrium that ensures that feasible, though non-optimal, actions at the boundary points are indeed non-optimal.

## 3.1 The Algorithm

The algorithm consists of an iterative procedure and subroutines for calculating initial values and profits. We begin with the iterative procedure. Each iteration, indexed by k , starts with a location that is a state of the game (the information sets of the players) L k = [ J k 1 , ..., J k n ], and has objects in memory, M k = { M k ( J ) : J ∈ s ∈ S } . Each iteration updates both the location and the memory. The rule for when to stop the iterations consists of a test of whether the equilibrium conditions defined in the last section are satisfied. We begin with the basic algorithm and then move on to testing. A more detailed discussion of increasing the efficiency of the algorithm is provided in the results section.

Memory: The elements of M k ( J ) specify the objects in memory at iteration k for information set J , and hence the memory requirements of the algorithm. Often there will be more than one way to structure the memory with different ways having different advantages. Here we focus on a simple structure that will always be available (though not necessarily always efficient; see Fershtman and Pakes, 2012).

M k ( J ) contains a counter, h k ( J ), which keeps track of the number of times we have visited J prior to iteration k . If h k ( J ) &gt; 0 it also contains

<!-- formula-not-decoded -->

If h k ( J ) = 0 there is nothing in memory at location J . When we need to evaluate policies at a J at which h k ( J ) = 0 we have an initiation procedure which sets

<!-- formula-not-decoded -->

The choice of initial values will be discussed below.

Updating L k : We find the values in memory associated with different b for each agent at location L k (or use the initiation procedure if needed), take a random draw on F i , and determine the optimal bid as glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

These bids determine which, if any, player wins the auction. Let b k ≡ Max i { b ∗ ( J k i , F i ) } be the highest bid at iteration k . If b k = ∅ there is an auction. We assume that if there is an auction and more than one firm bids b k there is a lottery that determines the winning bid.

The b k , the identity of the winner ( i k w ), and the participation decisions of all agents (the vector p k ) enable us to update the public information sets as glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

where ξ k ( τ k + 1) is notation for ξ k with τ changed from τ k to τ k + 1. That is if we are in a full information exchange period (if τ k = 0) we reveal all information about ω , delete the variables in ξ k (as the revelation of ω makes them irrelevant), and add the identity of the winning bidder. If τ k = 0 we simply add the newly generated information ( p k , i k w , b k w ) to the old information set and increase its τ by one.

After bids are submitted and information is revealed but before the next auction occurs, the firm that wins the auction gathers its new timber and all agents sell what they can sell to the market. The random draws from the harvest ( η ) and from the market sale ( glyph[epsilon1] i for each i ) are realized and each agent's stock of timber is augmented as

<!-- formula-not-decoded -->

Thus the information prior to the next auction is given by

<!-- formula-not-decoded -->

where it is understood that ω k i is omitted from firm's J k +1 i .

Updating The Values in Memory: The algorithm uses the information generated by the random draws that lead to the new location to update agents' perceptions of the values associated with the different policies. We only update objects in memory associated with the location L k , but we update each component of { W k ( b | J k i ) } b ∈B∪ ∅ for all i . That is we update the continuation values for the policies not taken as well as for those taken. The update for each W k ( b | J k i ) assumes that the profits and the continuation state that would have accrued to the agent had it chosen that b are those that would have been generated by the competitor's chosen policy, the current state, and random draws from the primitive processes.

The update of the expected value from pursuing strategy b at state J k i , i.e. W k ( b | J k i ), is obtained by assuming that the 'realized' value that would have been obtained from playing that b was one draw from the expected value of choosing strategy b at J k i . The 'realized' value is evaluated as the profits it would have earned had it played ' b ' plus its current perception of the discounted continuation value from the state that it would have moved to. More formally let J k +1 i ( b, b k -i , · ) be the updated information set were we to follow the updating procedure defined above after substituting b for b k i in those formula. This generates ξ k +1 ( b, b k -i , · ) and ω k +1 i ( b, b k -i , · ). Then the perceptions of the value for taking action b at state J k i are updated as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This updating procedure sets the current perception of the value of taking action b at state J i k equal to a simple average of what the perception of taking action b would have been had the agent taken that action every time in the past that it had reached J i k . Though this averaging procedure does satisfy the Robbins and Monroe (1951) criteria for convergence of a stochastic integral, it is unlikely to be efficient. This because the earlier values are associated with less precise evaluations. We come back to discussing ways of increasing computational efficiency in the results section, and now turn to the testing procedure.

## 3.2 Testing Procedures.

Appendix A provide a detailed explanation of how to test whether the output of the algorithm satisfies the conditions of a REBE. It is analogous to the test described in Fershtman and Pakes (2012), so in the text we suffice with a brief overview of how to construct the test statistic. We then consider testing for boundary consistency. This concept is new to this paper, and the test has elements which differ from the test used for REBE as it requires testing for the validity of moment inequalities. Accordingly we go over the test for boundary consistency in more detail.

## 3.2.1 Testing for a REBE.

We stop the algorithm at a particular iteration and fix the values for that iteration.

<!-- formula-not-decoded -->

The test is designed to check whether these values, together with the policies and the recurrent class that they generate, satisfy conditions C1 to C3 above.

The test is based on simulating a sample path with the optimal policies generated by ( { W ∗ ( b | J i ) } b ∈ B , W ∗ ( ∅ | J i ) } ) . Since the state space is finite, the simulated path will wander into a recurrent class after a finite number of iterations, and stay within that class thereafter. Every point within that class will be visited repeatedly. We keep a separate memory for each point visited in the test's simulation run.

The first time a particular point is visited we record the simulated continuation value resulting from taking every possible action at that point. I.e. the profits plus the discounted continuation value (evaluated by { W ∗ ( ·|· ) } ) generated by; their action, the policy chosen by their competitors and simulated random draws on the primitives. 18 We also record the square of this continuation value and initiate a counter for the amount of times this point was visited in the simulation run. Recall that we visit each point in the recurrent class repeatedly. At each subsequent time a given point is visited we again calculate a simulated continuation value for each possible policy and then form an average of the simulated continuation values from each time the point was visited for all policies at the point. A similar averaging is used for the continuation value squared. When the simulation run is stopped the memory for each point visited consists of the average of past simulated continuation values from that point, the average of the continuation values squared, and the number of times the point has been visited in the test run.

The squared difference between W ∗ ( b | J i ) and the estimated continuation value for playing policy b at J i is the mean square error of our estimate of W ∗ ( b | J i ). It can be additively decomposed in the standard way into the bias squared of our estimate and the variance of our estimate. The variance is unbiasedly estimated by the average of the squared value minus the estimate squared. So by differencing the mean square error from the estimate of the variance we are able to get an unbiased estimate of the bias in our estimator for W ∗ ( b | J i ). Our test statistic is a weighted average of the percentage bias (squared) in our estimates of W ∗ ( b | J i ). We weight the different b at a

18 Since the stage game is simultaneous move, we can evaluate a counterfactual choice of a given agent's policy by substituting it, and the optimal policies of competitors, into this calculation.

given J i equally, and the sum over b at different J i by the number of times that J i was visited in the simulation run.

More formally the test is an L 2 ( P R ( ns ) ) norm of the bias in the sum of simulated continuation values as estimates for W ∗ , where P R ( ns ) refers to the simulated estimate of the recurrent class generated by W ∗ . We accept the test when the test statistic is less than .001; heuristically when our R 2 is above .999. For more details see the Appendix.

## 3.2.2 Testing for Boundary Consistency or for C4.

We begin with a verbal explanation of the test for a given { W ( b | J i ) } b,J i . Initially we run a five million iteration simulation run from the last point visited in the algorithm. We call the points visited during that run as the points in the recurrent class, and tabulate the fraction of times each of those points was visited during this simulation run, say { h ( J i ) } J i .

We then start new simulation runs from every point visited in this simulation run for every possible policy from that point. This is analogous to the simulation procedure used in the test for a REBE, except that in the boundary consistency test we have to do it for every possible policy. We continue each of the simulation runs for every ( b, J i ) until the run enters a point in our estimate of the recurrent class. We keep track of the discounted profits that the firm earns from the simulation run until the simulation enters the recurrent class and this is added to the discounted proposed equilibrium continuation value from the entry point to the recurrent class. Under the null of a boundary consistent REBE, the result is an unbiased estimate of the expected discounted value from taking the policy b at J i . This is tabulated and averaged with the other simulated discounted values obtained from the given ( b, J i ). We then determine which of the ( b, J i ) are boundary couples by looking to see if any of the simulated runs starting at J i with policy b had a simulation run which did not enter the recurrent class immediately. Finally we introduce a test of C4 and apply it to the boundary couples.

We now provide a more formal description of the testing procedure we run after determining our estimate of the recurrent class. At each point, say J i , chose every b ∈ B ∪ ∅ and, using the policies generated by { W ( ·|· ) } , start R simulation runs. Index the runs from each ( J i , b ) couple by r and let the sequence of states visited during the r th simulation run be { J i,γ r } γ ∗ r γ r =1 , where γ ∗ r is the period in the simulation run where the simulation enters the recurrent class (or some sufficiently large number, which we take as 100).

Our estimate of the discounted value of net cash flows from run r for the couple ( b, J i ) is glyph[negationslash]

<!-- formula-not-decoded -->

where it is understood that b ( J i, 1 , F i, 1 ) = b or the policy we are evaluating. We keep in memory the average of the ˆ W r ( b | J i ), the average of ˆ W r ( b | J i ) 2 and the maximum of γ ∗ r from the R simulation runs from each ( b, J i ).

Our test statistic is glyph[negationslash]

Let χ ( b, J i ) = 1 whenever max r γ ∗ r ( b, J i ) = 1 , where it is understood that γ ∗ r ( b, J i ) is the γ ∗ associated with a particular ( b, J i ). Then

<!-- formula-not-decoded -->

is our estimate of the set of boundary couples. For each of these couples we have a sample mean W R ( b | J i ) which is an unbiased estimate of the population mean from R sample paths (in our case R = 20), and we use the average of the sum of squares of ˆ W r ( b | J i ) and this sample mean to calculate an unbiased estimate of V ar [ W R ( b | J i )].

We now use this information to form a test. Since we are testing inequalities, i.e. that the boundary point policies lead to discounted values of future net cash flows which are less than the optimal policy at the J i associated with the boundary point, we will have to use a test statistic which is not pivotal, i.e. whose distribution does not have a standard form (like the chi-square or normal). We define the statistic below and then explain how we can construct its distribution under the null that our conditions are satisfied. We accept the test if the observed value of the test statistic is less than the 95 th quantile of the distribution we construct.

The observed test statistic for boundary consistency for the points in ˆ B : Let ˆ B ( J i ) = { b : ( b, J i ) ∈ ˆ B } ⊂ B ∪ ∅ and # ˆ B ( J i ) be the number of elements in ˆ B ( J i ). Also let

<!-- formula-not-decoded -->

where [ W R ( b | J i ) -W ( b ∗ | J i )] + = max [ W R ( b | J i ) -W ( b ∗ | J i ) , 0].

Let J ˆ B be the set of J i for which there is an element in ˆ B . Recall that h ( J i ) is the number of visits to the point J i in the initial simulation run and calculate for each J i ∈ J ˆ B

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The simulated distribution of the test statistic under a conservative null: We now simulate the distribution of, T ( ˆ B ), under the null that W ( b | J i ) = W ( b ∗ | J i ) for each ( b, J i ) ∈ B , thereby insuring the size of the test. 19 For each ( b, J i ) ∈ ˆ B take ns independent random draws from a normal with mean zero and variance V ar [ W R ( b | J i )], and call them, z ( b, J i ) 1 , . . . z ( b, J i ) ns (we set ns = 50). For each draw, indexed by r = 1 , . . . , ns calculate

<!-- formula-not-decoded -->

19 The test used here is often referred to as the least favorable test statistic in the econometric literature; see for example Romano, Shaikh and Wolf (2014).

and if and only if

<!-- formula-not-decoded -->

Let ˜ T ( ˆ B ) . 95 ns be the 95 t h percentile of the distribution of ˜ T ( ˆ B ) r . Then we accept the test of

H 0 : Boundary Consistency

<!-- formula-not-decoded -->

## 4 Numerical analysis

A parameterized version of each of the baseline (B), information exchange (IE) and voluntary information exchange (VIE) models is computed, using the computational algorithm described above. The parameterization and the implementation of the algorithm are discussed below, together with a description of the resulting computational burden. An equilibrium is computed in each of the three models. These equilibria are described in section 4.3, together with a discussion of the economic content of these numerical results.

## 4.1 Parameter values

The parameter values that are used in the numerical analysis are given in table 1, below. In each model, there are two firms and four possible bids. This structure is adopted to enable us to compute an assortment of models in reasonable time (computational burdens are discussed in the next subsection). Similarly, the time between forced revelation periods in the baseline model is 4 periods, a choice arrived at through balancing the desire to have meaningful private information evolving over time with the need to keep the state space at a manageable scale. Participation costs are assumed to be uniformly distributed U [0 , 1]. To give some sense of scale, this means that the participation costs are between 0 and 50%, and on average 25%, of the mean revenue generated by a harvested lot of timber 20 .

## 4.2 Computational burden and updating procedure

A REBE is computed using the algorithm provided in section 3.1. Recall that there may be many equilibria that satisfy our equilibrium conditions. The choice of initial conditions for continuation values (our { W 0 ( ·} ) is one determinant of which equilibria the algorithm will compute. If the initial conditions are higher than possible equilibrium values then all policies are likely to be explored, and, as a result, any equilibrium the algorithm converges to is likely to be boundary consistent. The cost of choosing high initial conditions is that they are likely to cause the algorithm to require many iterations before it converges to equilibrium values.

20 Two points on this. First recall that harvesting and production costs are normalized to zero. Second we have also computed a model with participation cost that distribute U [0 , . 5] and the qualitative results are unchanged.

Table 1: Parameter specifications

| Parameters:                                 |                 | IE                 | V IE               |
|---------------------------------------------|-----------------|--------------------|--------------------|
| Periods between ω revelation                | T               | 1                  | { 1,4 }            |
| Common Parameters:                          |                 |                    |                    |
| Distribution of fixed cost of participation | F i             | U[0,1]             | U[0,1]             |
| Discount factor                             | β               | 0.9                | 0.9                |
| Mean timber in a lot                        | θ               | 3.5                | 3.5                |
| Disturbance around θ                        | η               | { -0.5,0.5 }       | { -0.5,0.5 }       |
| Probability on η realizations               |                 | { 0.5,0.5 }        | { 0.5,0.5 }        |
| Mean harvest capacity                       | e               | 2                  | 2                  |
| Disturbance around e                        | glyph[epsilon1] | { -1,0,1 }         | { -1,0,1 }         |
| Probability on glyph[epsilon1] realizations |                 | { 0.33,0.33,0.33 } | { 0.33,0.33,0.33 } |
| Bidding grid                                |                 | { 0.5,1,1.5,2 }    | { 0.5,1,1.5,2 }    |
| Number of firms/bidders                     |                 | 2                  | 2                  |
| Retail price of a unit of timber            |                 | 1                  | 1                  |

We incurred that cost and used as initial conditions

<!-- formula-not-decoded -->

for all ( b, J i ) ∈ ( B , J ). 21 To see why we chose these initial values, note that e/ (1 -β ) is the discounted value of being able to sell the mean harvest forever and e/ ( θ +1) is smaller than the periodicity that the firm would have to win the auction in order to have the timber needed to sell e units in every period. So ( F + . 5) e/ [( θ +1)(1 -β )] is less than the cost of bidding in enough periods to be able to sell e units in every period if all the auctions that the firm bid on were won and the winning bid was the lowest bid possible. Finally ω ( F + . 5) / ( θ +1) adds back in the cost of the timber the firm has already stored.

Table 2 provides statistics that summarize different aspects of the computational burden we incurred in computing the equilibria. Partly as a result of our choice of initial conditions, the number of states visited (and hence explored) in both the B and the V IE algorithms was large; 7.5 and 7.9 million respectively. Though the recurrent classes were (less than) an order of magnitude smaller than this (less than 330,000), there was a significant computational burden in finding them. Computation of the IE equilibrium was much less difficult; the number of states visited was only 2,724 and the cardinality of the recurrent class was 2089 reflecting the fact that the IE model does not require the continuation values associated with every possible different four period history after the period of revelation.

21 F is the average value of F i and is 0.5 under our parametrization.

To lessen the computational burden for the B and VIE model we used the following simple way of reducing the impact of the bias in the early iterations resulting from the high initial conditions.

1. First the computational algorithm was run for 50 million iterations resetting the counters for the states every 10 , 000 iterations as follows;

<!-- formula-not-decoded -->

2. Then the algorithm is run for 5 million iterations without resetting the counter.
3. Next a run of 5 million iterations is used to form the test for the REBE (recall that the test requires an R 2 statistic to be greater than .999).
4. If the test is passed we stop the algorithm. Otherwise we repeat steps 1 to 3.

Steps 1 through 3 were repeated six times for B before the test was satisfied and eight times for VIE. To obtain our results for the IE model we used a similar procedure but with shorter runs; step one above is run for 10 million iterations and it took only one round of our steps before convergence. The boundary consistency test was run, as described in section 3.2.2, after we accepted the test for the Restricted EBE. 22 All the equilibria we describe here were boundary consistent, though we did find one that was not which we do not report on. A summary of compute times is provided in the bottom half of table 2 and the footnote to the table describes the program and computer used for the runs. 23

To insure that our estimate of the recurrent class was accurate, we extended the last five million run by an additional five million and asked what fraction of the incremental iterations visited points that had already been visited in the initial five million. For the baseline, information exchange, and voluntary information exchange model the fractions were 99.42%, 100%, and 98.9%% respectively 24 .

22 The number of simulation runs used to determine whether a point in the recurrent class was a boundary point was fifty, and the number of repetitions to form the averages used in the test of the boundary points was twenty.

23 The total computation times, including testing, for each of the models, were (in hours): B - 110, IE 4.5, VIE - 185. There are many, likely quite helpful, ways one might improve on this, but optimizing the algorithm is beyond the scope of this paper.

24 Note also that the incremental points in the B and VIE cases are likely to be points that satisfy the boundary consistency conditions.

```
Size of recurrent class: B IE V IE 325,843 2,081 328,692 Number of all states visited during computation: B IE V IE 7,495,307 2,724 7,908,122 Computation times per 5 million iterations (in hours): B IE V IE 1:38 1:06 1:56 Computation times for testing for a REBE (5 million iterations, in hours): B IE V IE 1:43 1:09 2:00 Computation times for testing for boundary consistency (100,000 iterations, in hours): B IE V IE 3:03 0:16 75:41
```

Table 2: Computational details

Notes: Computation was conducted in MATLAB version R2013a using (a Dell Precision T3610 desktop with) a 3.7 GHz Intel Xeon processor and 16GB RAM on Windows 7 Professional. A round of computation includes steps 1 and 2 of the computational procedure given above. It is 55 million iterations for B and V IE and 15 million iterations for IE .

## 4.3 Computational Results

Table 3 shows a summary of average per-period performance metrics for each of the B , IE , and V IE models and for a social planner ( SP ) version of the model. The social planner observes all private information of both firms and maximizes total revenues minus participation fees. 25 Were it not for the existence of a non-zero minimum bid, which distorts participation somewhat, the planner's allocation problem would be equivalent to that of the ideal, perfectly coordinated, cartel; the planner maximizes the discounted value of the sum of future net cash flows.

The average bid for B , IE and V IE , is 1.09, 0.94 and 1.04 respectively. The ordering of bids across models is the same if we look at winning bids, or winning bids conditional on the number of bidders. So if lower prices correspond to weakened competition, the view that information sharing (of strategic data) is akin to collusion has some support, in that both phenomena generate lower bids.

Increased participation is often associated with more competition which should, in turn, lead to higher bids; and there is more participation in the IE than in the B equilibrium. Part of the participation difference might be attributed to the more

25 Specifically, the planner's objective is to maximize revenues minus participation fees. That is, the planner views the bid payment as a transfer between players while participation payments represent real costs to the society. As in the baseline case, each firm draws a stochastic i.i.d. participation cost from F i ∼ U [0 , 1] in each period. After observing the realization of the participation costs, the planner chooses which firm to assign the lot to or chooses not to assign the lot to any firm. In terms of the informational structure, we assume that the planner has access to the F i and ω i realizations of both firms.

Table 3: Summary statistics, in per-period terms, by model

|                                                    |     B |    IE |   V IE | SP    |
|----------------------------------------------------|-------|-------|--------|-------|
| Avg. bid                                           |  1.09 |  0.94 |   1.04 | -     |
| Avg. winning bid (revenue for the auctioneer)      |  1.11 |  0.98 |   1.07 | -     |
| Avg. winning bid with ≥ 1 firm participating       |  1.16 |  0.98 |   1.12 | -     |
| Avg. winning bid with 1 firm participating         |  1.06 |  0.67 |   0.99 | -     |
| Avg. winning bid with 2 firms participating        |  1.23 |  1.16 |   1.2  | -     |
| Avg. # of participants                             |  1.52 |  1.63 |   1.52 | 1     |
| Avg. # of participants with ≥ 1 firm participating |  1.59 |  1.63 |   1.59 | 1     |
| Avg. participation rate                            |  0.76 |  0.81 |   0.76 | 0.50  |
| % of periods with no participation                 |  4.39 |  0.15 |   3.85 | 0.004 |
| Avg. total revenue                                 |  3.35 |  3.49 |   3.37 | 3.50  |
| Avg. profit                                        |  0.81 |  0.87 |   0.84 | -     |
| % of periods; lowest omega wins                    | 66.37 | 60.8  |  65.32 | 85.96 |
| Average total social surplus                       |  2.73 |  2.72 |   2.74 | 3.10  |

glyph[negationslash]

Notes: Here, and in tables 4, 5, 6, and 7, the per-period profit is defined as π ( ω i ) -I { i = win } b i - { b i = ∅ } F i = min { ω i + I { i = win } ( θ + η ) , e + glyph[epsilon1] i } -I { i = win } b i -{ b i = ∅ } F i . Total revenue is defined as ∑ i π ( ω i ) = ∑ i min { ω i + I { i = win } ( θ + η ) , e + glyph[epsilon1] i } . Total social surplus is defined as ∑ i { π ( ω i ) -{ b i = ∅ } F i } . Averages are taken over periods. The statistics are computed based on a 5 million iteration simulation of each model.

glyph[negationslash]

detailed information structure in the IE equilibrium facilitating more coordinated bids, as there are less periods in the IE equilibrium when neither firm bids (.015 vs .04 percent). However, the statement that more information leads to softer competition seems to be clearly at odds with the relationship between bids and participation in the periods with at least one bidder, as even in those periods there is more participation in the IE than the B equilibrium (1.63 vs 1.59).

Of course what might be confusing differences in behavior in a model of a static (or a repeated) game, might not be confusing in the context of a dynamic game. In particular differing dynamic incentives will generate differences in the propensity to hold different stocks of lumber. We expect participation and bidding to differ with differences in those stocks, and the table's comparisons between the IE and B outcomes are comparing different weighted averages of the stock combinations. The probable role of dynamics in explaining differences in the implications of the information environment also comes out clearly when we compare Table 3 to Table 4. Table 3 indicates that more information (the IE equilibrium) generates a higher discounted cash flow and therefore higher average profits, but table 4 makes it clear that once we condition on the stock of timber the B equilibrium generates higher profits almost always. 26

Before leaving table 3 we note that all three models deliver (essentially) the same social surplus (albeit with IE being lowest by 0.01). However the maximal social surplus from the market equilibria, 2.73, is much lower than the social surplus attained by the planner (3.10). The participation numbers indicate why the planner does so much better. The planner only ever lets one firm enter the auction, thus saving on the cost F (the planner also benefits from being able to better coordinate the path of the ω -tuple). In the IE equilibrium the firms generate almost the same revenue (equivalently,

26 The only exception are states which are visited only .15% (1.12%) of the periods in the B (IE) equilibrium.

glyph[negationslash]

output) per period as does the planner, but requires much greater participation to do so, thus generating a lower social surplus. By contrast, firms in B are less effective at revenue generation (their stocks are not always high enough to satisfy the demand that faces them), but generate less wasteful participation.

Table 4: Probability Distribution by ω -tuple for B , IE and SP

|                 | Prob. Dist.   | Prob. Dist.   | (%) Profit   | (%) Profit   | (%) Profit   |
|-----------------|---------------|---------------|--------------|--------------|--------------|
| ( ω i ,ω - i )  | B             | IE            | SP           | B            | IE           |
| ( ≤ 4 , ≤ 4)    | 65.51         | 32.59         | 90.12        | 0.68         | 0.52         |
| ( ≤ 4 , 5 - 7)  | 12.61         | 19.09         | 4.52         | 0.57         | 0.58         |
| ( ≤ 4 , ≥ 8)    | 4.05          | 10.55         | 0.28         | 0.60         | 0.59         |
| (5 - 7 , ≤ 4)   | 12.61         | 19.09         | 4.52         | 1.51         | 1.26         |
| (5 - 7 , 5 - 7) | 0.88          | 5.72          | 0.22         | 1.49         | 1.46         |
| (5 - 7 , ≥ 8)   | 0.14          | 1.12          | 0.02         | 1.49         | 1.13         |
| ( ≥ 8 , ≤ 4)    | 4.05          | 10.55         | 0.28         | 1.62         | 1.58         |
| ( ≥ 8 , 5 - 7)  | 0.14          | 1.12          | 0.02         | 1.66         | 1.87         |
| ( ≥ 8 , ≥ 8)    | 0.01          | 0.17          | 0.00         | 1.72         | 1.56         |

Notes: This table shows the probability of intervals of ω -tuples for B , IE and SP . Here, and in tables 5, 6, and 7,the per-period profit is a probability weighted average, over the states underlying each ω -tuple.

To explain these phenomena we have to consider the relationship between the different information structures and dynamic incentives. We begin with the differences between the IE and B equilibria (the discussion of VIE is delayed until section 4.3.2). Table 4 divides the state space by ω -tuples, and shows the probability distribution over these ω -tuples for each of B and IE as well as the average per-period profits earned by the firms with ω 's in the tuple. The distribution for SP is also provided for comparison.

Both B and IE are dynamic games in which the control that the firm uses to change its stock of timber is its bid. Hence, to understand how differences in information sets shape the different paths taken through the state space, an examination of bidding is required. The salient feature of the data in table 4 that the bids must explain is how the IE information structure generates bids that keep the firms in higher ω tuples. The lower ω -tuples, the tuples in which both firms have ω ≤ 4, are the least profitable tuples in either equilibrium; indeed the maximal profits for a firm with ω ≤ 4 is less than half the minimal profits with ω ≥ 4. What is evident from table 4 is that the additional information available to firms in the IE equilibrium enables them to stay away from states with ω ≤ 4 with greater propensity than the firms in the B equilibrium are able to. The fraction of periods with both firms with ω ≤ 4 is 65.5% in B compared to 32 . 6% is IE, while the fraction of states with at least one firm with ω ≤ 4 is just over 62% for IE compared to just over 82% for B.

In contrast the social planner spends more time in the ( ≤ 4 , ≤ 4)-tuples than either firms in B or IE , thereby generating a smaller cost of holding the timber already procured. So IE firms maintain ω stocks that are greater, and in that sense even less efficient, than in the B equilibrium. Table 4 also reveals that firms in IE spend more time in states that are asymmetric, in the sense of having one firm with a high ω and

one with a low ω .

Table 5: Bids by ω -tuple for B and IE

|                 | Bids   | Bids   | Bids   | Bids   | Bids   | Bids   | Bids   | Bids   | Bids   | Bids   | Profit   | Profit   |
|-----------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|----------|----------|
| ( ω i ,ω - i )  | B      | B      | B      | B      | B      |        |        | IE     |        |        | B        | IE       |
|                 | ∅      | 0.5    | 1      | 1.5    | 2      | ∅      | 0.5    | 1      | 1.5    | 2      |          |          |
| ( ≤ 4 , ≤ 4)    | 0.22   | 0.13   | 0.27   | 0.31   | 0.07   | 0.07   | 0.13   | 0.28   | 0.47   | 0.06   | 0.68     | 0.52     |
| ( ≤ 4 , 5 - 7)  | 0.11   | 0.32   | 0.45   | 0.11   | 0.02   | 0.02   | 0.53   | 0.37   | 0.08   | 0.00   | 0.57     | 0.58     |
| ( ≤ 4 , ≥ 8)    | 0.08   | 0.58   | 0.29   | 0.04   | 0.02   | 0.00   | 0.88   | 0.12   | 0.00   | 0.00   | 0.60     | 0.59     |
| (5 - 7 , ≤ 4)   | 0.43   | 0.18   | 0.34   | 0.04   | 0.01   | 0.33   | 0.10   | 0.52   | 0.05   | 0.00   | 1.51     | 1.26     |
| (5 - 7 , 5 - 7) | 0.37   | 0.50   | 0.09   | 0.02   | 0.01   | 0.40   | 0.59   | 0.01   | 0.00   | 0.00   | 1.49     | 1.46     |
| (5 - 7 , ≥ 8)   | 0.39   | 0.53   | 0.06   | 0.01   | 0.01   | 0.11   | 0.89   | 0.00   | 0.00   | 0.00   | 1.49     | 1.13     |
| ( ≥ 8 , ≤ 4)    | 0.51   | 0.25   | 0.22   | 0.02   | 0.00   | 0.60   | 0.14   | 0.26   | 0.00   | 0.00   | 1.62     | 1.58     |
| ( ≥ 8 , 5 - 7)  | 0.53   | 0.39   | 0.06   | 0.01   | 0.00   | 0.84   | 0.16   | 0.00   | 0.00   | 0.00   | 1.66     | 1.87     |
| ( ≥ 8 , ≥ 8)    | 0.61   | 0.36   | 0.03   | 0.00   | 0.00   | 0.47   | 0.53   | 0.00   | 0.00   | 0.00   | 1.72     | 1.56     |

Notes: This table shows bids by intervals of ω -tuples for B and IE . ∅ indicates non-participation.

Table 5 contains the probability distributions over bids that underlie the distribution over the ω -tuples examined in table 4 together with average profits in those states. Grey shaded cells indicate bids that are more frequent in IE than in B . Notice first that, when both firms' have ω ≤ 4, bidding is more aggressive in the IE than in the B equilibrium; there is both more participation in IE and a higher fraction of bids are higher than the minimal bid in these states. This reinforces the impression that the increased information created when moving from B to IE is not allowing the firms in IE to better coordinate; more information actually intensifies competition when stocks of timber are low. Relative to IE the firms in the B model are less certain about their competitor's states and this softens competition.

The opposite seems to be true when at least one of the firm's has an ω greater than eight, or both firms have an ω between five and seven. In these states participation in IE is sometimes greater than in B but, conditional on bidding, the bids in IE are smaller. The result is that the winning bid in IE is the minimal bid much more frequently. For example, when both firms have an ω between five and seven the IE bidding patterns are consistent with firms participating when their F i draw is sufficiently low, and then bidding the minimal amount. The result is that in virtually every case the winning bid is the minimal bid. This essentially reduces the auction to a lottery. When both firms have an ω between five and seven in the B equilibrium participation is somewhat lower, but conditional on participating about a quarter of the bids are more than the minimal bid. A similar comparison holds when both firms have an ω greater than eight. In the ( ≥ 8 , 5 -7) -tuple and the ( ≥ 8 , ≤ 4) tuple the IE equilibrium has the high ω firm typically sitting out the auction, deferring to the lower ω rival who most often wins with the minimal bid. In contrast when the B equilibrium is at the tuple ( ≥ 8 , 5 -7) the high ω firm bids in 47 % of the time (compared to only 16% of the time in the IE equilibrium,) and 15% of those bids are greater than the minimal bid (compared to 0% for the IE equilibrium).

So when at least one of the firms has an ω greater than eight, or both firms have an

.

ω between five and seven, it seems that more information enables better co-ordination of bids. The one couple of states in table 5 that we have not discussed is when one firm has an ω less than or equal to four and the other has an ω between five and seven. There is a sense in which this couple of states lies 'in-between' the low stock states in which more information intensifies competition and the high stock states in which more information facilitates coordination. In this state the high ω firm participates more in the IE equilibrium (67% vs 57%), and 85% of the time that the high ω firm participates in the IE equilibrium it bids more than the minimum bid (compared to 68% of the time in the B equilibrium). The low ω firm in the ( ≤ 4 , 5 -7) participates more in the IE equilibrium, but bids less aggressively than it does in the B equilibrium. The fact that the high ω firm bids more aggressively in the IE equilibrium but the low ω firm does not, explains part of the difference between the probabilities of different states between the IE and B model provided in table 4, as it underlies the fact that the IE model typically generates disproportionate number of states where at least one firms has a high ω stock.

Tables 6 and 7 examine the differences in bids between the B and IE model in more detail. Table 6 looks at bids in the low ω states and shows the rather dramatic increase in aggressiveness that results from providing firms with the increased information in the IE equilibrium. At state (0 , 0) firms in IE participate 99% of the time (compared to 88% in B) and when they participate 78% of the time they chose the maximal bid (versus 28% in B). The differences between the bids in IE and B are similar in state (1,1). Even when there is some asymmetry in the states, as long both states are low the increased information in IE causes the firm with a higher ω to bid more aggressively in IE than in B. For example at (2,0), the firm with ω = 2 participates 95% of the time in IE (versus 72% of the time in B) and the IE firm bids 1.5 or more 91% of the time (versus 64% of the time in B).

Table 6: Competition in low ω -tuples

|                | Prob.   | Dist. (%)   | Bids   | Bids   | Bids   | Bids   | Bids   | Bids   | Bids   | Bids   | Bids   | Bids   | Profit   | Profit   |
|----------------|---------|-------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|----------|----------|
| ( ω i ,ω - i ) | B       | IE          | B      | B      | B      | B      | B      | IE     | IE     | IE     | IE     | IE     | B        | IE       |
|                |         |             | ∅      | .5     | 1      | 1.5    | 2      | ∅      | .5     | 1      | 1.5    | 2      |          |          |
| (0 , 0)        | 3.17    | .50         | .12    | .07    | .12    | .41    | .28    | .01    | .00    | .09    | .12    | .78    | -.22     | -.48     |
| (0 , 1)        | 3.70    | .88         | .12    | .08    | .13    | .46    | .20    | .04    | .00    | .09    | .44    | .43    | -.17     | -.44     |
| (0 , 2)        | 4.91    | 1.48        | .11    | .09    | .17    | .49    | .15    | .05    | .08    | .05    | .60    | .23    | -.09     | -.31     |
| (1 , 0)        | 3.70    | .88         | .18    | .06    | .13    | .49    | .15    | .01    | .04    | .00    | .29    | .66    | .41      | -.08     |
| (1 , 1)        | 2.36    | .80         | .18    | .12    | .23    | .40    | .07    | .03    | .09    | .00    | .74    | .15    | .46      | .20      |
| (2 , 0)        | 4.91    | 1.48        | .28    | .07    | .19    | .41    | .05    | .05    | .10    | .00    | .86    | .00    | 1.01     | .66      |

Notes: This table shows the probability of selected ω -tuples and bids by those ω -tuples for B and IE .

Table 7 focuses on bidding behavior when states are asymmetric. The firm with the larger stock has an ω = 7 but the pattern is representative of bidding in states in which its ω ∈ { 5 , 6 , 7 , 8 , 9 } . Relative to the B equilibrium the low ω firms in IE have a higher propensity to bid and, when bidding, to bid the minimum bid. Moreover those propensities increases as their state moves from 0 to 1 to 2. By contrast, at least for the couples (7,0), (7,1), and (7,2), the highω rival either does not participate or tends

to bid 1 (and so is likely to win if it does bid). As the low ω firm's stock increases, the high ω firm participates less. So the low ω firm is likely to win more often, and if it does win, it wins with the minimal bid. This insures that both firms profits increase as the low ω firm's state increases.

In the IE equilibrium this pattern of play shifts as the low ω firm passes ω =4. Then the high ω firm (if it bids) moves its bids toward the minimal bid, so that by the time the state (7,7) is reached each firm either does not participate or bids the minimal amount (in about equal proportions). The behavior in the B equilibrium in these cases is quite different. Participation and bids conditional on participation are higher, making the relative profitability of those states (relative to the low ω states) less profitable in the B than in the IE equilibrium.

Table 7: Bidding and participation in asymmetric ω -tuples

|                | Prob. Dist.   | (%)   | Bids   | Bids   | Bids   | Bids   | Bids   | Bids   | Bids   | Bids   | Bids   | Bids   | Profit   | Profit   |
|----------------|---------------|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|----------|----------|
| ( ω i ,ω - i ) | B             | IE    | B      | B      | B      | B      | B      | IE     | IE     | IE     | IE     | IE     | B        | IE       |
|                |               |       | ∅      | .5     | 1      | 1.5    | 2      | ∅      | .5     | 1      | 1.5    | 2      |          |          |
| (0 , 7)        | 1.49          | 2.36  | .05    | .23    | .61    | .09    | .03    | .01    | .33    | .62    | .03    | .00    | .22      | .02      |
| (1 , 7)        | .40           | .83   | .08    | .50    | .38    | .03    | .01    | .00    | .79    | .21    | .00    | .00    | .69      | .64      |
| (2 , 7)        | .35           | .89   | .14    | .64    | 0.18   | 0.02   | 0.01   | 0.00   | 1.00   | 0.00   | 0.00   | 0.00   | 1.06     | 1.07     |
| (4 , 7)        | 0.13          | 0.69  | 0.26   | 0.61   | 0.10   | 0.02   | 0.02   | 0.04   | 0.96   | 0.00   | 0.00   | 0.00   | 1.36     | 1.09     |
| (7 , 0)        | 1.49          | 2.36  | 0.46   | 0.10   | 0.41   | 0.03   | 0.01   | 0.26   | 0.00   | 0.74   | 0.00   | 0.00   | 1.55     | 1.17     |
| (7 , 1)        | 0.40          | 0.83  | 0.48   | 0.23   | 0.26   | 0.02   | 0.00   | 0.40   | 0.03   | 0.57   | 0.00   | 0.00   | 1.57     | 1.21     |
| (7 , 2)        | 0.35          | 0.89  | 0.48   | 0.29   | 0.21   | 0.02   | 0.00   | 0.50   | 0.11   | 0.39   | 0.00   | 0.00   | 1.57     | 1.39     |
| (7 , 4)        | 0.13          | 0.69  | 0.46   | 0.43   | 0.09   | 0.02   | 0.01   | 0.76   | 0.24   | 0.00   | 0.00   | 0.00   | 1.59     | 1.84     |
| (7 , 7)        | 0.02          | 0.26  | 0.45   | 0.47   | 0.06   | 0.01   | 0.00   | 0.47   | 0.53   | 0.00   | 0.00   | 0.00   | 1.61     | 1.49     |

Notes: This table shows the probability of selected ω -tuples and bids by those ω -tuples for B and IE .

Tables 6 and 7 are central to understanding how increasing a firm's information about its competitor changes the path of play. Providing more information about a competitor increases competition at low ω states which reduces profits in those states. In a static game a fall in profits that accompanies the increase in information would decrease participation. However here, despite the fact that profits are lower, participation is higher in the game with more information. This because firms respond to the possibility of higher future profits if they increase their stock of timber, and an increase in information at low ω states intensifies the competition over those future profits. Consider two firms initially at low ω states. A firm that does win the initial auctions proceeds to a higher ω state, and then participates less often in subsequent auctions. Compared to the B equilibrium, firms in the IE equilibrium are better able to asses when their competitor has a large stock. So a firm that loses the initial auctions is more certain of the extent to which the winning firm's stock increases and knows that when the increase is large enough its competitor is less likely to participate in the auction. Thus the firm with a low ω knows that it is likely to win subsequent auctions with a minimal bid and bids accordingly. Though this certainly does not dull the incentive to bid aggressively when both stocks are low, it does ameliorate the consequences of initial losses and support an equilibrium where both firms are at high ω (and hence highly profitable) states more often.

More generally the reduction in asymmetric information, caused by moving from B to IE , intensifies competition in lowω -tuples (causing a reduction in profits in those states) but mitigates competition in high ω states; and so colors competition throughout the recurrent class. 27 The result is an environment in which firms invest in maintaining higher ω stocks, and thus spend more time in parts of the state-space in which competition is less intense. Somewhat perversely this occurs precisely through the intensification of competition caused by a reduction in asymmetric information in those states (lowω -tuples) in which competition was most vigorous to start with.

## 4.3.1 The Model with Static Incentives (i.e. β = 0 ).

Note that when we set β = 0 the firms still use the prior history as signals on the likely current stock of timber held by their competitors. However they now bid to maximize current profits with no interest in investing for future use. The striking implication of the computational results in Table 8 is that when there is no incentive to invest in the future, whether or not firms share information has little impact on their behavior. Apparently the primary impact of the additional information in the IE equilibrium is to enable the firms to plan for the future, and this, in turn, changes the equilibrium distribution of states 28 .

<!-- formula-not-decoded -->

|                                                                                               | β = 0 . 9   | β = 0 . 9   | β = 0   | β = 0   |
|-----------------------------------------------------------------------------------------------|-------------|-------------|---------|---------|
|                                                                                               | B           | IE          | B       | IE      |
| Avg. bid                                                                                      | 1.09        | 0.94        | 0.61    | 0.59    |
| Avg. winning bid (revenue for the auctioneer)                                                 | 1.11        | 0.98        | 0.54    | 0.53    |
| Avg. winning bid conditional on ≥ 1 firm participating                                        | 1.16        | 0.98        | 0.62    | 0.60    |
| Avg. winning bid conditional on 1 firm participating                                          | 1.06        | 0.67        | 0.55    | 0.53    |
| Avg. winning bid conditional on 2 firms participating                                         | 1.23        | 1.16        | 0.82    | 0.82    |
| Avg. # of participants                                                                        | 1.52        | 1.63        | 1.10    | 1.10    |
| Avg. # of participants conditional on ≥ one firm participating                                | 1.59        | 1.63        | 1.25    | 1.25    |
| Avg. participation rate                                                                       | 0.76        | 0.81        | 0.55    | 0.55    |
| % of periods with no participation                                                            | 4.39        | 0.15        | 11.98   | 11.65   |
| Avg. total revenue                                                                            | 3.35        | 3.49        | 3.08    | 3.09    |
| Avg. profit                                                                                   | 0.81        | 0.87        | 1.03    | 1.04    |
| % of periods in which a firm with the lowest omega wins conditional on ≥ 1 firm participating | 66.37       | 60.80       | 96.24   | 96.15   |
| Average total social surplus                                                                  | 2.73        | 2.72        | 2.60    | 2.61    |

## 4.3.2 Voluntary information exchange ( V IE )

In the V IE model firms can elect, every 4 periods, to share information. If both firms elect to share information then the model switches, for the next four periods, from the B to the IE setting. After the four periods they chose between B and IE again. If one or both firms choose not to share, then firms spend the next four periods in the B setting.

27 Recall that the recurrent class are those states visited repeatedly in the course of equilibrium play.

28 We have also computed for β ∈ [ . 25 , . 5 , . 8]. As we increase β the difference between the IE and B equilibria in the rows of tables analogous to Table 8 grows.

Table 9 indicates that despite the fact that average profits in IE are larger than average profits in B, firms in V IE only choose to share information in 5% of the states where that choice is made (though one of the two firms choses to share in 24% of those states). As a result when we calculated the prior tables there was little difference between B and V IE . This raises the question of why firms in V IE cannot reliably coordinate on sharing information; after all it appears to be in their long term interest.

Table 9: Individual firm's choices to reveal by ω -tuple

|                 | Prob. Dist. (%)   | Pr( ∪ i χ i ≥ 1)   | Pr(Π i χ i = 1   | Profit   | Profit   |
|-----------------|-------------------|--------------------|------------------|----------|----------|
| ( ω i ,ω - i )  | V IE              | V IE               | V IE             | B        | IE       |
| ( ≤ 4 , ≤ 4)    | 62.98             | 24.75              | 4.76             | 0.68     | 0.52     |
| ( ≤ 4 , 5 - 7)  | 13.17             | 24.57              | 4.47             | 0.57     | 0.58     |
| ( ≤ 4 , ≥ 8)    | 4.58              | 28.06              | 6.09             | 0.60     | 0.59     |
| (5 - 7 , ≤ 4)   | 13.17             | 21.38              | 4.47             | 1.51     | 1.26     |
| (5 - 7 , 5 - 7) | 1.13              | 18.94              | 4.59             | 1.49     | 1.46     |
| (5 - 7 , ≥ 8)   | 0.19              | 24.38              | 9.73             | 1.49     | 1.13     |
| ( ≥ 8 , ≤ 4)    | 4.58              | 23.39              | 6.09             | 1.62     | 1.58     |
| ( ≥ 8 , 5 - 7)  | 0.19              | 24.60              | 9.73             | 1.66     | 1.87     |
| ( ≥ 8 , ≥ 8)    | 0.02              | 38.14              | 20.34            | 1.72     | 1.56     |

Notes: χ i ∈ { 0 , 1 } , χ i = 1 indicates that firm i chose to reveal, so ∪ i χ i ≥ 1 indicates that at least one firm chose to reveal and Π i χ i = 1 indicates both firms chose to reveal. Only periods in which firms decide on information sharing (or periods with τ = 0) are used in the calculation.

Table 9 shows that the propensity to share information is substantial only when both ω 's are greater than 4, and the highest is greater than 8. Since the default is B , in VIE these states occur relatively rarely, hence the low frequency of choosing to share information. Recall that profits are higher in the B equilibrium. As a result to enjoy the benefit of switching to the IE equilibrium the firm has to forsake profits in an intermediate period.

This tradeoff comes out clearly in the comparison presented in Table 10. It reports, for IE , the average of E F i [ V ( J i , F i ) | τ = 1] by the underlying state's ω i , weighted by the relative frequency with which a state is visited. It also reports the same expectation for an alternate scenario, in which optimal policies from the B model are followed (from the same initial state) for four periods, and then, for all subsequent states, IE -optimal policies are followed. Comparing the two expected valuations indicates the value of switching from no-information sharing directly to information sharing versus waiting four periods and then shifting to information sharing. The last column reports the frequency, in the simulated data, with which the value for IE was larger than the calculation with four periods of waiting; i.e. the fraction of times when any losses in the interim four periods of information exchange are worth less than any gains from information sharing in subsequent periods.

Tables 9 and 10 show the difficulty that the collective of firms have in maintaining information sharing, despite it's long term benefits. This suggests the importance of commitment devices in establishing an effective information sharing arrangement. In

Table 10: E F i [ V ( J i , F i ) | τ = 1] by ω i

| ω i   |   Number of states |   IE (A) |   B for 4 periods, then IE (B) |   Probability of (A) ≥ (B) |
|-------|--------------------|----------|--------------------------------|----------------------------|
| 0     |                146 |     6.22 |                           6.34 |                      22.92 |
| 1     |                120 |     6.89 |                           7.01 |                      32.57 |
| 2     |                131 |     7.72 |                           7.79 |                      36.47 |
| 3     |                136 |     8.54 |                           8.58 |                      29.87 |
| 4     |                127 |     9.35 |                           9.3  |                      63.57 |
| 5     |                120 |    10.1  |                          10.02 |                      44.79 |
| 6     |                113 |    10.87 |                          10.7  |                      75.12 |
| 7     |                 94 |    11.6  |                          11.37 |                      87.34 |
| 8     |                 87 |    12.27 |                          11.98 |                      90.58 |
| 9     |                 75 |    12.86 |                          12.52 |                      94.66 |
| 10    |                 63 |    13.4  |                          13.02 |                      99.93 |
| 11+   |                186 |    14.25 |                          13.88 |                      99.53 |

Notes: This table shows, for IE , the average of E F i [ V ( J i , F i ) | τ = 1] by the underlying state's ω i , weighted by the relative frequency with which a state is visited during a 1 million iteration simulation of the B model. It then replaces the first four periods of IE by B (and the IE continuation from the resulting end state) to form the same computation for ' B for 4 periods, then IE '. States are selected by taking all τ = 1 states visited during a 1 million iteration simulation of the B model. The number of states is the count of distinct states. The probability of (A) ≥ (B) is % of times with (A) ≥ (B) during a 1 million iteration simulation of the B model.

IE perfect commitment is externally imposed. In VIE, firms are able to commit for only four periods at a time and this is sufficient to break down information sharing.

## 5 Conclusion

This paper illustrates how the Experience Based Equilibrium concept facilitates investigation of the dynamics of complex auction environments. It also extends this equilibrium concept through a Boundary Consistency requirement which mitigates the the problem of multiplicity that can be generated by the conditions of Experience Based Equilibrium. Our example shows that allowing for the dynamics implicit in many auction environments is important in that it enables the emergence of equilibrium states that can only be reached when firms are responding to dynamic incentives. It also shows that the impact of information sharing can depend crucially on the extent of dynamics and suggests that treating information sharing, even of strategically important data, as a per se offense (in the case of U.S.) or as a restriction of competition by object (in the case of the E.U.), needs to be weighed against the possibility of type 1 error, falsely rejecting the hypothesis that conduct is pro-competitive.

## 6 Appendix A

## 6.1 Testing for REBE

In this appendix we discuss the testing for REBE and the boundary consistency for the baseline case. Analogous procedures are used for the IE and VIE case.

Notation and Memory. Iterations of the test will be denoted by l (in contrast to the k notations for iterations of the algorithm for computing policies). At each iteration there will be two information sets, one for each firm, so s l ≡ ( J 1 ,l , J 2 ,l ). In storage we have particular values of ( { W ( b | J i ) } b ∈ B , W (0 | J i ) } ) , say ( { W ∗ ( b | J i ) } b ∈ B , W ∗ (0 | J i ) } ) , for all J i with positive counters ( h ∗ ( J i ) &gt; 0), and our goal is to determine whether these values satisfy the conditions of a REBE.

At each point visited during the simulation run we draw an F i for each firm and calculate

<!-- formula-not-decoded -->

The argmax of this equation for each firm will be denoted with a star. Together with the random draws that determine the quantity of timber in the newly acquired lot together with those determining the harvest, these policies generate the next state. However since we are calculating a REBE we need to simulate the continuation values for all possible policies, i.e. for b ∈ B ∪ ∅ .

That is, at iteration l we calculate the simulated continuation values for firm i and policy b as

<!-- formula-not-decoded -->

We also calculate SCV l ( b | J l I ) 2 . We then update our memory for that point which consists of; an average of the simulated continuation values, an average of the square of the simulated values, and the counter for the number of times we have visitied that point.

Say we stop the simulation routine at a particular l = l at that point we have in memory an average of the estimated simulation value for each possible policy at each point visited more than once

<!-- formula-not-decoded -->

and can calculate an unbiased estimate of the variance of the simulated continuation values for each policy at every point

<!-- formula-not-decoded -->

Omitting the index l for notational convenience and letting # B be the cardinality of the set B plus one (for choosing not to enter), we note that the percentage means

square error of our estimates at W ∗ ( J i ) or

<!-- formula-not-decoded -->

where if E ( · ) takes expectations over the simulated draws,

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Our test statistic, labelled Υ, converges to an L 2 ( P ns | W ∗ ) norm in the percentage bias of the our estimates of W ∗ , where P ns is the empirical measure of the number of times each J i is visited in the simulation run (this will converge to L 2 ( P R| W ∗ ) , the invariant measure of a recurrent class generated by W ∗ ). To obtain a consistent estimate of Υ we note that

<!-- formula-not-decoded -->

so that

<!-- formula-not-decoded -->

We accept the test when Υ ≤ . 001.

## 6.2 Appendix B: NOT FOR PUBLICATION

Table 11: Probability Distribution and Actions by ω -tuple for B and IE

| ( ω i ,ω - i )          | Prob. B             | Dist. (%) IE   |                |                | B                   | Bids           | Bids           | Bids                |                     | IE                       |                |                          | Profit B IE              | Joint B                  | Profit IE           | Profit IE           |
|-------------------------|---------------------|----------------|----------------|----------------|---------------------|----------------|----------------|---------------------|---------------------|--------------------------|----------------|--------------------------|--------------------------|--------------------------|---------------------|---------------------|
| (0 , 0)                 | 3.17                | 0.50           | 0 0.12         | 0.5 0.07       | 1 0.12              | 1.5 0.41       | 2 0.28         | 0 0.01              | 0.5 0.00            | 1 0.09                   | 1.5 0.12       | 2 0.78                   | -0.22 -0.48              | -0.43                    | -0.97               | -0.97               |
| (0 , 1)                 | 3.70                | 0.88           | 0.12           | 0.08           | 0.13                | 0.46           | 0.20           | 0.04                | 0.00                | 0.09                     | 0.44           | 0.43                     | -0.17 -0.44 -0.09        | 0.24                     | -0.53               | -0.53               |
| (0 , 2)                 | 4.91                | 1.48           | 0.11           | 0.09           | 0.17                | 0.49           | 0.15           | 0.05                | 0.08                | 0.05                     | 0.60 0.71      | 0.23                     | -0.31                    | 0.92                     | 0.35                | 0.35                |
| (0 , 3)                 | 4.83                | 1.94           | 0.10           | 0.10           | 0.25                | 0.47 0.38      | 0.08           | 0.03                | 0.17                | 0.02 0.17                | 0.62           | 0.07 0.00                | -0.02 -0.26              | 1.32 1.40                | 0.52                | 0.52                |
| (0 , 4) (0 , 5)         | 3.83 3.02           | 2.27 2.47      | 0.08 0.07      | 0.13 0.15      | 0.39 0.53           | 0.24           | 0.03 0.02      | 0.02 0.02           | 0.19 0.20           | 0.38                     | 0.40           | 0.00                     | 0.04 -0.24 0.14 -0.09    | 1.63                     | 0.49 0.94           | 0.49 0.94           |
| (0 , 6)                 | 2.19                | 2.48           | 0.06           | 0.18           | 0.61                | 0.13           | 0.02           | 0.02                | 0.19                | 0.61                     | 0.18           | 0.00                     | 0.19 -0.00               | 1.72                     | 1.15                | 1.15                |
| (0 , 7)                 | 1.49                | 2.36           | 0.05           | 0.23           | 0.61                | 0.09           | 0.03           | 0.01                | 0.33                | 0.62                     | 0.03           | 0.00 0.00                | 0.22 0.02 -0.01          | 1.76 1.78                | 1.19                | 1.19                |
| (0 , 8) (0 , 9)         | 0.97 0.64           | 2.14 1.80      | 0.04 0.04      | 0.36 0.51      | 0.53 0.41           | 0.05 0.04      | 0.02 0.01      | 0.00 0.00           | 0.58 0.82           | 0.42 0.18                | 0.00 0.00      | 0.00                     | 0.24 0.31 0.17           | 1.86                     | 1.13 1.47           | 1.13 1.47           |
| (0 , 10)                | 0.41 0.53           | 1.29 1.21      | 0.04 0.03      | 0.64 0.78      | 0.27 0.12           | 0.04 0.05      | 0.01 0.02      | 0.00 0.00           | 1.00 1.00           | 0.00 0.00                | 0.00 0.00      | 0.00 0.00                | 0.38 0.69 0.50 0.81      | 1.99 2.22                | 2.44 2.67           | 2.44 2.67           |
| (0 , 11+)               |                     |                | 0.18           | 0.06           | 0.13                | 0.49           | 0.15           | 0.01                |                     | 0.00                     | 0.29           | 0.66                     | -0.08                    | 0.24                     | -0.53               | -0.53               |
| (1 , 0) (1 , 1)         | 3.70 2.36           | 0.88 0.80      | 0.18           | 0.12           | 0.23                | 0.40           | 0.07           | 0.03                | 0.04 0.09           | 0.00                     | 0.74           | 0.15                     | 0.41 0.46 0.49           | 0.93                     | 0.39                | 0.39                |
| (1 , 2)                 | 2.54                | 1.07           | 0.17           | 0.14           | 0.32                | 0.32           | 0.05           | 0.03                | 0.10                | 0.07                     | 0.81           | 0.00                     | 0.20 0.32                | 1.52                     | 0.96                | 0.96                |
| (1 , 3)                 | 2.09                | 1.16           | 0.15           | 0.16           | 0.43                | 0.23           | 0.02           | 0.02                | 0.13                | 0.33                     | 0.53           | 0.00                     | 0.55 0.32                | 1.89                     | 1.19                | 1.19                |
| (1 , 4) (1 , 5)         | 1.42                | 1.13           | 0.13           | 0.22           | 0.52                | 0.13           | 0.01 0.01      | 0.01                | 0.16                | 0.59 0.72                | 0.24           | 0.00 0.00                | 0.59 0.40                | 1.97                     | 1.45                | 1.45                |
| (1 , 6)                 | 0.98 0.64           | 1.08 0.97      | 0.11 0.08      | 0.29 0.43      | 0.51 0.43           | 0.08 0.05      | 0.01           | 0.00 0.00           | 0.28 0.52           | 0.48                     | 0.00 0.00      | 0.00                     | 0.62 0.52 0.66 0.55      | 2.10                     | 1.65                | 1.65                |
| (1 , 7)                 | 0.40                | 0.83           | 0.08           | 0.50           | 0.38                | 0.03           | 0.01           | 0.00                |                     | 0.21                     | 0.00           | 0.00                     | 0.69                     | 2.20 2.26                | 1.74 1.85           | 1.74 1.85           |
|                         |                     |                |                |                |                     |                |                |                     | 0.79                |                          |                |                          | 0.64 0.82                |                          |                     |                     |
| (1 , 8) (1 , 9)         | 0.24                | 0.63           | 0.07 0.08      | 0.62           | 0.27                | 0.03 0.03      | 0.01           | 0.00 0.00           | 1.00 1.00           | 0.00 0.00                | 0.00 0.00      | 0.00 0.00                | 0.74 0.75 0.89           | 2.37                     | 2.34                | 2.34                |
| (1 , 10)                | 0.14 0.08 0.09      | 0.40 0.19      | 0.08           | 0.68 0.70      | 0.19 0.16           | 0.03           | 0.02 0.03      | 0.00                |                     | 0.00 0.00                | 0.00 0.00      | 0.00                     | 0.95                     | 2.40 2.47                | 2.63 2.84           | 2.63 2.84           |
| (1 , 11+)               |                     | 0.10           | 0.10           | 0.69           | 0.15                | 0.04           | 0.03           | 0.00                | 1.00 1.00           |                          |                | 0.00                     | 0.76 0.79 0.97           | 2.56                     | 2.92                | 2.92                |
| (2 , 0)                 | 4.91                | 1.48           | 0.28           | 0.07           | 0.19                | 0.41           | 0.05           | 0.05                | 0.10                | 0.00                     | 0.86           | 0.00                     | 1.01                     | 0.92                     | 0.35                | 0.35                |
| (2 , 1) (2 , 2)         | 2.54 2.57           | 1.07 1.32      | 0.28 0.26      | 0.14 0.17      | 0.27                | 0.29 0.22      | 0.02 0.01      | 0.06 0.04           | 0.09 0.18           | 0.00 0.11                | 0.85 0.66      | 0.00 0.00                | 0.66 1.03 0.64 1.01 0.62 | 1.52 2.02                | 0.96 1.24           | 0.96 1.24           |
| (2 , 3)                 | 2.02 1.33           | 1.36           | 0.24           | 0.23           | 0.34 0.39           | 0.13           | 0.01           | 0.03                | 0.21                | 0.39                     | 0.37           |                          | 0.72 0.86                |                          | 1.66 1.94           | 1.66 1.94           |
| (2 , 4)                 |                     | 1.26           | 0.21           | 0.32           | 0.40                | 0.06           | 0.01           | 0.01                | 0.29                | 0.65                     | 0.05           | 0.00 0.00 0.00           | 1.02 1.04 1.06           | 2.36 2.40                |                     |                     |
| (2 , 5) (2 , 6) (2 , 7) | 0.91 0.58           | 1.20 1.06      | 0.18 0.15      | 0.45 0.57      | 0.32 0.24           | 0.04 0.03      | 0.01 0.01      | 0.01 0.01           | 0.43 0.73           | 0.56 0.27                | 0.00 0.00      | 1.06 1.06                | 0.92 0.97                | 2.51 2.59 2.63           | 2.02                | 2.02                |
|                         | 0.35                | 0.89           | 0.14           | 0.64           | 0.18                | 0.02           | 0.01           |                     | 1.00                | 0.00                     | 0.00           | 0.00 0.00                | 1.07                     |                          | 2.27                | 2.27                |
| (2 , 8)                 | 0.22                | 0.62           | 0.14           | 0.69           | 0.14                | 0.02           |                | 0.00 0.00           |                     | 0.00 0.00                | 0.00           |                          | 1.03                     | 2.71                     | 2.46                | 2.46                |
| (2 , 9) (2 , 10)        | 0.13                | 0.37 0.17      | 0.14           | 0.70           |                     | 0.03           | 0.01 0.01      | 0.00                | 1.00                | 0.00                     | 0.00 0.00 0.00 | 1.08 1.08 1.06           | 1.01 1.01                | 2.73 2.78                |                     |                     |
|                         | 0.07                |                |                | 0.72           |                     | 0.03           | 0.01           |                     | 1.00 1.00           |                          | 0.00 0.00      | 1.08                     |                          |                          | 2.75 2.87 2.92      | 2.75 2.87 2.92      |
| (2 , 11+) (3 , 0)       | 0.07 4.83           | 0.09 1.94      | 0.13 0.17 0.35 | 0.68 0.07      | 0.12 0.11 0.11 0.26 | 0.03           | 0.01 0.02      | 0.00 0.00 0.06      | 1.00 0.03           | 0.00 0.07                | 0.00 0.83      | 0.00 0.00                | 1.00 1.34 0.78           | 2.87 1.32                | 2.97 0.52 1.19      | 2.97 0.52 1.19      |
| (3 , 1) (3 , 2) (3 , 3) | 2.09 2.02 1.54      | 1.16 1.36 1.34 | 0.34 0.33 0.31 | 0.16 0.22 0.30 | 0.31 0.33 0.32      | 0.30 0.16 0.11 | 0.02 0.02 0.01 | 0.12                | 0.10 0.13 0.20      | 0.23 0.39 0.57           | 0.55 0.35 0.12 | 0.00 1.34 0.00 1.33 1.34 | 0.87 0.93 1.03           | 1.89                     | 1.66 2.06           | 1.66 2.06           |
| (3 , 4)                 | 0.97                | 1.22           |                |                |                     | 0.06           |                | 0.12 0.11           |                     |                          | 0.00           | 1.35                     | 1.20                     | 2.36 2.68 2.72           | 2.20                | 2.20                |
| (3 , 5)                 | 0.65                | 1.17           | 0.28           | 0.40           | 0.28                | 0.04           | 0.00           | 0.07 0.04           | 0.43                | 0.50 0.29                | 0.00           | 0.00 0.00 0.00           | 1.19 1.16                |                          | 2.35                | 2.35                |
| (3 , 6)                 | 0.41                | 1.03           | 0.25 0.22      | 0.48 0.57      | 0.23 0.19           | 0.03 0.03      | 0.00 0.00      |                     | 0.67 0.93           | 0.05                     | 0.00 0.00      |                          | 1.13                     | 2.79                     | 2.65                | 2.65                |
| (3 , 7)                 | 0.25                | 0.80           | 0.20           | 0.61           | 0.16                | 0.03           | 0.01           | 0.01                | 1.00                | 0.00                     |                | 1.33                     | 1.06                     |                          |                     |                     |
| (3 , 8)                 | 0.15                | 0.51           | 0.21           | 0.67           |                     |                | 0.01           |                     |                     | 0.00                     | 0.00           | 0.00 0.00 0.00           | 1.30 1.27 1.31           | 2.84 2.88 2.94           |                     |                     |
| (3 , 9)                 | 0.08                |                | 0.21           | 0.67           | 0.09 0.09           | 0.02 0.02      | 0.01           | 0.00 0.00 0.00      |                     | 0.00                     | 0.00 0.00      | 1.28 1.28 1.33           | 1.05 1.02                | 2.97                     | 2.78 2.91 2.95      | 2.78 2.91 2.95      |
| (3 , 10) (3 , 11+)      | 0.05 0.05           | 0.27 0.11 0.06 | 0.22 0.27      | 0.65 0.63      | 0.11 0.09           | 0.01 0.01      | 0.01 0.00      | 0.00 0.00           | 1.00 1.00 1.00 1.00 | 0.00 0.00                | 0.00           | 0.00 0.00 0.00           | 1.02 0.73                | 3.00 3.10                | 2.98 2.98           | 2.98 2.98           |
| (4 , 0) (4 , 1)         | 3.83 1.42           | 2.27 1.13      | 0.37 0.35      | 0.06 0.14      | 0.34 0.40           | 0.22 0.10      | 0.01 0.01      | 0.09 0.17           |                     | 0.13 0.61                | 0.78 0.23 0.05 |                          | 1.06 1.08                | 1.40 1.97 2.40           | 0.49 1.45 1.94      | 0.49 1.45 1.94      |
| (4 , 2) (4 , 3)         | 1.33 0.97           | 1.26           | 0.35           | 0.19           | 0.38                | 0.07           | 0.01           | 0.17 0.16           | 0.00 0.00 0.00      | 0.78                     | 0.00           | 0.00 0.00 0.00 0.00      | 1.36 1.38 1.36 1.36      | 2.72                     | 2.20                | 2.20                |
| (4 , 4) (4 , 5)         |                     | 1.22 1.11 1.11 | 0.34 0.31 0.29 | 0.29 0.44 0.51 | 0.33 0.22           | 0.03 0.02 0.02 | 0.01 0.01 0.01 |                     | 0.07 0.32 0.61      | 0.77 0.56 0.30           | 0.00 0.00      | 0.00 0.00 0.00           | 1.00 1.40 1.38 1.35      | 2.80 2.87 2.88           |                     |                     |
| (4 , 6) (4 , 7)         |                     | 0.95           | 0.26           | 0.57           | 0.17 0.13           | 0.02           | 0.02 0.02      |                     |                     | 0.00 0.00                |                | 0.00                     | 1.36                     | 2.95                     |                     |                     |
| (4 , 8)                 | 0.58 0.38           | 0.69           | 0.26           | 0.61           | 0.10                | 0.02           |                | 0.13 0.10           | 0.85 0.96           | 0.08 0.00                | 0.00           |                          | 1.13 1.13 1.13 1.09 1.04 |                          | 2.25 2.47           | 2.25 2.47           |
| (4 , 9) (4 , 10)        | 0.23 0.13 0.07 0.04 | 0.38 0.17      | 0.27 0.25      | 0.62 0.63      | 0.08 0.09           | 0.02 0.03      | 0.01 0.01 0.01 | 0.08 0.04           | 0.98 1.00 1.00      | 0.00 0.00                | 0.00 0.00      |                          | 1.35 1.33                | 2.97 2.99 3.08           | 2.81 2.94 2.98 3.00 | 2.81 2.94 2.98 3.00 |
| (4 , 11+)               | 0.02                | 0.06           | 0.32           | 0.56           | 0.09                | 0.02           | 0.00           | 0.02 0.00 0.00 0.05 | 0.95                | 0.00 0.00                | 0.00           | 0.00 0.00                | 1.38 1.46                |                          |                     |                     |
|                         | 0.02                | 0.02           | 0.40           | 0.52           | 0.07                | 0.01           |                |                     |                     |                          | 0.00 0.00 0.00 | 1.49                     | 1.02 0.98 1.06 1.02      | 3.24                     | 2.98                | 2.98                |
| (5 , 0) (5 , 1)         |                     | 2.47 1.08      | 0.41 0.41      | 0.09 0.19      | 0.42 0.35           | 0.08 0.05      | 0.01 0.01 0.00 | 0.16 0.24           | 0.00 0.00           | 0.51 0.71                | 0.34 0.05      | 0.00                     | 1.48                     | 1.63 2.10 2.51           | 2.96 0.94 1.65 2.02 | 2.96 0.94 1.65 2.02 |
| (5 , 2) (5 , 3)         |                     | 1.20           | 0.40           | 0.25           | 0.30                | 0.04           |                | 0.25                | 0.00 0.16           | 0.75 0.56                | 0.00 0.00      | 0.00 0.00                | 1.13 1.45 1.46           | 2.79                     | 2.35                | 2.35                |
| (5 , 4)                 | 3.02 0.98 0.91      | 1.17           | 0.39           | 0.34           |                     | 0.03           | 0.00           | 0.28 0.29           | 0.41                | 0.30                     | 0.00           | 0.00                     | 1.10 1.16                | 2.87                     | 2.47                | 2.47                |
| (5 , 5) (5 , 6) (5 , 7) |                     | 1.11 1.07 0.86 | 0.37 0.35      | 0.44 0.49      | 0.24                | 0.02 0.03 0.03 | 0.01 0.01 0.01 |                     | 0.65 0.79           | 0.07 0.00                | 0.00 0.00      | 0.00 0.00                | 1.49 1.33 1.46 1.39 1.44 |                          | 2.77 2.93           | 2.77 2.93           |
|                         | 0.65 0.38 0.24 0.15 |                | 0.34           | 0.52           | 0.16 0.12 0.10      |                | 0.01           | 0.28 0.21 0.12      | 0.88                |                          | 0.00           |                          | 1.42                     |                          | 2.98                | 2.98                |
| (5 , 8)                 | 0.59                | 0.33           |                | 0.55           | 0.07                | 0.03           | 0.02           | 0.95                | 0.00 0.00           | 0.00                     |                | 0.00                     | 1.29 1.42                | 2.93 2.94 3.03 3.06 3.10 | 3.00                | 3.00                |
| (5 , 9)                 | 0.08 0.04           | 0.32           | 0.33           | 0.57           | 0.06                | 0.01           | 0.01           | 0.05 0.00           | 1.00                | 0.00                     | 0.00           | 0.00 0.00                | 1.16 1.44                |                          | 2.98                | 2.98                |
| (5 , 10)                | 0.02                | 0.14           | 0.35           | 0.58           |                     |                | 0.00           |                     |                     |                          |                |                          | 1.07                     |                          |                     |                     |
|                         | 0.01                | 0.05           | 0.38           | 0.56           | 0.05 0.05           | 0.01 0.01      |                |                     |                     | 0.00                     | 0.00           |                          | 1.46 1.57                | 3.13                     | 3.36                | 2.98 2.98           |
| (5 , 11+) (6 , 0)       | 0.01 2.19           | 0.02 2.48      | 0.47 0.44      | 0.49 0.10      | 0.03 0.43           | 0.01 0.03      | 0.00 0.01      | 0.01 0.10 0.20      | 0.99 0.90 0.00      | 0.00 0.00 0.00 0.74 0.06 |                | 0.00 0.00                | 1.01 1.01 1.17 1.54 1.15 | 1.72                     |                     | 1.15                |

| (6 , 4)                            | 0.23                          | 0.95           | 0.42           | 0.42           | 0.12                     | 0.02                | 0.02                     | 0.54                     | 0.42           | 0.04                     | 0.00           | 0.00           | 1.53 1.67                     | 2.88                | 2.81                | 2.81                | 2.81                | 2.81                | 2.81                |           |
|------------------------------------|-------------------------------|----------------|----------------|----------------|--------------------------|---------------------|--------------------------|--------------------------|----------------|--------------------------|----------------|----------------|-------------------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|-----------|
| (6 , 5) (6 , 6)                    | 0.15 0.86                     | 0.40 0.40      | 0.47           |                | 0.09                     | 0.02                | 0.02                     | 0.54                     | 0.46           | 0.00                     | 0.00           | 0.00           | 1.51 1.64                     | 2.94                | 2.93 2.97           | 2.93 2.97           | 2.93 2.97           | 2.93 2.97           | 2.93 2.97           |           |
| (6 , 7)                            | 0.08 0.65 0.04                | 0.42 0.40      | 0.52           | 0.50           | 0.08 0.07                | 0.02 0.01           | 0.00 0.00                | 0.44 0.31                | 0.56 0.69      | 0.00 0.00                | 0.00 0.00      | 0.00 0.00      | 1.53 1.49 1.52 1.33           | 3.06 3.07           | 2.98                | 2.98                | 2.98                | 2.98                | 2.98                |           |
| (6 , 8)                            | 0.02 0.22                     | 0.39           |                | 0.51           | 0.08                     | 0.01                | 0.00                     | 0.15                     | 0.85           | 0.00                     | 0.00           | 0.00           | 1.48 1.17                     | 3.10                | 2.99                | 2.99                | 2.99                | 2.99                | 2.99                |           |
| (6 , 9)                            | 0.01                          | 0.45           |                | 0.48           | 0.05                     | 0.02                | 0.00                     | 0.06                     | 0.94           | 0.00                     | 0.00           | 0.00           | 1.57 1.10                     | 3.25                | 3.00                | 3.00                | 3.00                | 3.00                | 3.00                |           |
| (6 , 10)                           | 0.10 0.00 0.03                | 0.45           | 0.47           |                | 0.07                     | 0.02                | 0.00                     |                          | 0.93           | 0.00                     | 0.00           | 0.00           | 1.57 1.71                     | 3.17                | 3.00                | 3.00                | 3.00                | 3.00                | 3.00                |           |
| (6 , 11+)                          | 0.00 0.02                     | 0.59           |                | 0.34           | 0.03                     | 0.03                | 0.00                     | 0.07 0.23                | 0.77           | 0.00                     | 0.00           | 0.00           | 1.10 1.31                     | 3.46                | 3.05                | 3.05                | 3.05                | 3.05                | 3.05                |           |
| (7 , 0)                            | 1.49 2.36                     | 0.46           | 0.10           |                | 0.41                     | 0.03                | 0.01                     | 0.26                     | 0.00           | 0.74                     | 0.00           | 0.00           | 1.55 1.17                     | 1.76                | 1.19                | 1.19                | 1.19                | 1.19                | 1.19                |           |
| (7 , 1)                            | 0.40 0.83                     | 0.48           | 0.23           |                | 0.26                     | 0.02                | 0.00                     | 0.40                     | 0.03           | 0.57                     | 0.00           | 0.00           | 1.57 1.21                     | 2.26                | 1.85                | 1.85                | 1.85                | 1.85                | 1.85                |           |
| (7 , 2)                            | 0.35 0.89 0.25                |                | 0.48 0.47      | 0.29           | 0.21                     | 0.02                | 0.00 0.00                | 0.50 0.65                | 0.11 0.17      | 0.39 0.18                | 0.00 0.00      | 0.00 0.00      | 1.57 1.39 1.61                | 2.63 2.88           | 2.46 2.78           | 2.46 2.78           | 2.46 2.78           | 2.46 2.78           | 2.46 2.78           |           |
| (7 , 3)                            | 0.80 0.13 0.69                | 0.46           | 0.36           |                | 0.14                     | 0.02                | 0.01                     | 0.76                     | 0.24           | 0.00                     | 0.00           | 0.00           | 1.65 1.84                     |                     |                     |                     |                     |                     |                     |           |
| (7 , 4)                            | 0.08 0.59                     |                | 0.43           | 0.43           | 0.09 0.09                | 0.02                | 0.00                     | 0.76                     |                | 0.00                     | 0.00           | 0.00           | 1.59 1.61                     | 2.95                | 2.94                | 2.94                | 2.94                | 2.94                | 2.94                |           |
| (7 , 5)                            | 0.42                          | 0.42           | 0.47           |                |                          | 0.01                |                          |                          | 0.24           |                          |                | 0.00           | 1.82 1.65                     | 3.03                | 2.98                | 2.98                | 2.98                | 2.98                | 2.98                |           |
| (7 , 6)                            | 0.04 0.02                     | 0.45           |                | 0.49           | 0.08                     | 0.02                | 0.01                     |                          | 0.37           | 0.00                     | 0.00           |                | 1.55                          | 3.07                | 2.98                | 2.98                | 2.98                | 2.98                | 2.98                |           |
| (7 , 7)                            | 0.26                          |                | 0.47           |                | 0.06                     | 0.01                | 0.00                     | 0.63 0.47                | 0.53           | 0.00                     | 0.00           | 0.00           | 1.61 1.49                     | 3.22                | 2.99                | 2.99                | 2.99                | 2.99                | 2.99                |           |
| (7 , 8)                            | 0.01 0.13                     |                |                | 0.46           | 0.05                     | 0.01                | 0.00                     |                          | 0.75           | 0.00                     | 0.00           | 0.00           | 1.58 1.28                     | 3.16                | 2.98                | 2.98                | 2.98                | 2.98                | 2.98                |           |
| (7 , 9)                            | 0.00 0.06                     | 0.49 0.52      | 0.40           |                | 0.06                     | 0.01                | 0.00                     | 0.25 0.19                | 0.81           | 0.00                     | 0.00           | 0.00           | 1.57 1.24                     | 3.37                | 3.02                | 3.02                | 3.02                | 3.02                | 3.02                |           |
| (7 , 10) (7 , 11+)                 | 0.00 0.02                     | 0.56 0.69      | 0.34 0.27      |                | 0.09                     | 0.01 0.00           | 0.00 0.00                | 0.24 0.34                | 0.76 0.66      | 0.00 0.00                | 0.00 0.00      | 0.00 0.00      | 1.57 1.35                     | 3.28 3.50           | 3.18 3.16           | 3.18 3.16           | 3.18 3.16           | 3.18 3.16           | 3.18 3.16           |           |
|                                    | 0.00 0.01                     |                |                |                | 0.03                     |                     |                          |                          |                |                          |                |                | 1.80 1.44                     |                     |                     |                     |                     |                     |                     |           |
| (8 , 0)                            | 0.97                          | 0.47           | 0.12           |                | 0.38                     | 0.03                | 0.00                     | 0.32                     | 0.02           | 0.66                     | 0.00           | 0.00           | 1.54 1.14                     | 1.78 2.37           | 1.13 2.34           | 1.13 2.34           | 1.13 2.34           | 1.13 2.34           | 1.13 2.34           |           |
| (8 , 1)                            | 2.14 0.24 0.63                |                | 0.32           |                |                          | 0.01                | 0.00                     | 0.56                     | 0.17           | 0.27                     | 0.00           | 0.00           | 1.63 1.52                     |                     |                     |                     |                     |                     |                     |           |
| (8 , 2) (8 , 3)                    | 0.22 0.62                     |                | 0.48 0.49      | 0.38           | 0.18 0.12                | 0.01                | 0.00                     | 0.67                     | 0.21           | 0.12                     | 0.00           | 0.00           | 1.63 1.71 1.64                | 2.71 2.94           | 2.75 2.91           | 2.75 2.91           | 2.75 2.91           | 2.75 2.91           | 2.75 2.91           |           |
| (8 , 4)                            | 0.15 0.07                     | 0.51 0.38      | 0.48 0.48      | 0.42 0.43      | 0.10 0.08                | 0.01 0.01           | 0.00 0.00                | 0.78 0.90                | 0.20 0.10      | 0.03 0.00                | 0.00 0.00      | 0.00 0.00      | 1.85 1.62 1.94                | 2.97                | 2.98                | 2.98                | 2.98                | 2.98                | 2.98                |           |
| (8 , 5) (8 , 6)                    | 0.04                          |                | 0.49 0.49      | 0.43           | 0.07 0.05                | 0.01                | 0.00 0.00                | 0.91                     | 0.09           | 0.00 0.00                | 0.00 0.00      | 0.00 0.00      | 1.93                          | 3.06                | 3.00 2.99           | 3.00 2.99           | 3.00 2.99           | 3.00 2.99           | 3.00 2.99           |           |
| (8 , 7)                            | 0.02                          |                |                |                |                          | 0.02                |                          | 0.80                     |                |                          |                |                | 1.64 1.63 1.58                |                     |                     |                     |                     |                     |                     |           |
|                                    | 0.01                          |                |                |                | 0.04                     |                     |                          |                          | 0.20           |                          |                | 0.00           |                               |                     |                     |                     |                     |                     |                     |           |
|                                    |                               |                |                |                |                          |                     |                          |                          | 0.34           | 0.00                     |                |                |                               | 3.10                |                     |                     |                     |                     |                     |           |
|                                    | 0.32 0.22                     |                | 0.49           | 0.44 0.44      |                          | 0.02                | 0.00                     | 0.66                     |                |                          | 0.00           | 0.00           | 1.82 1.69 1.52                | 3.16 3.38           | 2.98 3.04           | 2.98 3.04           | 2.98 3.04           | 2.98 3.04           | 2.98 3.04           |           |
| (8 , 8) (8 , 9) (8 , 10) (8 , 11+) | 0.13 0.00 0.07 0.00 0.00      | 0.03 0.01      | 0.53 0.56 0.66 | 0.42 0.43 0.28 | 0.04 0.01 0.07 0.00      | 0.01 0.00 0.00 0.00 | 0.00 0.00 0.00 0.00      | 0.45 0.55 0.50 0.42 0.44 | 0.50 0.58      | 0.00 0.00 0.00 0.00 0.00 | 0.00 0.00 0.00 | 0.00 0.00      | 1.69 1.66 1.60 1.56 1.49      | 3.45 3.52 3.51      | 3.18 3.19           | 3.18 3.19           | 3.18 3.19           | 3.18 3.19           | 3.18 3.19           |           |
|                                    | 0.00 0.00                     | 0.83           | 0.17           |                |                          |                     |                          |                          | 0.56           |                          |                | 0.00           | 1.72 1.89                     | 1.86                | 3.08 1.47           | 3.08 1.47           | 3.08 1.47           | 3.08 1.47           | 3.08 1.47           |           |
| (9 , 0) (9 , 1)                    | 0.64 1.80 0.14 0.13           | 0.40 0.37      | 0.50 0.51      | 0.16 0.34      | 0.32 0.12 0.08           | 0.02 0.03           | 0.00 0.01 0.00           | 0.47 0.69 0.21 0.76 0.24 | 0.03           | 0.51 0.11 0.01 0.00      | 0.00 0.00      | 0.00 0.00 0.00 | 1.56 1.29 1.65 1.74 1.65 1.86 | 2.40 2.73           | 2.63 2.87 2.95 3.00 | 2.63 2.87 2.95 3.00 | 2.63 2.87 2.95 3.00 | 2.63 2.87 2.95 3.00 | 2.63 2.87 2.95 3.00 |           |
| (9 , 2) (9 , 3) (9 , 4)            | 0.08 0.27 0.04                |                | 0.51 0.52 0.54 | 0.39 0.38 0.37 | 0.02                     |                     | 0.00 0.01                |                          | 0.16 0.03      | 0.00 0.00                | 0.00 0.00 0.00 | 1.68 1.65 1.66 | 1.90 1.98 1.97                | 2.97                |                     |                     |                     |                     |                     |           |
| (9 , 5)                            |                               | 0.17 0.14      | 0.55           |                | 0.08 0.08                | 0.01 0.01           |                          | 0.84 0.97 0.97           |                | 0.00                     |                | 0.00           |                               | 2.99 3.10           |                     |                     |                     |                     |                     |           |
|                                    | 0.02                          |                |                |                |                          |                     | 0.00                     |                          | 0.03           |                          |                | 0.00 0.00      |                               | 3.25                |                     |                     |                     |                     |                     |           |
| (9 , 6)                            |                               |                |                | 0.37           |                          |                     |                          |                          | 0.12           | 0.00                     | 0.00           |                | 1.90                          |                     | 2.98 3.00 3.02      | 2.98 3.00 3.02      | 2.98 3.00 3.02      | 2.98 3.00 3.02      | 2.98 3.00 3.02      |           |
| (9 , 7)                            | 0.01                          |                | 0.54 0.61      | 0.39           | 0.07 0.06 0.04           | 0.01 0.01           | 0.00                     |                          |                |                          | 0.00           |                | 1.78                          | 3.37                |                     |                     |                     |                     |                     |           |
|                                    | 0.00                          | 0.10 0.06      | 0.35           |                |                          | 0.00                | 0.00                     |                          | 0.27           | 0.00                     |                | 0.00 0.00 0.00 | 1.68 1.80 1.79                | 3.45                | 3.18                | 3.18                | 3.18                | 3.18                | 3.18                |           |
| (9 , 8)                            | 0.00 0.03                     |                | 0.64           | 0.33           | 0.02                     |                     | 0.00                     | 0.88 0.73 0.50           | 0.50           | 0.00                     | 0.00           | 1.77           | 1.58                          |                     |                     |                     |                     |                     |                     |           |
| (9 , 9) (9 , 10) (9 , 11+)         | 0.00 0.01 0.00 0.00 0.00 0.00 | 0.67 0.58 0.83 | 0.31 0.42 0.17 |                | 0.00 0.02 0.00 0.00 0.00 | 0.00 0.00           | 0.00 0.00 0.00           | 0.49 0.42 0.38           | 0.51 0.58 0.62 | 0.00 0.00                | 0.00 0.00      | 0.00 0.00      | 1.57 1.48 1.56 2.04 1.52      | 3.54 3.24 3.73      | 3.14 3.13           | 3.14 3.13           | 3.24                | 3.14 3.13           | 3.14 3.13           |           |
| (10 , 0)                           | 0.41 1.29                     | 0.51           | 0.23           |                | 0.24                     | 0.01                | 0.01                     | 0.63                     | 0.32           | 0.00 0.05                | 0.00 0.00      | 0.00 0.00      | 1.75                          | 1.99 2.47           | 2.44 2.84           | 2.44 2.84           | 2.44 2.84           | 2.44 2.84           | 2.44 2.84           |           |
| (10 , 1)                           | 0.08 0.19                     | 0.54 0.54      |                | 0.35           | 0.09 0.08                | 0.02                | 0.00                     |                          | 0.21           | 0.00                     | 0.00           | 0.00           | 1.61 1.70 1.88                | 2.78                | 2.92                | 2.92                | 2.92                | 2.92                | 2.92                |           |
| (10 , 2) (10 , 3)                  | 0.07 0.17 0.05                |                | 0.37 0.37      |                |                          | 0.01                | 0.00                     | 0.79 0.85 0.90           | 0.15 0.10      | 0.00 0.00                | 0.00 0.00      | 0.00 0.00      | 1.72 1.91 1.72                | 3.00                |                     |                     |                     |                     |                     |           |
| (10 , 4)                           | 0.02                          |                |                | 0.33           |                          |                     |                          | 0.99                     |                | 0.00                     |                |                | 1.70                          | 3.08 3.13           |                     |                     |                     |                     |                     |           |
| (10 , 5)                           | 0.11 0.06 0.01 0.05           |                | 0.56 0.58 0.62 | 0.30           | 0.06 0.08 0.07           | 0.01 0.01           | 0.00 0.00                |                          | 0.01           | 0.00 0.00                | 0.00           | 0.00 0.00      | 1.96 2.00 1.97 1.89           | 3.17                | 2.98 2.98           | 2.98 2.98           | 2.98 2.98           | 2.98 2.98           | 2.98 2.98           |           |
| (10 , 6) (10 , 7)                  | 0.00 0.03                     |                |                | 0.32 0.31      |                          | 0.01 0.02           | 0.00 0.00 0.00           | 0.96 0.86 0.75           | 0.04 0.14 0.25 | 0.00 0.00                | 0.00 0.00      |                | 1.66 1.60 1.71                | 3.28                | 2.98 3.00           | 2.98 3.00           | 2.98 3.00           | 2.98 3.00           | 2.98 3.00           |           |
| (10 , 8)                           | 0.00                          |                | 0.58 0.66      |                | 0.09 0.02                | 0.01                |                          |                          |                |                          |                | 0.00 0.00 0.00 | 1.81                          | 3.52                |                     |                     |                     |                     |                     |           |
| (10 , 9)                           | 0.02 0.00                     |                | 0.26           |                |                          | 0.00                |                          |                          | 0.49           | 0.00                     | 0.00           | 1.75           | 1.83 1.63 1.56 1.49           |                     |                     |                     |                     |                     |                     |           |
|                                    | 0.01 0.00 0.00                | 0.74           |                | 0.21           |                          | 0.00                | 0.00 0.00                | 0.51                     | 0.53 0.70      | 0.00                     | 0.00 0.00      |                |                               | 3.24                | 3.18 3.19 3.13      | 3.18 3.19 3.13      | 3.18 3.19 3.13      | 3.18 3.19 3.13      | 3.18 3.19 3.13      |           |
| (10 , 10)                          | 0.00                          | 0.79 0.83      | 0.17           |                | 0.00 0.00 0.00           |                     | 0.00                     | 0.47 0.30 0.36           |                | 0.00                     | 0.00           | 0.00           | 1.57                          | 3.13                |                     |                     |                     |                     |                     |           |
| (10 , 11+)                         | 0.00 0.00                     | NaN            | NaN            |                | NaN                      | 0.00 NaN            | NaN                      |                          |                | 0.00                     | 0.00           | 0.00           | NaN 1.48                      | NaN                 | 2.98                | 2.98                | 2.98                | 2.98                | 2.98                |           |
| (11+ , 0)                          | 0.00 0.53                     |                |                | 0.30           | 0.01                     |                     | 0.00                     | 0.75                     | 0.64 0.25      |                          | 0.00           | 0.00           | 1.72 1.86 1.78                |                     | 3.12 2.67           | 3.12 2.67           | 3.12 2.67           | 3.12 2.67           | 3.12 2.67           |           |
| (11+ , 1)                          | 1.21 0.09 0.10                | 0.57 0.62      |                | 0.32 0.32      | 0.12 0.06 0.05           | 0.01 0.00           | 0.00 0.00                | 0.91 0.94 0.94           | 0.09 0.06 0.06 | 0.00 0.00 0.00           | 0.00 0.00      | 0.00 0.00      | 1.94 1.79                     | 2.22 2.56           | 2.92 2.97 2.98      | 2.92 2.97 2.98      | 2.92 2.97 2.98      | 2.92 2.97 2.98      | 2.92 2.97 2.98      |           |
| (11+ , 2) (11+ , 3) (11+ , 4)      | 0.09 0.06 0.02                | 0.63 0.64 0.67 |                | 0.31           |                          | 0.00                |                          |                          | 0.00 0.00      | 0.00 0.00                |                | 0.00 0.00      | 1.97 1.77 1.96 1.78           |                     | 2.96                | 2.96                | 2.96                | 2.96                | 2.96                |           |
| (11+ , 5)                          | 0.07 0.05 0.02                |                |                | 0.29           | 0.05 0.04 0.04           | 0.00                | 0.00 0.00                | 0.89 0.80                | 0.11 0.20      | 0.00                     |                |                | 1.91 1.79                     | 2.87 3.10 3.24 3.36 |                     |                     |                     |                     |                     |           |
| (11+ , 6)                          |                               | 0.65           | 0.31           |                |                          |                     | 0.00 0.00                |                          | 0.00           | 0.00 0.00                |                |                | 1.81                          |                     |                     |                     |                     |                     |                     |           |
| (11+ , 7)                          | 0.01 0.00 0.00 0.00           | 0.66           | 0.29 0.27      |                |                          | 0.00 0.00 0.00      | 0.65 0.00 0.57 0.00 0.50 | 0.35 0.43 0.50           | 0.02           | 0.00 0.00 0.00           | 0.00           |                | 1.76 1.74 1.70 1.62           | 3.46 3.50           |                     | 3.51                | 2.98 3.05 3.16 3.08 |                     |                     |           |
| (11+ , 8) (11+ , 9)                | 0.02 0.01 0.00 0.00           | 0.00 0.83      | 0.69 0.75      | 0.25 0.17      | 0.05 0.03 0.00 0.00      | 0.00 0.00           | 0.00                     | 0.47 0.53                | 0.00 0.00      | 0.00 0.00                | 0.00 0.00      | 0.00           | 0.00 0.00                     | 3.73                | 0.00 0.00           | 0.00 0.00           | 3.24                | 0.00 0.00           | 0.00 0.00           | 0.00 0.00 |

## References

Andrews, Isaiah, and Ariel Pakes (2016), Linear Moment Inequalities, mimeo, Harvard University.

Aoyagi, Masaki (2003), Bid Rotation and Collusion in Repeated Auctions, Journal of Economic Theory , 112, 79-105.

Asker, John (2010), A Study of the Internal Organization of a Bidding Cartel, American Economic Review , 100(3), 724-762.

Athey, Susan and Kyle Bagwell (2008), Collusion with Persistent Cost Shocks, Econometrica , 76(3), 493-540.

Athey, Susan, Jonathan Levin and Enrique Seira (2011), Comparing Open and Sealed Bid Auctions: Evidence from Timber Auctions, Quarterly Journal of Economics , 126(1), 207-257.

Bajari, Patrick and Lixin Ye (2003) Deciding Between Competition and Collusion, The Review of Economics and Statistics , 85(4), 971-989.

Baldwin, Laura, Robert Marshall and Jean-Francois Richard (1997), Bidder Collusion at Forest Service Timber Sales, Journal of Political Economy , 105(4), 657-699.

Balat, Jorge (2015), Highway Procurement and the Stimulus Package: Identification and Estimation of Dynamic Auctions with Unobserved Heterogeneity, Working paper, Johns Hopkins University.

Besanko, David, Ulrich Doraszelski and Yaroslav Kryukov (2014), The Economics of Predation: What Drives Pricing When There Is Learning-by-Doing?, American Economic Review , 104(3), 868-897.

Clarke, Richard (1983), Collusion and the Incentive for Information Sharing, Bell Journal of Economics , 14(2), 383-394.

Doraszelski, Ulrich, and Ariel Pakes (2007), A Framework for Applied Dynamic Analysis in IO. In Armstrong, Mark and Robert Porter (eds.) The Handbook of Industrial Organization Vol. 3. Elsevier, New York.

Doyle, Maura, and Christopher Snyder (1999), Information Sharing and Competition in the Motor Vehicle Industry, Journal of Political Economy , 107, 1326?1364.

Ericson, Richard, and Ariel Pakes (1995), Markov Perfect Industry Dynamics: A Framework for Empirical Work, Review of Economic Studies , 62(1), 53-82.

Fershtman, Chaim, and Ariel Pakes (2012), Dynamic Games with Asymmetric Information: A Framework for Empirical Work, Quarterly Journal of Economics , 127(4), 1611-1661.

Fudenberg, Drew, and David K. Levine (1998), The Theory of Learning in Games, MIT Press, Cambridge.

Gal-Or, Ester (1985), Information Sharing in Oligopoly, Econometrica , 53(2), 329-343.

Gal-Or, Ester (1986), Information Transmission - Cournot and Bertrand Equilibria, Review of Economic Studies , 53(1), 85-92.

Graham, Daniel A., and Robert C. Marshall (1987), Collusive Bidder Behavior at Single-Object SecondPrice and English Auctions, Journal of Political Economy , 95(6), 1217-39.

Groeger, Joachim (2014), A Study of Participation in Dynamic Auctions, International Economic Review , 55(4), 1129-1154.

Jeziorski, Przemyslaw, and Elena Krasnokutskaya (2016), Dynamic Auction Environment with Subcontracting, RAND Journal of Economics , 47(4), 751-791.

Jofre-Bonet, Mireia and Martin Pesendorfer (2003), Estimation of a Dynamic Auction Game, Econometrica , 71(5), 1443-1489.

Kirby, Alison (1988), Trade Associations as Information Exchange Mechanisms, RAND Journal of Economics 19(1), 138-146.

Kuhn, Kai-Uwe and Xavier Vives (1995), Information Exchanges Among Firms and their Impact on Competition , Office for Official Publications of the European Community, Luxemburg.

Luco, Fernando (2017), Who Benefits from Information Disclosure? The Case of Retail Gasoline, mimeo, Texas A&amp;M University.

Mailath, George J., and Peter Zemsky (1991), Collusion in Second Price Auctions with Heterogeneous Bidders, Games and Economic Behavior , 3(4), 467-86.

Maskin, Eric and John Riley (2000), Asymmetric Auctions, Review of Economic Studies 67, 413-438.

Maskin, Eric and Jean Tirole (2001), Markov Perfect Equilibrium: Observable Actions, Journal of Economic Theory , 100, 191-219.

Mermelstein, Ben, Volker Nocke, Mark Satterthwaite and Michael D. Whinston (2014), Internal vs. External Growth in Industries with Scale Economies: A Computational Model of Optimal Merger Policy, NBER Working Paper w20051, National Bureau of Economic Research.

McAfee, Preston and John McMillan (1992), Bidding Rings, American Economic Review , 82(3), 579-599.

Pesendorfer Martin (2000), A Study of Collusion in First-Price Auctions, Review of Economic Studies , 67(3), 381-411.

Porter, Robert and Douglas Zona (1993), Detection of Bid Rigging in Procurement Auctions, Journal of Political Economy , 101(3), 518-538.

Porter, Robert and Douglas Zona (1999), Ohio School Milk Markets: An Analysis of Bidding, RAND Journal of Economics , 30(2), 263-288

Robbins, Herbert, and Sutton Monro (1951), A Stochastic Approximation Technique, Annals of Mathematics and Statistics , 22, 400-07.

Romano, Joseph, Azeem Shaikh and Michael Wolf (2014), A Practical Two-Step Method for Testing Moment Inequalities, Econometrica , 82(5), 1979-2002

Saini, Viplav (2013), Endogenous Asymmetry in a Dynamic Procurement Auction, Rand Journal of Economics , 69(3), 25-41.

Shapiro, Carl (1986), Exchange of Information in Oligopoly, Review of Economic Studies , 53(3), 433-446.