## Dynamic Pricing and Learning with Long-term Reference Effects

Shipra Agrawal ∗

## Abstract

We consider a dynamic pricing problem where customer response to the current price is impacted by the customer price expectation, aka reference price. We study a simple and novel reference price mechanism where reference price is the average of the past prices offered by the seller. As opposed to the more commonly studied exponential smoothing mechanism, in our reference price mechanism the prices offered by seller have a longer term effect on the future customer expectations.

We show that under this mechanism, a markdown policy is near-optimal irrespective of the parameters of the model. This matches the common intuition that a seller may be better off by starting with a higher price and then decreasing it, as the customers feel like they are getting bargains on items that are ordinarily more expensive. For linear demand models, we also provide a detailed characterization of the near-optimal markdown policy along with an efficient way of computing it.

We then consider a more challenging dynamic pricing and learning problem, where the demand model parameters are apriori unknown, and the seller needs to learn them online from the customers' responses to the offered prices while simultaneously optimizing revenue. The objective is to minimize regret, i.e., the T -round revenue loss compared to a clairvoyant optimal policy. This task essentially amounts to learning a non-stationary optimal policy in a timevariant Markov Decision Process (MDP). For linear demand models, we provide an efficient learning algorithm with an optimal ˜ O ( √ T ) regret upper bound. 1

∗ Columbia University. Email: sa3305@columbia.edu .

† Columbia University. Email: wt2359@columbia.edu .

1 A one-page abstract has been accepted at the 25th ACM Conference on Economics and Computation (EC'24).

Wei Tang †

## 1 Introduction

In modern marketplaces, the demand for a product is influenced not only by the current selling price but also by customers' price expectation, aka reference price. Intuitively, the reference price is what customers perceive as the 'normal" price for a product based on the historic market prices. When the current price is lower than the reference price, customers are likely to perceive a bargain or 'gain", which could lead to an increased purchase. And when the selling price is significantly higher than the reference price, customers are likely to perceive a rip-off or 'loss", potentially leading to a negative effect on demand. The reference effects are often asymmetric, owing to customers' different attitudes towards a perceived loss v.s. a perceived gain (Lattin and Bucklin 1989, Rajendran and Tellis 1994).

In the existing literature, a common way to model the reference price mechanism in markets with repeated-purchases is to consider reference prices formed endogenously based on the past prices of a product (see, e.g., Fibich et al. 2003, Popescu and Wu 2007, Nasiry and Popescu 2011, Chen et al. 2017, den Boer and Keskin 2022). The motivation for this endogenous mechanism stems from the observation that the returning customers in these markets may remember the historical prices of the product.

A commonly used model for endogenously formed reference prices is the exponential smoothing mechanism (henceforth ESM ). In ESM , customers' reference prices are formed as a weighted average that puts exponentially decreasing weights over time on past prices (Mazumdar et al. 2005) in order to reflect the psychological intuition that customers have a fast-diminishing memory over the past prices. In doing so, ESM essentially captures a short-term reference effect dominated by the recent prices. The short-term nature of reference price effects in the ESM is also evident from the observation made by the previous works that in this setting, a fixed-price policy is near-optimal. More precisely, the revenue on repeatedly offering the optimal fixed-price is within a constant (independent of the sales horizon) of the optimal revenue assuming loss-averse customers 2 and linear base demand model (Fibich et al. 2003, Popescu and Wu 2007, den Boer and Keskin 2022). Intuitively, the result follows from the insight that due to exponential weighting in ESM , only a constant number of past prices significantly influence the current reference price and thereby the customer response.

However, the proliferation of online platforms and marketplaces has led to a paradigm shift in customer expectation and response behavior. The reference effects are becoming more pronounced and longer term, since the information about historic prices is widely and easily available. It is common for platforms like Google Hotel to put labels 'Great Deal, 28% less than usual' or 'Deal, 16% less than usual' to signal how good the current hotel price is compared to the average price of this hotel over the past year. 3 Certain products in Amazon have a 'Was Price' displayed, which is determined using recent price history of the product on Amazon. A number of third party websites, like CamelCamelCamel, PriSync, Wiser Solutions, Price2Spy, are dedicated to price tracking and monitoring, and are increasingly popular among customers for comparing product prices to historic prices before making a purchase. Thus the customers no longer need to rely on their individual memory in order to look back at historic prices over a week or a month or even year(s). The need for studying new reference price mechanisms with longer memory is therefore apparent, which was interestingly also mentioned as an important avenue for future research in a recent work on ESM (den Boer and Keskin 2022).

2 Formal definitions of 'loss-averse" vs 'gain-seeking" customers are provided in Section 2.

3 See Google hotel deals policy.

Motivated by these observations, in this work, we consider a novel and simple reference price mechanism aimed at capturing longer-term reference effects. In our mechanism that we refer to as averaging reference mechanism (henceforce ARM ), the reference price is formed as a simple average (i.e., with equal weights) of the past prices. See formal definition in Definition 2.1.

We demonstrate that this simple modification to the reference price mechanism significantly changes the nature of the dynamic pricing problem. Specifically, as opposed to ESM , the revenue of any fixedprice policy under ARM can be highly suboptimal compared to the optimal revenue: the difference can be as large as linear in sales horizon (see Proposition 1). Furthermore, we show that a markdown policy is always near-optimal under ARM . This matches the practical intuition that under significant and long-term reference effects, a seller may be able to set customer expectations better by starting with a higher price and then decreasing it in order to give a perception of a bargain to the customer (or, avoid the perception of a rip-off). This result also provides a technical justification of the widespread use of markdown pricing in industry (see, e.g., IBM 2023, Google 2023) and in academic literature (Lazear 1986, Wu et al. 2015, Jia et al. 2021, Birge et al. 2023).

Next, we summarize our contributions. Besides modeling contributions, our results include a detailed characterization of the optimal dynamic pricing policy under the ARM model, and algorithm design and analysis for learning and regret minimization when the demand model parameters are unknown.

## 1.1 Our Contributions and Results

A main modeling contribution of our work is to introduce and pioneer the study of the average reference model (i.e., ARM ) in dynamic pricing and learning literature which has thus far been dominated by the exponentially weighted average (i.e., ESM ) model for reference effects. We contend that ARM is able to capture longer-term reference effects on customer behavior and matches the practice much more closely in its implications on the seller's optimal pricing strategies. We provide several rigorous technical results to support this claim. Further, for ARM with linear demand models, we provide algorithmic and computational insights towards efficiently implementable near-optimal dynamic pricing strategies, in both full information and learning settings.

Our main technical contributions are summarized as follows.

Suboptimality of fixed-price and (near-)optimality of markdown pricing. In our first result that sets the ARM model distinctively apart from ESM , we demonstrate that under ARM , the revenue of any fixed-price policy can be highly sub-optimal compared to that of the optimal pricing policy. Specifically, in Proposition 1, we show that the difference between these two revenues can grow linearly in sales horizon T .

Next, in Theorem 1, we establish the (near-)optimality of markdown pricing in ARM models. Here, markdown pricing policy is defined as any policy under which the sequence of prices offered are always decreasing (or non-increasing) over time. In Theorem 1a , we show that under ARM with gain-seeking customers, markdown pricing is optimal. And further, in Theorem 1b , we show that for loss-averse customers, markdown pricing is near-optimal: specifically, there always exists a markdown policy that achieves a revenue within O (log( T )) of the optimal revenue in T rounds.

Our results complement the existing literature supporting markdown strategies in dynamic pricing problems (Lazear 1986, Wu et al. 2015, Jia et al. 2021, Birge et al. 2023, Jia et al. 2022), and provide a technical justification for the widespread practice of such strategies (Google 2023, IBM

2023).

## Characterization and implementation of near-optimal pricing for linear demand models.

Having established the near-optimality of the markdown pricing policy, we then explore how to characterize and compute such a desired markdown pricing policy. Notice that reference price effects make the dynamic pricing problem substantially more involved. In particular, as we elaborate later, with ARM , under any pricing policy, demand evolves as a non-stationary stochastic process. Finding an optimal pricing policy then amounts to solving a time-variant Markov decision process (MDP) with continuous action/state space. Moreover, the reward function (i.e., the seller's revenue function) is nonsmooth under asymmetric reference price effects. For a more tractable setting, we follow the existing literature (Fibich et al. 2003, den Boer and Keskin 2022, Ji et al. 2023) and consider a linear base demand model. Note that in this model, while the demand is linear in the offered price, there still may be some non-linearity in demand stemming from the asymmetry in reference price effects. For details, see Section 2.

Under the linear base demand model, firstly, in Proposition 2, we provide a detailed structural characterizations for a (nearly) optimal markdown pricing policy for both gain-seeking and lossaverse customers. Next, in Proposition 3, we utilize this markdown structure to provide an efficient algorithm for computing such policies.

Learning and regret minimization under linear demand models. We also study the dynamic pricing problem with reference effects in the presence of demand model uncertainty. This problem is motivated by the practical consideration that the demand model parameters may not be fully known to the seller a priori . Instead, the seller may have to learn these (implicitly or explicitly) from the observed customer response through sales data, in order to compute/improve the pricing policy.

Following the previous literature, again we consider a linear base demand model. We assume that seller has no apriori knowledge of the base demand model parameters or the parameters of the ARM model. In line with the online learning and multi-armed bandits literature, we focus on evaluating the algorithm performance via the cumulative regret , defined as the total expected revenue loss compared to a clairvoyant policy.

Our main contribution in this setting is a learning algorithm that achieves a regret upper bound of ˜ O (¯ p 3 . 5 √ T ) (see Theorem 2). A main insight that we derive and utilize for this algorithm is that although under ARM the near-optimal markdown policies are highly non-stationary, they can be parameterized by a two-dimensional parameter vector which also fully characterizes the local greedy price (namely the price that maximizes a single round revenue) under a certain range of reference prices. This insight allows us to offer estimated greedy prices in the learning phase and use the demand response from those prices to learn the desired markdown policy.

Finally, in Proposition 4, we derive a regret lower bound that closely matches our upper bound. In particular, building upon the proof of the existing lower bounds for dynamic pricing problem, we show that even when there are no reference effects, there exists a set of instances satisfying our assumptions such that any learning algorithm must incur a worst-case (over this set) expected regret of at least Ω(¯ p √ T ) .

## 1.2 Related Work

On the research related to dynamic pricing, our work mainly connects to two streams of works: (i) dynamic pricing with reference effects, (ii) dynamic pricing and learning (with reference effects). As we elaborate later, the seller's learning problem is essentially that of learning in a time-variant MDP. Thus, our work also relates to works on reinforcement learning for non-stationary MDP.

Dynamic pricing with reference effect. Dynamic pricing with reference effects has been extensively studied in both revenue management and marketing literature over the past decades (see Mazumdar et al. 2005 and Arslan and Kachani 2010 for a review). Below we mention mostly related works. As previously mentioned, ESM is one commonly-used model for reference price in the current literature. Under this mechanism, Popescu and Wu (2007) show that optimal pricing policy will eventually converge to a constant price when customers are loss averse, similar message is also delivered in den Boer and Keskin (2022). In addition, Chen et al. (2017) develop strongly polynomial time algorithms to compute the optimal prices under certain conditions. Other works that focus on ESM include Fibich et al. (2003), Wu et al. (2015), Hu et al. (2016), Cao et al. (2019), Golrezaei et al. (2020), just to name a few. As mentioned, two key features of ESM is that it is a stationary process and customers have diminishing memories over the past prices. These two features also appear in other reference price mechanism like peak-end anchoring (Nasiry and Popescu 2011). Our work differs with these works significantly as the considered ARM is a different reference price model that aims to capture the long-term reference effects.

Dynamic pricing and learning (without and with reference effect). The learning part of our work also relates to the expanding literature on learning and earning in dynamic pricing problems, see, e.g., Kleinberg and Leighton (2003), Besbes and Zeevi (2009), Araman and Caldentey (2009), Broder and Rusmevichientong (2012), Keskin and Zeevi (2014), Besbes and Zeevi (2015), Agrawal et al. (2023), Chen et al. (2022, 2024), just to name a few. We refer the interested readers to the survey by Den Boer (2015a). These works are focusing on learning without reference effects. The mostly related works are den Boer and Keskin (2022), Ji et al. (2023), similar to our work, they also consider learning with reference effects. In particular, den Boer and Keskin (2022) study dynamic pricing with learning with linear demand model under ESM , and Ji et al. (2023) focus on multi-product selling setting where the reference effects are formed by the comparison between the prices of all products. Notice that in these two works, a fixed-price policy is already near-optimal. Thus, the high-level idea behind the algorithms in both works is an epoch-based learning-and-earning algorithm that integrates least squares estimation to estimate the underlying demand parameters and compute a good estimate of the optimal fixed price. Our work differs from these two works significantly as in our problem, the seller needs to learn an optimal non-stationary policy, making their algorithms not applicable to our setting.

Learning in non-stationary MDP. The underlying demand function with our reference price mechanism essentially evolves according to a time-variant MDP. There are also works studying dynamic pricing and learning in a non-stationary environment, e.g., Besbes and Zeevi (2011), Den Boer (2015b), Chen et al. (2019), Keskin and Zeevi (2017), Agrawal et al. (2021). However, these works are either consider a very broad non-stationarity or focusing a particular structured model which is fundamentally different from ours. There is also much recent work on learning and regret minimization in stateful models using general MDP and reinforcement learning frameworks (e.g., Auer et al. 2008, Agrawal and Jia 2017, Cheung et al. 2020, Fei et al. 2020). However, these results typically rely on an assumption that each state can be visited many times over the learning process. This is

ensured either because of an episodic MDP setting (e.g., Fei et al. 2020), or through an assumption of communicating or weakly communicating MDP with bounded diameter, i.e., a bound on the number of steps to visit any state from any starting state under the optimal policy (e.g., Cheung et al. 2020, Agrawal and Jia 2017). Our setting is non-episodic, and under ARM , the state (i.e., the reference price) can take steps that linear in current time to reach other state. Therefore, the results in the above papers on learning general MDPs are not applicable to our learning problem.

## 2 Problem Formulation

We consider a dynamic pricing problem with an unlimited supply of durable products of a single type to be sold in a market across T time periods. At the beginning of each time round t ∈ [ T ] , the seller sets a price p t ∈ [ p, ¯ p ] for her product, where p and ¯ p are the pre-determined smallest and largest feasible prices satisfying 0 ≤ p &lt; ¯ p &lt; ∞ . Given the price p t , the seller observes the demand D t and collects revenue p t D t . The observed demand D t is influenced by: (i) the current selling price p t , (ii) a reference price r t , and (iii) the unobservable demand shocks in the following manner:

<!-- formula-not-decoded -->

Here, r t ∈ [ p, ¯ p ] denotes the reference price at the beginning of round t ; ε t ≤ ¯ ε, t ∈ [ T ] denote unobservable demand shocks which are independently and identically distributed random variables with zero mean, and are all upper bounded by ¯ ε ≥ 0 . And, the demand function D ( · , · ) is given by 4

<!-- formula-not-decoded -->

where H ( p ) represents the non-decreasing base demand in price with the absence of reference effects, parameters η + ≥ 0 and η -≥ 0 control the impact of reference price on demand: if the current selling price p is less than the reference price r then the expected demand increases by η + ( r -p ) , but if p exceeds r then the expected demand decreases by η -( p -r ) . The aggregate-level demands are classified as loss averse, loss/gain neutral, or gain seeking depending on whether η + &lt; η -, η + = η -, or η + &gt; η -, respectively. For presentation simplicity, we assume p ≡ 0 . Motivated by the pricing application that we consider, we also assume the non-negative demand: 5

Assumption 2.1. To avoid negative demand, we assume that D ( p, r ) ≥ 0 for all p, r ∈ [0 , ¯ p ] .

The reference-price dynamics. In this paper, we consider the following dynamics of evolution of the reference-price ( r t ) t ∈ [ T ] .

Definition 2.1 ( ARM ) . A reference-price dynamic is said to be an averaging reference mechanism ( ARM ) if

<!-- formula-not-decoded -->

where r 1 is the starting reference price at t = 1 .

4 Here operation ( x ) + denotes max { x, 0 } .

5 The assumption on demand non-negativity is commonly made in dynamic pricing literature, see, e.g., Besbes and Zeevi (2015), den Boer and Zwart (2014), den Boer and Keskin (2022), Chen et al. (2022, 2024).

Another equivalent way to describe ARM is through the running average: given reference price r t and selling price p t at time t , the next reference price r t +1 is given by:

<!-- formula-not-decoded -->

where 1 -ζ t = 1 / ( t +1) for all t ∈ [ T -1] can be seen as the averaging factor that captures how the current price p t affects the reference price in next round. Intuitively, in this mechanism, once a product has been around for a long time, the customer price expectations may not be impacted much by a few rounds of price variations.

Remark 1. We emphasize that our ARM is a non-stationary mechanism as the averaging factor ζ t is time-dependent. This stands contrast to ESM where r t +1 = ζr t +(1 -ζ ) p t where the averaging factor ζ is assumed to be a constant across the whole time horizon.

Given a price p ∈ [0 , ¯ p ] , a reference price r ∈ [0 , ¯ p ] , and a demand function D ( p, r ) defined in (1), we denote the expected single-round revenue by function Rev ( p, r ) :

<!-- formula-not-decoded -->

Given a starting reference price r at time t , and the sales horizon T , the optimal expected cumulative revenue starting from time t is given by

<!-- formula-not-decoded -->

We use V p ( r, t ) to denote the seller's revenue on applying a price sequence p = ( p s ) s ≥ t , starting with a reference price r at time t . For notation simplicity, when starting time t = 1 , we use the notation V ∗ ( r ) ≡ V ∗ ( r, 1) , V p ( r ) ≡ V p ( r, 1) . We use p ∗ ( r, t ) /defines ( p ∗ s ) s ≥ t to denote an optimal pricing policy that maximizes V p ( r, t ) .

Dynamic pricing and learning under unknown model parameters. We also consider a partial-information setting where demand function is initially unknown to the seller and has to be inferred from the observed demand response to offered prices. In this part, to make the problem more tractable, we focus on linear base demand models, i.e., H ( p ) /defines b -ap . Here a ∈ R and b ∈ R are the base demand model parameters which captures the customers' price-sensitivity and the market size, respectively.

Then, in the learning setting, we assume that the base demand parameters ( a, b ) and the reference effect parameters ( η + , η -) , are all apriori unknown to the seller. At any time t , given reference price r t , on offering a price p t , the seller can observe a noisy outcome from demand response function D ( p t , r t ) = b -ap t + η + ( r t -p t ) + -η -( p t -r t ) + , which may be used to infer the demand model and thereby the optimal pricing policy. To aid this inference, we make the following assumption on the base demand model H ( p ) ; similar assumptions are commonly made in the existing literature on dynamic pricing with linear demand models (see, e.g., Keskin and Zeevi 2014, den Boer and Keskin 2022, Ji et al. 2023, den Boer and Zwart 2014).

Assumption 2.2. Assume that the maximizer of the function pH ( p ) with linear demand model H ( p ) = b -ap , lies in the interior of the feasible price range [0 , ¯ p ] . In particular, assume b 2 a ≤ ¯ p -Ω(1) . 6

6 We would like to note that this assumption is not contradicting to Assumption 2.1. In fact, there exists a rich range of model parameters that satisfy both assumptions, e.g., under linear demand, any b ∈ [( a + η -)¯ p, 2 a ¯ p ) with η -&lt; a would satisfy these commonly-made assumptions.

We use vector I = ( a, b, η + , η -) to denote the parameters of a given problem instance. We assume that all parameters are normalized so that I ∈ [0 , 1] 4 . We use I ⊆ [0 , 1] 4 to denote the parameter set of all feasible instances satisfying Assumption 2.2.

An online dynamic pricing algorithm π ALG maps all the historical information collected up to this time round into a price in [0 , ¯ p ] . Given a starting reference price r 1 ∈ [0 , ¯ p ] , we measure the performance of a pricing algorithm π ALG via the T -period cumulative regret, or regret, which equals to the total expected revenue loss incurred by implementing the algorithm π ALG instead of a clairvoyant optimal pricing policy for the problem instance with parameter I , i.e.,

<!-- formula-not-decoded -->

where the expectation E p ∼ π ALG [ · ] is over the sequence of prices induced by the algorithm π ALG when run starting from reference price r 1 on a problem instance with parameter I .

## 3 Characterizing (Near-)Optimal Pricing Policy

In this section, we derive our main results on the structure and computation of optimal and nearoptimal dynamic pricing policies under ARM .

First in Section 3.1, we show that a fixed (aka static) price policy, which is known to be near-optimal under ESM , can indeed be highly suboptimal in ARM in the sense that it can suffer with a linear revenue loss compared to the optimal policy.

Next, in Section 3.2, we demonstrate that a markdown policy is optimal for ARM when customers are gain-seeking, and near-optimal (within O (log( T )) revenue) when customers are loss-averse.

Finally, in Section 3.3, we provide a detailed characterization and computational insights for this markdown policy for the case when the base demand model H ( p ) is linear. This characterization will be later utilized in Section 4 for learning and regret minimization under demand uncertainty.

## 3.1 The Sub-Optimality of Fixed Price

We refer to p as a fixed price policy if p = ( p, . . . , p ) for some p ∈ [0 , ¯ p ] . We show that under ARM , any fixed-price policy can have linear (in number of rounds T ) revenue loss compared to an optimal pricing policy, even if we restrict to the linear base demand model and loss-neutral customers.

Proposition 1. There exists an ARM problem instance with linear base demand model, i.e., H ( p ) = b -ap and loss-neutral customers (i.e., η + = η -), and an initial reference price r 1 such that for any fixed-price policy p , we have V ∗ ( r 1 ) -V p ( r 1 ) = Ω( T ) .

The above results highlight a fundamental difference between our ARM model and the previously well-studied ESM model for reference effects. In particular, for the base linear demand, when the reference-price dynamics follow ESM and when the customers are loss-neutral, previous work (den Boer and Keskin 2022) have shown that the seller can safely ignore the reference effect (notice that when the reference effects are ignored, the single-round expected revenue would equal p ( b -ap ) and thus the optimal fixed-price would be b / 2 a ). They show that a fixed-price policy with the selling price at b / 2 a yields a revenue very close to the revenue under the optimal pricing policy (whenever η + ≤ η -): the difference between these two revenues is bounded by a constant independent of the sales horizon T . However, the above Proposition 1 shows that the performance of such fixedprice policy can be arbitrarily bad under ARM . One intuition behind this performance dichotomy

(compared to the optimal total revenue) of fixed-price policy between ESM and ARM is as follows: Recall that the ESM model consider constant averaging factor ζ ≡ ζ t for all t , in doing so, the customers essentially put exponentially decreasing weights on the past prices to form the reference price. This is similar to a setting where the reference price at time t equals to the average of a constant window size (independent of sales horizon T ) of past prices. While under ESM , this window size scales linearly in time t .

Remark 2. We would like to note that a fixed-price policy has the (nearly) same revenue under both ESM and our ARM . Together with the results in Proposition 1, this demonstrates that under ARM , the optimal revenue can be significantly higher than the optimal revenue under ESM .

We prove Proposition 1 by showing that when customers are loss-neutral, there exists a simple two-price policy with Ω( T ) -larger revenue compared to the revenue obtained under any fixed-price policy. The policy starts with the higher price, and then at some round, switches to the lower price which is offered for the remaining rounds. 7

## 3.2 The (Near) Optimality of Markdown Pricing

In this section, we show that for any ARM problem instance I , there always exists a markdown pricing policy that is near-optimal. We first define the markdown pricing policy.

Definition 3.1 (Markdown pricing policy) . A markdown pricing policy is defined as any pricing policy which, when applied starting at any t 1 ∈ [ T ] and reference price r , generates a price curve ( p t ) t ≥ t 1 satisfying p t ≥ p t +1 for all t ≥ t 1 .

The main results of this subsection are summarized as follows:

Theorem 1 (Near optimality of markdown pricing policy) . Fix any starting reference price r 1 ∈ [0 , ¯ p ] at time t = 1 ,

- 1a when η + ≥ η -, i.e., when customers are gain-seeking, optimal pricing policy is a markdown policy;
- 1b when η + &lt; η -, i.e., when customers are loss-averse, there exists a markdown policy p that is near-optimal, namely, V ∗ ( r 1 ) -V p ( r 1 ) = O (¯ p (¯ p -r 1 )( η -+ η + ) ln T ) .

From the above results, we can see that the optimality of the markdown pricing policy depends on the relative values of reference effects η + and η -. When the customers are gain-seeking (i.e., η + ≥ η -), the optimal pricing policy is a markdown policy. This characterization shares certain similarity with previous results on the optimal pricing policy when the reference-price dynamics follow the ESM . In particular, Hu et al. (2016) have shown that when the customers are insensitive to the loss (i.e., η + &gt; 0 , η -= 0 ) and they only remember the most recent price (i.e., the averaging factor ζ in ESM equals to 0 ), the optimal pricing policy is a cyclic markdown pricing policy . While in our setting with ARM , no matter how customers are sensitive to the losses, as long as customers value more on the gains, the optimal pricing policy is always a markdown policy.

On the other hand, when customers are loss-averse (i.e., η + &lt; η -), the markdown pricing policy may not be optimal. However, it is guaranteed that there exists a markdown pricing policy that is near-optimal, namely, its total revenue is within O (ln T ) of the optimal revenue.

7 We construct an instance with initial reference price r 1 = 0 to prove Proposition 1. However, we note that the proof can be easily extended to consider arbitrary r 1 .

To prove Theorem 1, we first show the optimality of markdown pricing policy when the customers are gain-seeking. This is summarized in the following lemma.

Lemma 3.1. Fix any starting time t 1 ∈ [ T ] and a starting reference price r t 1 = r , when η + ≥ η -, the optimal pricing policy starting from time t 1 is a markdown pricing policy.

The main intuition behind the above markdown optimality is that: under ARM , the high price and low price contribute the same to the reference prices in later rounds. Since customers are gainseeking, i.e., the impact of a perceived gain (how low is price compared to reference price) is more than the impact of the same amount of perceived loss, it is always better for the seller to charge the high price before the low price. Indeed, we prove the above results by showing that whenever under a pricing policy p = ( p t ) t ∈ [ t 1 ,T ] there is an increase in price, i.e., p k &lt; p k +1 for some time rounds k and k +1 , then we can obtain a new pricing policy p ′ with a higher payoff by simply switching the prices p k and p k +1 at time round k and k +1 .

To analyze performance of the markdown pricing policy when the customers are loss-averse, we first show that when the starting reference price is the highest possible price ¯ p , then even in this case, the optimal policy is a markdown pricing policy. Intuitively, in this case, the markdown policy will never offer a price p t that is above the current reference price r t , so there is never a perceived loss.

Lemma 3.2. Fix any starting time t 1 ∈ [ T ] and a starting reference price r t 1 = ¯ p , if η + &lt; η -, then the optimal pricing policy starting from time t 1 is a markdown pricing policy.

We then show the following Lipschitz property on how the optimal revenue V ∗ ( r, t 1 ) depends on the starting reference price r . Notably, this property holds for any reference effects.

Lemma 3.3 (Optimal revenue gap w.r.t. different starting reference price) . Fix any starting time t 1 ∈ [ T ] , the optimal revenue function V ∗ ( r, t 1 ) is increasing w.r.t. reference price r . Moreover, for any ( η + , η -) , we have V ∗ ( r ′ , t 1 ) -V ∗ ( r, t 1 ) ≤ O (¯ pt 1 ( r ′ -r )( η -+ η + ) ln T / t 1 ) for any r ′ ≥ r .

The near optimality of markdown pricing for loss-averse customers immediately follows by combining Lemma 3.2 and Lemma 3.3. Namely, fix an arbitrary starting reference price, one can simply implement the price sequence from optimal pricing policy (which is a markdown pricing policy) for starting reference price ¯ p , then one can guarantee near-optimal revenue. The missing proofs for the above three lemmas and the detailed proof of Theorem 1 are deferred to Appendix A.2.

## 3.3 Characterizing Near-optimal Markdown Price Curve

In preceding discussions, we show the near optimality of markdown pricing policy. In this section, focusing on base linear demand, namely, H ( p ) = b -ap , we provide detailed structural characterizations for such near-optimal markdown policy. We also provide computational results for computing such policy.

The structure of the near-optimal markdown price curve. When the base demand is H ( p ) = b -ap , we can characterize the following structure of a (near-)optimal markdown policy:

Proposition 2. Given a problem instance I = ( a, b, η + , η -) with linear base demand model, and a

starting reference price r , we define price curve ˜ p ( r ) /defines ( p t ) t ∈ [ T ] as:

where p † and t † are some deterministic functions of ( a, b, η + , r ) , and r t = r + ∑ t -1 s =1 p s t , t ∈ [ T ] . Then the price curve ˜ p ( r ) is optimal when η + = η -, i.e., V ∗ ( r ) = V ˜ p ( r ) ( r ) .

<!-- formula-not-decoded -->

Furthermore, the price curve ˜ p (¯ p ) (i.e., the above price curve computed with r = ¯ p ) is near optimal when η + = η -, namely, for any starting reference price r , V ∗ ( r ) -V ˜ p (¯ p ) ( r ) ≤ O (ln T ) .

/negationslash

We can see that price curve ˜ p ( r ) keeps charging the price as the highest possible price ¯ p until the time round t † -1 , and then at t † it strictly markdowns its price to p † . Given the price p t at time round t ≥ t † , the markdown amount for the next price p t +1 depends on the previous reference price r t , and the current time round t +1 , and also the parameters a, b, η + . In particular, the parameters a, η + control the degree of the markdown amount. The time round t † and the price p † can be determined by the parameters ( a, b, η + ) , and the starting reference price r .

We below give a graph illustration in Figure 1 about the shape of the price curve ˜ p . In the figure, we show that the time round t † could be strictly larger than first time round (see Figure 1a), or could equal to first time round t = 1 , which in this case, the price curve then becomes a strict markdown price curve (see Figure 1b) throughout the sales horizon.

Figure 1: An illustration of the price curve defined in Proposition 2.

![Image](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAArgAAAElCAIAAACAsgZ4AABdkElEQVR4nO3dZVxU2/s//D3FEEM3CAqCtCgSBh7FJlSwEEwsDOzWI3ZioXIoURRsBVERDExAUESlREK6u2eYul+H/b/3d344egxgA3O9H83smPlMX7P2WmsTuFwuAgAAAADAD5HvUgAAAAAABEHIv7dbbm7uu3fvWltbORxOR0cCQNARCAQqlTp8+HBVVVW8s3RTubm5CQkJdDod2kQB6HAEAkFUVHTKlClkMvn3C4XIyMg1a9bMnj1bQUEBagUAOhaLxTp37tyVK1ecnJzwztJNRUZGrlixws3NTUhICO8sAPQqBAKhtbX1/v37VlZW0tLSv18oIAiir69//PhxBQWFDk0IAPhXcnIy/Ff+AQKBoKamdurUKfQfDwCgYxUVFbFYrA7oowBtCQB0Bg6HQyRC/6H/QCAQ2Gw23ikA6IXYbDbvH5U/+jKCfzwAdAb4ZP0keKIA6IJPFvxrAQAAAMB3QaEAAAAAgO+CQgEAAAAA3wWFAgAAAAC+CwoFAAAAAHwXFAoAAAAA+C4oFAAAAADwXVAoAAAAAOC7oFAAAAAAwHdBoQAAAACA74JCAQAAAADfBYUCAAAAAL4LCgUAAAAAfBcUCgAAAAD4LigUAAAAAPBdUCgAAAAA4LugUAAAAADAd0GhAAAAAIDvgkIBAAAAAN8FhQIAAAAAvgsKBQAAAAB8FxQKAAAAAPgu8vdXAQAADths9o0bN5KTk0kkErqEy+XSaLQlS5bIy8vjnQ4AgQOFAgCgeyGRSJMnTx4/fjyCIAQCAS0UiESilJQU3tEAEER8CgUul9vQ0MBkMtGPKLqETCaLi4sTiXCoAgDQuZhMpoiIiLi4OIIgra2tXC6XSqXiHQoAwcWnUGAymdeuXcvJycHa/dhstrKy8oIFC6CiBwB0qsrKyocPHxYUFEydOhVBkOTk5MrKyr59+9rY2JDJ0AIKAA74fPCEhIRcXV3xCAMAEHRRUVFDhgzJzs5eunTpmjVrrK2tW1paXFxcREVFx40bh3c6AAQRn0MJGRkZX79+RY84pKWlpaSkcDgcPLIBAARLRUVFY2OjlpZWSUmJnJycvb29lJSUkpKSsLBwREQE3ukAEFDtWxTevn2bkZGRmJg4fPhwBEEkJCRycnIePHiwfv16OEwIAOhUoqKi1tbWDQ0Nnz9/njZtmoiICIIgdDq9vLxcRUWl3cYEAkFYWBinpAD0ZmQyGet70L5QaG5uTk5Otre3f/369eHDhwMDA42MjMrKymbNmmVgYDB58mRsS8L/r2vDAwB6M7E2b9++LSsrs7CwQBcWFxfn5uY6ODjwbkkkEhsaGi5cuCAkJIT2tp4+fbqQkBC6tqysLCoqis1mo1f19PRMTU2xfd+8eZOVlYVeplAoY8eOxUZdNjc33717F92Ry+WqqalZWVlhO378+DE5ORm7Onr0aDU1NWzHqKio2tpa9Kq0tLSNjQ3W+/vr16/R0dHYCI5hw4Zpa2ujqzgczuPHjysqKtCrEhIS48aNExMTwx77s2fPuFwuelVfX3/IkCFYgJcvX+bl5aE3SyQSp0yZgvYARRCkqqoqKiqKwWCgVzU0NCwtLbEdExMTU1NT0csEAmHs2LHKysroVRaLdfv2bSaTiV5VVFQcN24c9kDS0tLev3+P3c6wYcO0tLTQy62trVFRUZWVldhLOXXqVOzHpqCg4MWLF9iOgwcPNjQ0RC9zudwXL14UFhaiV0VFRceOHYv1h6uqqoqMjMRatfv374/+if32pSQSiRMnTpSTk0Ov1tXVRUVFNTU1oVdVVFTGjBmD/WalpqYmJibyfSnZbPa9e/caGxvRq3JycmPGjMH+J3/9+jUmJgbb0cTExMDAAHspo6KiSktL0atkMnnGjBkUCgW9WlpaGhUVhT2Qdi9lTEwM2pCPHv0fO3Ys9kAaGxvv3buHvSfV1dVHjx79vZfSyspKVVUVvdrU1BQVFVVXV4c9kIkTJ2IvZVZWVmxsLPaeHDFiRP/+/bEH8uDBg/z8/P/1CuLyyM/Pv3XrFp1OHzVqlLu7O7qwpqZm5MiRu3fv5t3S09NTX1+/qKiIdyGTyczMzKyuruZdyOVyi4qKcnNz2y1sbGzMyMhobm5ut/zr16/FxcXtFlZWVmZmZrJYrHZ3l5GRUVVV9e3d5eTk8L27pqamdstzcnLaPQoAuoOxY8deuXKFK5DOnz8/aNCgiooK9GpwcLCKikpcXFy7bbDvSvRnifeb5/nz57yNDWvXruXdd+nSpdgqCQmJmJgYbFVhYaGoqCj2tTt58mTeHXfu3IntSCQSw8LCeHc0MjLC1g4ePJhOp2NrL1++jP1EEQgEPz8/bFVra+uwYcOwHXV1dXm/vh49esTbhXPjxo28eaZNm4bdrLCwcGZmJrbq7du3srKy2I5z587lcDjY2q1bt2KrSCTS48ePsVWNjY2SkpLY2jFjxjAYDGztyZMneZ/2CxcuYKtqamqGDh2Krerbty/v13tYWBjvjocPH8ZWsVgsOzs7bJWamlpKSgq2NiEhAav/EARZuHAh7zOwePFibJWQkFB8fDy2Ki0trW/fvthaa2tr3l+QY8eO8eYJDQ3FVjU3N2tqamKrLCwseN9aly5d4t3Rw8MDW8VkMnm70dBotIaGBmzt06dPsaIBQZBNmzbxPpAFCxZgq6SlpXnf7Tk5OWjTGvrmsbe3591xy5YtvC/lw4cPsVV5eXn6+vrYWnNzc96XMiAggPc9efHiRWwVg8EwMTGhUqlYfgJWq6J1BJfLzczMnDp16pkzZyZOnIgWUJMmTXJxcdm+fTu62fv3752dnbOysubMmWNnZ8dmswkEApFIvHfv3p07d3R0dFavXk2j0TgcDolEKigoOHr0aGtr68qVKwcPHsxisQgEAofD8fX1ffv2raWl5aJFiwiEf2NQKJTY2Fh/f38pKak1a9b069cP3bipqenIkSPFxcX29vYODg5oqUuhUG7dunX//v2+fftu2bJFREQEvYWsrKwzZ840Nja6urpaWFiggzw5HI6fn198fPzQoUOXLVuG3h2ZTH737p2vr6+wsPDZs2etrKw4HI6IiIiEhAT2hDQ2NjY1NaHbEwgEKSkp7C2Lli8cDgddS6VSeYeEtLS01NfXYy+DhIQE9rXF5XJra2tbW1vRHUkkkqysLLYlg8Gora3FqjxxcXHsawutkel0OrajlJQU9iXCYrHQdzO6tt0DaWhoaG5uxh6ItLQ09pZls9m1tbXoU/3tA2lubm5oaMDySElJYZU1l8utrq7GdhQSEpKSksIeCJ1Or6+vx7Zs90Bqa2sZDAa6I5FIlJOTw3ZsbW2tra3FHoiYmBiNRsN2rK+vb2lpwXaUkpLifSDV1dXoK4K+lLzfd9hLiV7lfSk5HE5tbS36VkHfRTIyMtiOdDq9trYWLcM5HI6UlBTvL1BNTQ32UlIoFCkpKaxgRx8I9gy0eyB1dXUtLS1EIhF9IDIyMth/LxaLVVVVNW3atFWrVjk7OyOCZ/HixXV1dbdv30af/9mzZ6urq588eZL3J9PX1/fgwYMfPnxA35AEAkFUVBR7fdlsNvo7jV6lUCi8R04ZDAb2jxk9fsE7sxP684ZeJZPJvC93axvsqrCwMBaJy+W2tLRg/xeJRCLvG57FYtHpdOwqlUrl/c1oaWnBGj+IRKKIiAjvA2lpacG2FGqDXaXT6SwWC3sgIiIi2NuPw+G0tLT85AMRERH53jNAIpGwXyl0TBzWSvHtM0Cn03kfyA+egR88ECKRKCwszPtAmpubsS1/8FKirRG8O9LpdOwVafcM/OCBoN972I4kEklYWBh7RX7+pfz2PfmDl7Kj3pPtXkre92S7l7LdA2n3DDQ0NMydO/f8+fP/r7GN+42rV68OHDgQq2ojIiJkZWUfPHiAbeDm5ob0LjQazczMbODAgdu2beOtuz09PY2MjIyNjQcOHDho0KDY2FjeqtPS0nLgwIHGxsZGRkZLlizhLdaCg4PRVaiQkBBsVUNDw7x587CbnTBhAu8/j+fPn2M7GhkZnT9/HlvFZrPXrFmDrkV35P0DkZaWZmpqiuX5+++/eV/TvXv3Yvdoamr66dMnbBVaCGI3u2LFCjabja1FDz9heaKiorBVdDrd1tYWu0dnZ2f0Bx51//597OEPHDiQt1xlsVjLli3D8gwdOpS3pScuLm7IkCHYzR4/fpz3gezatQvb0dLSMikpCVtVVFRkbm6O7djuT+Tp06exJ3bw4MGvXr3CVpWUlDg4OGDPgJOTEzp2HxUaGmpoaIjuaGhoyPvPo7W11dnZGdvRwcGhpKQEW/vq1avBgwdjz8CpU6d486xduxZ7IObm5rzNWsnJyZaWliIiIjdv3uQKnvr6ejMzs4kTJ5aXlzc1NXl5eTk6OpaWlrbbzMfHR11dnfdDBwDoKGw2e9q0aeXl5ehVPsMj4+LitLW1sa5Djx8/1tLSGjFiBLZBv379iEQih8MRExPT1tZGKyYOh/Plyxe0QtHS0pKQkOBwOEQisaysrKioCEEQGRmZfv36oX/4GAzGly9f0MrL0NCQQqGg/yDz8/OrqqoQBOnTp4+CggK6cWNjY2ZmJlor6ejofHt3AwYMEBMTQ/+ZYXcnJyenpqaG3iyTyUxJSUFLKl1dXfSgJpFI/Pr1K/qfj06nKygomJmZGRoa8na80NfXnzx5MplMRrdXUFDAVpFIJBsbm+bmZiKRyGKxtLW1eWej0tTUnDJlCnpTBAKBtwWMQqH89ddf6urqJBKJw+FIS0vz9hlRVlaePHkykUhEYw8YMABbRSAQLCwsxMXF0R2lpKR42wykpKQmT56M/r9nsVgmJia8r+mgQYMYDAb6QMhksrS0NG+RNGHChOrqaiKRyGazsWcYpa2tbWdnh5bMLBYLO5CJPgPjxo2rqqpCd1RXV+etrNXU1KZOnYo+/xwOBzsiiz6Q4cOHy8nJoXmoVCrvMyAvLz9lyhTsgWAHMlEmJiZoSxWHwxEVFeV9IKKionZ2dmjDAJPJ5G0Hxl5KEomEvpRKSkrYKhERkTFjxujr66M3q6amxvtSqqurT548GX1oTCZTXV0dW0UkEkeNGqWpqYk+A0pKSrwFu5KS0pQpU9D3MJvNxg5koiwsLERFRdF3PoVC4f3vJSUlZW1t/fXrV8EcbZSZmVlfX7906dInT540NjZyOBwvLy/eVnRegvkUAdDZ0D/M2NX/c+gBbXCYMmWKvr6+l5cXgiBfvnyZM2fOtm3bZsyYgW1TV1e3atWq8PDwM2fOjB8/Hvvljo2NvXPnjqGh4axZs7BCobS0NCAgoLGxcdGiRdra2lih8PDhwxcvXowZM8bGxgYrFD5//hwcHCwpKblgwQIlJSWsUAgODs7MzJwxY8bQoUOxu4uJiQkNDdXT03N2dsYKheLi4sDAwObm5rlz5+ro6GCFwv3791++fDl27NhJkyZhhUJGRoavr29ERERVVZW+vn5oaCjvDzMAOJowYcLChQsF8NCDn5/f+fPnHzx4ICUlxWAwsN557fj6+h46dOjLly8w8AGADsdisWbNmuXr68v/0ENSUpKWltbSpUvftVm5cmVAQMC37RJ+fn7GxsZYbyMMb7v9f/rzjTvk7mJjY/v06YMgyMCBA3kb8wHAC4vFGjNmjGB2Zly8ePGCBQv+czNfX191dXX0MDwAoGMxmUwHBwfs0EP7CZdSU1OFhYWXL19eU1NTUFCwZMmSRYsWfVtuoEcNeHuRoH5pwOSfb9whdzds2DA/Pz95efmkpKQlS5ZUV1f//G0CADoKnU6PiIiIiYlhs9np6ek/3hj9/uI9aAUA6CTtC4W4uDgtLS0TE5Px48c7ODgMHjwYEQDW1tZnz54VFRV9+fLlsmXLGhoa8E4EgMBpbGysrq7evHnzqFGj8vLyvu1/gI4Y4nA4eXl5ly5dKi4uDgoKwjqZAwA6yf/pzNjY2BgfH29jY4MIHkdHx8LCws2bN9+5c0dRUfHo0aO8g9kAAJ1NTk5uzpw5P9igoaFh7969y5cv9/T0fPPmDYIg27dvRwf7dGFMAAS4RaG4uNjb27uxsbGwsPD169f/vWfbaFekF9m4ceP+/fsJBMI///yjra198uRJ+LMCQPdBJpPRIbjYksbGRmj/A6DrWhRERUUnTJhgZ2fHYDB4Z6rBtLa2pqSk6OjoiImJoVMbTZs2TVhYmMvlysnJHThwoF+/fuiW2dnZ69atw3owjBs3btOmTdjteHp6Yud3IZPJHh4eenp66NWCgoK///67rKwMvTpkyJB9+/ZhhyHv3Lnj7++P3c6uXbuwQZutra3Lly8vLi5Gr+ro6Ozfvx8bPfjs2TN0/iz06qJFi2bNmoVe5nA4f//9NzqXJzrmc9CgQR8+fCgtLd2zZ4+dnR2MgwCgO6DT6QwGQ09Pj0gkrlq1KjExMT4+nslk5uTk8A7eBgB0YouClJSUsbGxnp7eoEGDNDQ0vt20qqpq7969OTk5vAu/d8YH7GQQP1772zt+u/bHO/7MvhwOR1xcfMyYMejC5ubmDx8+8L0pAEAXCwoKWr16dXZ29okTJ4KDg52cnND+1B4eHtgcoACATvHz4yUYDMb69evREzF4e3sPHjy4pqaG2+sUFhZiM5YPGTIkLy8P70RA4Ajy8Mjvqaqqys3N3bt3b2JiYllZmb+/v6ioKDrp7NmzZ/FOB4AgDY/8noSEhIiICBqN9vr167i4OPTsGrwzV/caqqqqu3fvtre3l5eXf//+/datW78dBQoA6GIyMjKysrIpKSmtra0KCgpcLldeXh7t/Hj27NmCggK8AwLQa/1soZCcnPzmzRsmk/mpTbv5HHuZSZMmhYaG7t69G0GQ69eve3t7450IAEHB+/ejqakJO1cQeuIcehu0dxGFQlm5cqWysnJGRka7c/oBAHAoFFxcXLZt20an05ctW+bq6opOuoz0aitWrJg/fz6CIPv27Xv27BnecQDo/ZKSkgICAs6dO1dcXPz8+fOQkJBz5859+vQJXUskEoWEhLBz3LW2tpqZmaHn5z1z5kxubi6u2QFABL1QQKv79PT0uro6RDAQicTDhw8PGjSoqqpqw4YN5eXleCcCoDerr69/+/btlClT7ty5s3XrViEhIScnp759++7atQs9VxyZTBYREeE99xibzXZzc1NSUqqoqPD09MQ1PgCCUShwudyUlBT0aB+Lxfr06dPnz5/b7dDrGxJ4qaioeHl5SUlJffr0afPmzbxn7wYAdKwvX74oKirSaLTy8nJTU9MRI0aQyWRdXd3MzEx0ADOFQpGUlOQ90yabzVZVVV23bh2CIFevXn3//j2ujwAAASgUXr16lZiY6OHhER4efufOnYqKisjISC8vL3TeISKR2O6MwL1VbGzsyZMna2pqEAQZPny4u7s72lkhODgYTmsLQCfp16/fyJEjk5KSOBzOqFGj0IU1NTV1dXVojU4mk+Xk5HgLBfR/y7x58/T09MrLy6E7EQCdWyiUl5dnZ2dPnTo1Nzf39OnTQ4YMGTdu3JQpUy5fvhwfH4+W8+Li4kJCQkhvFx8f7+XlhRYKCIKsWrXK0dERndNp1qxZ2LROAIAOJC8vLyUlFRsbq6SkhM3elpaWxmQy1dTUEAQhkUiWlpYyMjLtdlRRUVmzZg2CIMHBwXFxcXhkB0AwCoXS0tK+ffvS6fSSkhIHBwctLS0EQWg0WkNDQ1paGoIgVCpVVlaWSqUivR2RSKRQKNi8TEJCQgsXLiSRSGw2+86dOwEBAXgHBKB3QsdV6evrY5PDPn/+3NDQEJ0glUAgjBs3TkpK6tsd58yZM3jwYAaDsX///srKyi4PDoBgFAoGBgZWVlZod8Xhw4ejC8vKyqqqqtBWBBKJpKqq2svO78AXOsUE7xJVVVWsQkL7VQEAOlxpaWl6erqJiQlapqekpHz48GHt2rW8hxv4EhcXRxsVHj58aGVlhf63AQB0cKFAIpGIROKHDx+kpaU1NTXRhegcxmjrApVKnT59Ot9yvpchEontumLo6ent2rULPZ/ky5cvYcpYADpDZmZmSUlJeXk5nU4vLS319PScO3eura3tz+wrKyuLXkhJSfHz8+vkpAAI8KiHuLi4wYMHo6dTYrPZkZGRFhYWxsbGaCWhqakpCIceJk+e7OXlpaioiC0hk8nbtm2LiIgQExP7+PHjmTNncA0IQO+UkJCgpaWlp6d3+/btq1evTpo0aceOHbzjIX9AVVUV674As6kC0FmFQnV1dXJyMtaNKDY2Nj09fePGjWJiYogg0dDQsLKy+ra109LScsmSJQiCeHt7Q9smAB2Lw+HEx8ebmpra29tPmTLFzc1t+vTpP7+7iYlJYGAg2pshKSmpurq6M8MCIKiFwufPn+vr60tLS1NTU6OjowMDA3fu3ImNUwIIgqxfv15HR6e4uPjgwYMCNaUEAJ2tsrIyIyMDPWe0hITEbwywmjx5sq+vL41Gi46OvnXrVufEBECwC4W3b99qaGgsXLiwsLCwoqJi06ZNM2bMwC9bd9S3b9/169cjCHLr1q1Hjx7hHQeAXqKsrCwoKKisrKy8vLysrOy3b8fS0nLatGkIgpw8ebK2trZDMwKACHqhwGazExIS9PX1Bw0aNHHiRAcHBz09PUQgNTY2lpaWotNMfcvFxcXKyorJZB44cKChoaHL0wHQC5WVlUlJSR05ckRUVPRPxjeSyeQtW7ZISEhkZGT8888/HZoRAETQC4Xi4uLU1FQTExNc83QLd+/eXbp0aWlpKd+1QkJC27dvp9FocXFxV69e7fJ0APRCAwcOXLx48aJFixYvXmxgYPAnN2VgYODi4oIgyIULF+BMUQB0WKGQk5Pj6+vL4XDS09NhvvTy8vK0tDQGg/G9DcaNGzd9+nQ2m33kyJHv1RMAALysWrVKXV09OzsbxkkC0GGFAo1Gmzlz5s2bN+fOnaukpIQINhKJxDsz47cIBMK2bdukpKRyc3NhqCQA3Y22tjZ6jvjz58/n5eXhHQeAXlEoyMvLGxsb6+npmZiYqKqqIoKNw+GwWKwfD2rQ1dVdu3Yt2rwZEhLShekAAP9t1apVqqqqFRUVUMoD0JGjHgBKWVl50KBB/zlZ9dKlS/X09MrKylxcXDw9PWG0JADdh5KS0vLlyxEEuXHjxufPn/GOA0APBoUCHzNnzrx27ZqysvKPN1NUVOzTpw+CIPX19T4+PgUFBV0VEADw3xYsWDBgwICioiLoqQDAn4BC4Xf6KKAIBAI2zXNLSwudTu+SdACAn6KmprZgwQIEQS5dupSSkoJ3HAB6KigUfh+JRDp+/Pi8efNoNFp+fj70VACgu1m2bJmKikpNTc348eNv3boFxwcB+A1QKPwRRUXFy5cvL1y4kMvlnjhx4suXL3gnAgD8j7i4OHrS19LS0i1btkCzHwC/AQoFPj5//hwaGtrY2PiT2+/cuVNTU7OysnLPnj0/mH0BANDFiEQiNoyLzWb/5/FEAMC3oFDgIzw8fN26deXl5T+5vZKS0o4dOxAECQsLCw8P7+R0AICfRaFQPD09zc3NEQRpbW2tq6vDOxEAPQ8UCnxQKBQREZFf+vPh5ORka2vb0tLi7u5eVVXVmekAAL/AyMjo+vXrsrKyZWVlgYGBeMcBoOeBQqFjiIqK7t69m0ajpaamHj9+HO84AID/0dDQmD17NoIgN2/erKiowDsOAD0MFAp8MJnMlpaWX+0gbWZmhp6B2tfX9+XLl52WDgDwy+bPny8pKZmYmBgZGYl3FgB6GCgU+LC2tj5x4oS8vPyv7rhmzRoTE5Oampo9e/YwmczOSQcA+GWDBw8eP348giD+/v4sFgvvOAD0JFAo8GFgYDBjxgxxcfFf3VFOTu7gwYNCQkIvXrzYuXMnnFgSgG6CQqEsX76cSCTGxMQ8ffoU7zgA9CRQKHSwSZMmzZs3D0EQDw+PmTNnFhYW4p0IAPCvv/76a8SIERwOJyAggM1m4x0HgB4DCoWOZ2lpiV6Ijo5+8OAB3nEAAP9rVCCTyU+fPo2Pj8c7DgA9BhQKfHA4HCaT+duzvaqoqIiIiKCXi4uLOzQaAOD3TZgwwcjIqLa29urVq3hnAaDHgEKBj1u3bs2ZM+e3exiMHz/+/PnzgwYNQhDk+vXrmZmZHR0QAPA75OTk5s6diyBIcHBwbm4u3nEA6BmgUOCjuLg4MTGxpaXl93YnEAjOzs5Xr15VV1fPzMw8ePAgnIoGgG5izpw5ysrKdXV1AQEBeGcBoGeAQoEPIpFIJpP/cFp4PT09dF7noKCgW7dudVw6AMDvU1RUXLx4MTr5UkFBAd5xAOgBoFDgg81m/0kfBYyLi4u1tTWHw9m7d29eXl4HpQMA/JGZM2cqKSllZGTcu3cP7ywA9ABQKPAhLy+vq6tLpVL/8HaEhIQOHz6sqKiYlpa2b98+OAABQHcwcODAKVOmIAhy9OhROIsbAP8JCgU+7O3t/f39FRUV//ymjI2Nt2zZgiDI5cuXQ0JCOiIdAOBP6erqIghSUFAwb948GCoJwI9BocCHuLi4iooKmUzukFtzc3OztrZmsVg7duzIycnpkNsEAPyJhoYG9EJNTU1+fj7ecQDo1qBQ6HRCQkIHDhxQUVHJyMjYs2dPU1MT3okAEHSOjo4GBgboZfhIAvBjUCh0BRMTk507d5JIpMuXL5uYmMC5JQHAl46OzqNHj8aNG4cgiI+Pz2+PhQZAEEChwEdeXt6rV6869rvDycmJQqEgCJKRkXHgwAGYah4AfKmqqrq7u1MolPj4eOg/BMAPQKHAR1hY2IoVKzr23I+ioqJ9+/ZFL2dnZ7e2tnbgjQMAfoOlpeW0adMQBPH09KypqcE7DgDdFBQKfHA4nA4/Yz2VSvX39x82bBja1zooKKhjbx8A8KsIBMLKlSslJSUTExPv3r2LdxwAuikoFPggtOnwmx05cmRUVJSTkxOLxdq3b19iYmKH3wUA4JeMHDnSzs6OzWafPHmyvr4e7zgAdEdQKHTizIzfEhER2b9/v7a2dlFR0fbt2xkMRoffBQDg5xEIhHXr1gkLC6ekpFy+fBnvOAB0R1Ao8GFubr58+XJpaenOuPH+/fsfPXqUQqE8fvz4+PHjnXEXAICfZ2pqOn/+fARBTp8+DTOdAPAtKBT4sLS03Lx5cycVCgiCODg4LFmyBEEQDw8PGCoJAO5Wr16tqqqanZ3t5+eHdxYAuh0oFPCxa9cuCwuLurq6NWvWlJeX4x0HAIFmaGi4dOlSBEG8vLxSUlLwjgNA9wKFAj6UlZU9PDxoNFpSUtLu3bvhfFEA4GvlypVaWloNDQ1Hjhzp8EFPAPRoUCjgZuTIkbt27SKTyX5+fsuXL09LS8M7EQCCS15eftu2bUJCQleuXHF2di4qKsI7EQDdBRQKfDx48MDNza2srKyz72jVqlUjRozgcDh+fn5OTk7Z2dmdfY8AgO8xMzNDeybdunVr3759eMcBoLuAQoGPL1++3L9/vwtOFSMmJqampoZeTkpKysrK6ux7BAB8j3Ab9DK0KACAgUKBDzKZLCIi0hlzLn1r3rx5srKy6OXnz593wT0CAPjS1NTcsmWLqKgogiBZWVm5ubl4JwKgW4BCgQ9um665rwkTJjx//nzOnDnoMO779+93zf0CANohk8krV6589eqVqqrqly9fPDw88E4EQLcAhQIfFApFWFi4a1oUEAQxMjLy9vYeMWIEg8HYvHkzHIAAAEdDhgxZvXo1giAXL16Mjo7GOw4A+INCgY+5c+eGh4f36dOny+5RXFz83LlzysrKX7582bhxI0ztDACO3NzcTE1NW1pa9uzZ09jYiHccAHAGhQIfkpKSffr0IZPJXXmngwYNOnjwIIFAuHfv3smTJ7vyrgEAvMTExNzd3YWFhV++fBkcHIx3HABwBoVCN+Li4uLq6oogyKFDh+7du4d3HAAEl52d3ezZs9ETvcK4ZSDgoFDoXg4cODB8+PDGxsYNGzZAZwUA8EIgEHbt2qWiolJSUuLu7g5zNQJBBoUCH6WlpUlJSbh0FJCVlfXx8VFWVs7Ozp4yZcq5c+daW1u7PgYAQFNTc9++fUJCQjdu3Lh69SrecQDADRQKfISFhS1durQLZmbky8jIaPv27QiCfP78efXq1T4+PrjEAAA4Ozvb2tqy2ew9e/Z8/foV7zgA4AMKBT5aWlqqq6vZbDZeAQwMDIjE//fSvH79Gq8YAAg4ERGRQ4cOKSoq5uTkmJqanjhxAsevBQDwAoUCHwQCgUQi4Rhg8ODBs2bNQjPEx8d/+vQJxzAACDJdXV1LS0sEQWpqavbv319YWIh3IgC6GhQKfLDZ7NbWVhxP/SwtLX358uXQ0FANDY2CggJXV9fKykq8wgAg4DQ1NdELzc3NxcXFeMcBoKtBocCHoaHhzJkzxcXFccxAoVAmT5587tw5YWHh+Pj4HTt2QJsnALhwc3Ozt7cnkUhMJtPLy4vJZOKdCIAuBYUCHxMmTDh69Ki8vDzeQRAbG5u9e/eSSCR/f/9Tp07hHQcAQaSurn7nzh0vLy8hIaErV64EBATgnQiALgWFQnfn5ubm6OiIIMjevXsjIyPxjgOAICISiQsWLJg2bRqCIHv27ElLS8M7EQBdBwqF7k5UVNTT09PCwqKxsdHNzS0lJQXvRAAIImFh4WPHjmlqapaVla1fv76hoQHvRAB0ESgUegA5OTkvLy91dfXs7GwXF5dHjx7BUVIAup6amtqhQ4eEhISePHly/PhxvOMA0EWgUOAjOjraw8Ojuroa6TaGDBni6elJpVITEhJsbW03bdoEfRsB6HqOjo5ubm5cLvfo0aP379/HOw4AXQEKBT7i4+P/+eef2tpapDuxs7NDz3zNZrPPnz8PUzsDgItdu3aNGDGCwWBs2rTp8+fPeMcBoNNBocAHmUymUqkEAgHpTohE4tSpU9HLra2t4eHheCcCQBBJSUn9888/ampqGRkZDg4O3t7eLS0teIcCoBNBocAHl8vlcDhIN0MkEt3d3b28vAwMDFgs1tq1a9+9e4d3KAAE0cCBA1evXo0gyJcvX9auXXv79m28EwHQiaBQ4INEIgkJCXW3FgUEQSQlJVeuXHnz5s0BAwYUFxcvWrQIJpQFABfm5uboCVmYTGZcXBzecQDoRFAo8GFvb+/v76+oqIh0S/r6+t7e3pKSkikpKatXr66vr8c7EQACx8TEZPbs2WQyGT3fLIxbBr0YFAp8qKmpDRs2TEREBOmuxowZc/LkSRERkbt3727dupXFYuGdCADBIi4uHhgYGB4erq6uXlRUtGzZsoqKCrxDAdApoFDoqebPn79hwwYEQXx9fWF2ZwC6HoVCmTBhwpkzZ8TFxd+8ebN169Zu2LcJgD8HhUJPRSaT9+zZM2/ePC6X6+7uvmXLlsTERLxDASBwpk6dunv3bgRBLl68eOLECRzPOgtAJ4FCgQ8mk9nc3Nz9P/BkMvn06dPDhw+n0+keHh7W1tZQKwDQ9dasWTNv3jwEQfbt2wcjIEDvA4UCH/7+/paWlvn5+Ui3JyMjY2dnh14uLy+HAZMAdD0KhXL8+PFx48Y1NjauWLEiNjYW70QAdCQoFPhobW1tbGzsKYcbJ0+ejM7YiCBIYGBgWVkZ3okAEDgKCgre3t7a2tpVVVXOzs4bNmwoLy/HOxQAHQMKBT4IBAI6QrpHMDQ0fPLkyfbt28XFxePi4pYsWVJXV4d3KAAEjpaWlre3t5SUVF5e3qlTp1asWMFgMPAOBUAH6DE/h12JxWIxGIzu30cBo6ure+jQocOHD5NIpAcPHmzevBlOGQVA17OwsNDX10cvv3//vkccvgTgP0GhwIeGhsbo0aNFRUWRHmXVqlXu7u5oH4u9e/fC5AoAdDEajebk5IROwZKXl3f69Gn4GIJeAAqFnjcz4w9s27Zt8eLFCIJ4eHj4+PjgHQcAgePm5hYREbFs2TIEQby9vU+cOIF3IgD+1L/zj4J2iG2QHkhISOjkyZMNDQ03b97cvHkziUSys7NTU1PDOxcAAmTUqFEjRoxobm4ODg52d3eXlJRcvnw53qEA+H098ucQ/ICEhMTZs2dHjhxJp9NXrVo1efLkFy9e4B0KAMFCJpPPnDkzadKk1tbWLVu23Lp1C+9EAPw+KBR6IQUFhUOHDomLi3O53E+fPnl4eOCdCACBIy0tfeHChWHDhjU0NCxfvjw0NBTvRAD8JigU+EhMTLxw4UKPHmSoq6traGiIXv706dOHDx/wTgSAwFFWVr58+fLgwYOrq6uXL1++YcOGjx8/4h0KgF8GhQIfr169OnLkSHV1NdJjycnJnT59esqUKdLS0kVFRYsXL87NzcU7FAACR0tL6+rVqwMGDCgvLz916pSdnV16ejreoQD4NVAo8EFog/Rw5ubmYWFhFy5cEBcX//Dhw9KlS0tKSvAOBYDA0dXVdXFxQS8XFRXdv38f70QA/BooFHo5e3t7Hx8fcXHxp0+fLl26tKamBu9EAAiccePGKSkpoZf9/Pzevn2LdyIAfgEUCvz1ghYFjLOz85EjR4hEYnh4+Nq1axsbG/FOBIBgMTU1jYiIOHv2rJqaWlZW1rx58z59+oR3KAB+FsyjwMeECRPU1dXl5eWR3mLlypX19fW7d+8OCgpCz4o7aNAgMhlefQC6yKA2RkZGTk5OGRkZzs7Oly5dMjU1xTsXAP8NWhT40NPTc3BwoNFoSC+ycePGzZs3IwgSFBRkbW29c+dOmFwWgC42atSoCxcuqKiopKWlOTk5wXnhQY8AhYKgoFAoe/futbKyQhCksrLy9OnTSUlJeIcCQOBMmjQpKChIRUUlKytr9uzZQUFB2dnZeIcC4EegUBAgJBLJxsYGvdza2nrlyhW8EwEgiMaMGRMUFKShofH169f58+dbW1s/fvwY71AAfBcUCnzQ6fS6ujoOh4P0OsuXL9+5c+eQIUMQBDl9+vSBAwfwTgSAIBozZkxAQACJREIQJDMz89ChQ3gnAuC7oFDgIywsbOnSpaWlpUivQ6PRDhw4EBERMXPmTA6Hs2/fPk9PT7xDASCIhg4dqqmpiV6Oj48PDg7GOxEA/EGhwEdRUVFCQgKdTkd6KXl5+fPnz0+ZMoXJZG7atOnkyZN4JwJA4IiIiAQHBzs5Oenr69Pp9GXLlp05cwbvUADwAYUCHyQSSUhIqDdNpfAtCQmJgIAAa2trFou1Y8cOb29vvBMBIHDMzc2vXr364MGDv/76q6WlZcuWLUePHuVyuXjnAuD/gEKBDy6Xy2azkd5OTk4uKCho4sSJDAZj/fr1K1eufPToUa/smQFAd6ahoXHr1i1bW1sGg7Ft27bVq1c3NzfjHQqA/4FCgQ9JSUl1dXUKhYL0drKyspcvXx49ejSDwfD29p42bRqcDBeArqegoHDp0qV58+YhCOLl5eXg4HDgwIHCwkK8cwHwLygU+Jg3b154eLiqqioiABQUFHbu3Ekk/vtOaG5uhjGTAOBCVlb2woULW7ZsERUVffz48a5du5ydnSsqKvDOBQAUCvyQyWRhYeHe3UeBl7m5+ZgxY9DLERERPj4+eCcCQBCRyeSDBw+OGDECvRodHf3kyRO8QwEAhQJo69h49erVCxcuDB8+nE6nr127FsZMAoALMpm8aNEiERERtLPUhg0bbt++jXcoIOigUAD/kpeXd3FxCQkJGT9+fGtr67Zt27y8vPAOBYAgmj17dkREREBAgLa2dllZ2eLFi2EAM8AXFAp8ZGdnP336VAA7HisqKl69enXixIl0On3dunVHjhzBOxEAgmjUqFGLFi26d+/emDFj6uvrN27cuHz58srKSrxzAQEFhYJgzcz4n+Tk5C5fvmxjY8NisXbt2rV58+a3b9/CwG4Aup6uru61a9ecnZ0RBPH19Z0+ffo///zz6dMnvHMBgQOFAh8kEolKpQpOZ8Z2FBQUrly5Ym9vz2Kxjh8/Pnr06L1797a2tuKdCwCBo6CgEBgYePDgQRqN9urVq1WrVo0bN+7Fixd45wKCBQoFwIeUlJSXl5euri6CIC0tLRcuXMjLy8M7FACCiEKh7NixY+fOnejVyspK6GsMuhgUCnywWCw6nS7g7e0qKiq2trbo5YKCAg8PDwHstAFANzFz5kw9PT30clhYmIuLS1lZGd6hgKAg4x2gOxo9erSkpKSsrCwi2Hbt2qWkpPTw4cPnz5/7+/sTCIQTJ07QaDS8cwEgcPr373/9+vUXL148f/787t27gYGBWVlZp06dMjU1xTsa6P2gRYGPIUOGLFmyRFJSEhFskpKSmzZtioiI2LBhA5FI9PPzW7x4cXV1Nd65ABBEAwcOXLNmze3bt48dOyYpKRkdHW1jY3Px4kU4PwvobFAogP9ApVIPHTq0Y8cOBEFu3rzp4uJSXFyMdygABBSJRNq8efO1a9e0tbUrKipcXV2tra0XLVr0+fNnvKOBXgsKBfDfqFTq/v37jx49KiIicu/ePUdHx5s3b2ZnZ+OdCwABZW1tHRkZOWPGDCaT+fjx44sXLy5atAjvUKDXgkJBcE8z/as2b9586tQpERGR6OhoR0dHBwcHqBUAwIumpub58+f79u2LXo2Pj/f09IQvLtAZoFDg4969eytWrIBOxe0QCARXV9ctW7agV5OTky9fvox3KAAEl6SkpIeHR79+/UgkEpfLXbdu3Zw5c3JycvDOBXobKBT4yM7OjoqKampqwjtIdzRx4kRFRUX0sr+//6NHj/BOBIDgmjlzZkSb+fPnIwhy48YNGxubkJAQvHOBXgUKBT5IJBKFQhHYmRl/bNiwYVeuXFm2bJmmpmZJSYmjo+P169fxDgWA4NLV1R0/fnxgYKCPj4+qqmp6evqsWbOWLVvm7e2dkJCAdzrQG0ChwAeHw2GxWAI+4dIPjB071tfXNzQ0dODAgXV1da6urufOncM7FAACDT0yGB4ePm7cODab7e/vv3LlyqlTp6ampuIdDfR4UCjwISkpqaamRqFQ8A7SrQ0cOPDOnTtjx46tr69fvXr133//zWAw8A4FgEAzNja+d+/ezJkz0avFxcXHjx+vqKjAOxfo2aBQ4MPe3t7f319JSQnvIN2dlpYWevooBEEOHTpkbW29bt066AQKAI5ERES2b9+uoqKCXg0MDLS3t3/27BneuUAPBoUCHzIyMlpaWtCi8DMUFRWDgoJWrlzJ5XKfP3/u6em5detWvEMBINAGDx78+PFjPz8/Nzc3SUnJ2NhYGxub7du3l5aW4h0N9EhQKIA/RaPRDh06hJ0D4u7du3BYFAB8GRgYLF269OzZsyEhIWZmZgwG48iRI7a2tgsWLDh48CCdTsc7IOhJoFAAHUBSUnLv3r3y8vIIgtTV1U2fPv358+d4hwIAIGPGjHn8+LG7u7uMjExiYuLly5f//vvv3bt3450L9CRQKPBRVlaWlJQEXfN+ydq1ayMiInx8fFRUVL58+eLk5HT79m28QwEAECkpqb179547d05ERARdEhAQEBgYCGeTAj8JCgU+bt26NWfOnJKSEryD9CQkEmnIkCGurq537tzR09MrKyubN2/e0aNH4csIgO7A0dFx5cqVROK/3/lVVVUuLi6Ojo4fPnzAOxfoAaBQ4IPFYrW2tsI8Cr9n6NChoaGho0aNotPp27dvX7x48fXr16uqqvDOBYBAIxKJBw8ejIiICAsLc3BwQBDk9u3btra2hw8frqiogIlowQ9AocAHoQ3eKXowHR2dkJAQJycnAoEQGBjo7Ozs6OhYXV2Ndy4ABBqVSp0wYcKUKVNu3boVFBSkq6tbUlKyY8cOTU1NVVXVffv2wb8jwBcUCnywWCwGgwGfmT8hIyPj5+c3cuRI9GycUVFRT548wTsUAOBfJBJp7ty5ERERGzZsIBKJjY2NdXV1+/fvhz7IgC8oFPgwNjaeM2eOhIQE3kF6NhqNtmjRIjKZjF7dtGlTeHg43qEAAP9Pv379Tpw4sWrVKvQqi8WaOXPmvn37YLoF0A4UCnyMGzfuwIEDcnJyeAfp8ebPn3/z5k13d/c+ffoUFhbOnTv3n3/+wTsUAOB/duzYsWDBAkNDQyUlperq6t27d9va2gYHBzOZTOiJDFBQKIDO5eDgsHfv3gcPHgwfPry2tnbVqlVr166tra3FOxcA4F9KSkqBgYEJCQmJiYkbN26UlZVNTEycN2/eqFGjZsyY4eHh0djYiHdGgDMoFEBXMDY2vnXr1vTp0xEEOXPmjLOzc1hYWE5ODt65AAD/olKpysrKx48ff/jwoYODA4FAePPmTWho6LZt2y5cuIB3OoAzKBRAF1FRUbly5cquXbuEhYUjIiKmTZs2cuTI169f450LAPA/5ubmt2/fPn36NHqVw+Fs27Zt9+7deXl5eEcDuIFCgY8nT57s2LGjsrIS7yC9DZVK3bdv344dO9AvoKKiIuz7CADQTRCJRFdXVxcXF5E2LS0t+/bts7a2/ueffxoaGvBOB3AAhQIfSUlJV69era+vxztI7zRnzpwBAwagl+/du7dlyxY4CApAt0KlUv39/dPT05OTk3fu3KmkpPT58+dVq1b99ddfrq6ua9euzcrKwjsj6DpQKPBBJpOFhYVhzqVOoqmpGRQUtHXr1mHDhrFYLA8Pj4ULF0LDJgDdColEUldX79+//4EDByIiIpYuXSokJPTx40c/P78zZ86sXr0aplsVHFAo8MFtg3eK3szc3PzIkSOPHz/esGGDkJDQnTt3bG1tX758iXcuAAAfgwYN8vPze/z4saqqKrokMjLS2dn52bNnMIRSEEChwAeZTBYSEoIWhc5Go9E8PDy8vLykpKRSU1Nnzpy5fv3669evw5EIALqhUaNGeXh4yMrKioqKEgiEx48f29nZOTo6xsTEMNrgHRB0FigU+Jg5c+a1a9eUlZXxDtL7EYnEJUuW3Lt3z8TEpKKi4vTp087Ozm5ubnjnAgDw4eTk9Pbt248fP16/fn306NEtLS23b9+eMGHCsGHDLCwsDh8+zGaz8c4IOh4UCnwoKioaGhpSqVS8gwiKkSNH3rx5U0FBAT3uExwcnJiYiHcoAAAfmpqa2tras2bNCgsLu3z5sqmpaXNz84cPHz59+rRjx45Lly7BcdveBwoF0C30799/zZo16OEeNpttb29/7do1vEMBAL5LQkJi3rx5r169Wr9+PbbQ1dV17ty5sbGxuEYDHQwKBdBdbNiwISQk5PDhwxoaGgUFBYsXL3Z3d29qasI7FwDgu0RERHbu3Ong4ECj0eTk5Fgs1tWrV62trWfPnv3ixYvk5OTs7Gy8M4I/BYUCH1VVVZmZmUwmE+8ggkVERMTe3n7btm0RERHjxo1raWnZv3//jBkzvnz5gnc0AMB3ycrK3r59u6SkJDs729fX18LCor6+/saNG+PHj7ewsPjrr7/u3LmDd0bwR6BQ4OPu3bvLli2Dc63iRUdH58aNGxs2bCAQCJGRkdbW1pMnTz516hQMxAKgeyISiTQaTUJCYtmyZQ8fPgwMDDQ1NWWxWC0tLcXFxatXr757925rayveMcFvgkKBj/r6+oKCAmhRwJGMjMyJEyeuXLmipKSUk5Pz4MGDDRs2hIeH450LAPAfZGRkFixY8PTp00GDBqFLSkpKpk2bZmVldeXKFZgavyci4x2gOyISiWQyGeZRwJ2Tk1NhYeGWLVvQq0ePHh0wYICOjg7euQAA/0FSUvLKlSsBAQEkEik1NfXRo0exbUxNTbW0tFpaWlxdXa2trfGOCX4KFAp8sNlsJpMJg3y6g6VLl8bFxUVHR5eXl8fExEycOPHIkSOzZ8/GOxcA4D/o6+ufOHEC/UZ98eKFn59fZGRkQhsEQd6/f3/+/PmJEyfiHRP8Nzj0wIeWltb48ePFxMTwDgIQKSmpy5cvR0REBAQEqKur5+Xlubi4bNmypba2Fu9oAICfQiKRxo4de+PGjcjISAsLC3RhYWHhwoULXV1d3759C92PujloUeBj8uTJdnZ2RCJUUd2CmJiYSZthw4atW7fu8ePHHh4esbGxbm5uMjIyEyZMwDsgAOCnDBs27Pz584sWLSosLKRQKPn5+X5+fhcvXrSxsZkxY0Zzc3Pfvn2hjaEbgkKBD0IbvFOA9vT09G7cuHHy5MmjR4/GtCESiadOnVqzZg3e0QAAP8XQ0DAsLKy8vJxCoTx//tzX1zc5OTmsDYIgoqKi58+fd3Jywjsm+D/gTzPoSaSkpPbt2xcQECAiIoIgCIfD8fLy+vz5M965AAA/S1lZ2djYWF9ff9WqVW/evLlx48bQoUPRVc3NzStXrty6deu7d+/wjgn+BwoF0PPMnDkT6y+dkZExbdo0GDkJQE8kJiY2a9askJAQc3NztDdDbW3tsWPHrK2tp0+fHhYW9vz58ydPnsBgdXzBoQc+EhISEhMTHR0dJSUl8c4C+KBSqX5+fiYmJnl5effv309PT58+ffrq1au3bdsmKyuLdzoAwK9RVlYOCQl58uSJvLz8+/fv7969++HDh5CQkLt375JIJC6Xu3LlSk9PT7xjCi5oUeDj5cuXhw4dqqqqwjsI+C5ZWdmdO3f6+fndvXt31KhRDAbj+PHjM2bMePHiRXFxMQxtBaBnUVVVXbhwoa2trbu7e0RERFBQ0Pjx4zkcDpPJZLFYZ86ccXFxefXqFTQt4AIKBT7IZLKwsDD0Z+wRLCwswsPDt2/fLi4u/uLFi/Hjx2tra69cuRLmiwWgh1JUVJw7d+79+/cXL16MLQwMDBzTxt/fPyoqKiQkBE431WXg0APo8cTExA4dOmRpablkyZKSkhIWi+Xj42NjYzN58mS8owEAfhOVSj1x4oSZmRmDweByuVeuXElMTIxuQ6FQmEzm4MGDg4OD9fX18U7a+0GLAh9sNhv+j/Y4NjY2mzZtwq4uW7bM398fGioB6LkkJSVdXV3XrFmzdu3a169fR0ZGzp8/X0ZGBv1cf/jwYc6cOd7e3unp6Xgn7eWgRYEPe3t7Y2NjRUVFvIOAX7N48eLGxsbo6Ojk5OTS0lJXV9dXr17t27dPQ0MD72gAgD9CpVLHtQkJCVmwYEFjYyOCIB8/fly5cmW/fv1Gjx49efLk4uJieXl5R0dHvMP2NlAo8KHZBu8U4JdJSkq6u7uzWKy8vLwdO3bcvHkzODg4Ojp69+7dxsbG6urqMCYCgJ5u2rRpYmJikZGR/fr1S01NffLkSW5ubmBg4OXLlzkcDoVCyczM3LFjB0yt24GgUAC9DZlM7t+//+XLl8eMGbN79+7c3FwXFxdFRUUDAwMvLy9dXV28AwIA/sjENujl9PT0Fy9eBAYGxsfHIwjCZDIPHDgQFxc3c+ZMW1tbOTk5vMP2BlBzgd6JSqW6urpGRUWNGzcOQZCysrJnz57t37+fzWbjHQ0A0GF0dXWXL1/+8OHDuXPnoksYDEZ4ePjChQuNjIzWrFnj7u6+bNmy6OhovJP2YNCiwAerDZVKhRGSPZ2BgcHZs2fNzMzQI5o3btwQFhZ2d3fv27cv3tEAAB1GRkbmzJkzNjY2VCqVyWTeuXMnIiKitLT07Nmz6AaPHj0KCgoaNmwYhULBO2zPAy0KfAQFBdnY2BQVFeEdBHQAXV3dS5cu2dnZjRgxgs1mX7hwwcrK6sqVKywWC+9oAIAOIy0t7eTkNG3aNEdHx+vXryclJR07dqxfv37o2vz8fLQvpLe3d3JycktLS05OTn19Pd6pewYoFPioq6srKCiAkXW9xrRp0+7fvx8REXH27Fl5efmcnJwFCxa4uLjExMR8/vwZpnEEoJchEokaGhqbN28ODg7W1dUVERGRlpZms9mvXr1auXLlpEmTbGxsrK2tFy5cmJubi3fYHgAKBT4IBAKJRMI7Behg4uLibm5uz549c3BwYLPZwcHBY8aMsbCwOHLkCHRcAKBXGjFixPPnzxMSErKzs8PCwlxcXLS1tYuLi1+8ePHly5fQ0NB58+aFhITk5+fjnbRbgz4K351wCf5o9kqGhoZXr1719/ffuXNnQ0NDa2urh4fH0KFDrays8I4GAOh4Sm0QBLFrk52d/eDBgy1btqCz6qFTPerr648YMcLW1ra0tJROpy9evJhGo+EdvBuBQoEPVVVVMzMzYWFhvIOATiEsLLx69eqMjIxz584hCFJTUzNz5swdO3YsWbJEQkIC73QAgE7Uv3//tWvXSkpKBgcHy8jItLS0xMXFpbUJCAjgcDgIgrx69er06dNqamp4h+0uoFDgw97e3traGirK3m3Pnj2ysrI5OTmvX7/OycnZuHFjeHj4/v37hw8fjnc0AEDnWrhwobOzs5CQEIfD+fDhQ0xMzK1bt7AhlCEhIV++fEFnexw5ciSJRGIymYL8iwB9FPigUqkSEhIwsVfvJisru2fPnkuXLr169Wr58uXCwsLPnj2bNGnSunXrzp07Fxsbi3dAAEAnEhISQrs9DhkyZM2aNc+ePfv777+xVampqV5eXjY2NoMGDbKwsFBWVl66dCmdTkcEErQoAEHXp08fb29va2vrvXv3JiYmenp6IgiioqISFhZmamqKdzoAQFegUCh79uwxMzNrbW2VlZVNSEi4ffv2x48fMzMz0Q3Onz9fXV09f/780aNHS0pKIoIECgUA/jVlypS//vprzZo1QUFBCIIUFxdv2bLFy8tLT08P72gAgK5AIpGmTJmCXrayslq7dm16evq5c+ewvgshISGhoaEqKiqTJk0aMWLEly9fZGRkXF1de33dAIUCH2lpaenp6RMmTBDkg1ICSEpKateuXYmJiampqQiCPH/+3NraesuWLS4uLiIiIninAwB0KSEhoYEDB54+fVpWVvb+/ftmZmZ5eXnx8fFFRUUBbdDN3r9/f/To0T59+pDJvfb3tNc+sD/x5MkTb2/vwYMHQ6EgaLS1te/evRsbG5ufn3/x4sWvX7+uWrXq1q1bu3fvNjAwEBMTExUVxTsjAKDriIqKHj58eM+ePVQqlcVi5eTkhIWFRUVFRUZGohvcvHnz4cOHw4cPnzx5sqmpaUtLS11dnaWlZW86HxUUCvzBJAoCS6sNgiCzZs06duzYxYsXX7x4kZiYqK2tLS8vf/DgQRMTE7wzAgC6FJVKRc9Mq62tvWnTpnXr1m3YsMHf3x9dXldX97iNhIQEo42Dg4Onp2evGWAJhQIA/A0YMOD8+fP29vaHDx+OjY19//49giDV1dVPnz4VFxfHOx0AADdkMvnUqVPz588XFxeXlZVFGxjev3+fnJyMbhAaGpqYmGhnZ2dlZWVsbKygoJCdna2qqqqgoID0QDACkA9uG7xTgG7Bzs7u1q1bWCvi27dvZ8+eDYMnARBwJBLJ1NRUR0dHTk7O0dHx4sWLkZGR+/btQ3sqEInEvLw8Ly+vGTNmWP//HBwcvnz5gvRA0KLAx19//SUpKSkjI4N3ENAtqKioXLhw4ejRo83NzampqQ8fPnz+/PnKlStXrVqloaGBdzoAQLegoqKya9euvn37pqWljRgxIiUl5fnz56mpqVltEAQpKyuztraeO3eupaWlvr5+nz596uvrWSxW9/+tgUKBD5M2eKcA3cjkyZNHjRrV3Nz85s2bffv2ffz48cSJE+Hh4ZMmTRITE5s3b56Ojg7eGQEA+Js/fz56YfLkyVu2bElLS4uIiNi/f39jYyOCIDk5Ofv370cQxMDAQFtbu7q6mk6n79y5ExuW2T1BoQDAT5Fo4+DgMHr0aG9vby8vr/Q2CILcv38/PDy8T58+eGcEAHQjJBLJqE2/fv2uXLmioqIiIiKSkJCQlJSU2gbdzMnJaePGjVgzA4IgDQ0N3aojFBQKAPwaaWnpHTt22NnZLV68OCEhAUGQpKSk9evXHz16VFNTE+90AIBuZ1Yb9HJTU1NKSsr79++PHDlSUFCAIEhzczPazGBoaKivr0+hUPLz88eNG/f33393kzMJQKHAB5vN5nA4ZDKZQCDgnQV0UwMHDjx48OCcOXMqKysRBLl9+3ZMTMy6desWLVrUm8ZPAwA6lpiYmEUbc3PzPXv2tLS0qKqq5ubmpvz/0M1ev36dmZnp6Oiora2NHtlkMBgUCgWX0gEKBT7CwsLu3bt35MgR9CzmAPA1YcKEBw8eFBQU5Obmenp6FhYWbt26NTQ01M3NTVtbW1FRsW/fvnhnBAB0U6amprdv32az2WJiYg0NDZ8/f05KSgoKCnr16hW6QXAbDQ0NAwMDdXX17OxsGo3m4eHR9X2ooVDgIycn5+XLl83NzX9yIyUlJffv3589e7aEhETHRQPdC/rPAJud6cqVK3FxcfHx8VQqVV5e/tq1ayNGjMA7IwCgmxIWFkYviIuLm7eZNWvW+vXr3717p66uzmQyk5OTc9pgu6Snp69bt87IyEhHR0dKSorD4TQ1NXV2h4YeXCjQ6XQhIaHOaIchk8lUKvUPjzvU1NQ8ePBgypQpUCgIAnV19XPnzjk7O7u7u0dFRdHp9IKCgo0bN167dg2GUAIAfpKEhISvr29paamqqiqbzU5LS/vy5cvDhw+vX7+OnuQ6NTV16dKlkpKSem1qa2vLy8uXLl26YMECpGcVCg0NDenp6YaGhp1xKp2qqqrS0tK4uLgHDx6cPXu2M7qac7lc9Fxhf0JJScnCwgIOVwuU4cOHnzt3ztbW9uvXrwiCxMfHDx8+fP369fPnz4fDWL1MRUVFUVGRsbEx9GQCHYtMJqO/a2QyeWCbmTNn/vXXX6dOnRITE1NWVs7MzExNTY1rg+4SFxf37t27sWPH6ujoDBgwgEwmFxQUyMjIiImJdUwkpBNEREScOHHizp07Hf4r3traeu/evfr6+vdtOmn+RCqVKiEh8attFWgXSAqFwmAw/Pz88vPzW1patm3bZmBg4OLi0hk5QTekq6sbFBR0584dOp0eFhZWVFS0devWa9eubdy40cbGBkGQ7j+5Si9QWloaHBzs7OysoqLSsbdcX1//8ePHtLQ0tK3o4sWLHXv7APDl4uIyYcIEUVFRaWnpwsLCL1++vH//PjAw8PPnz+ivj1cbNTU1HR0dCQmJL1++GBoaHjlypF+/fkj3LBRev36tpKQkLy/f4bdMoVCcnJzIZHJAQEB0dDTSOZYsWbJgwYJfag5hsVjbt29XUFDYvHkzmUw2MjJSVlZGXyr4KylohrdBEGTr1q0nT54MCgr6+PHj/PnzRUVFhYSEdu/evXbtWrwz9nKvXr3y9vaePn16h99yY2NjQUHBgAEDSCQSk8nkcrnQogC6hqqqKnqhT5uxY8dOmzZt+/btX79+ReduSk9PL2iDbpaampqUlDR79mwTE5MBAwZoaGhkZmYyGIzBgwfjWSigLfb19fWpqakjR44kk8kcDqdj+xAQCAS09webzUY6DaXNL+1SVlb29OnTpUuXopNsjB49Ojs7+/79+25ubpKSkp2WFHRr6urqp0+fdnR09PT0vHHjRlObnTt3WlhYDB06FO90vdn79+91dHQ6Y9SJiorKnDlz2Gz26dOnO/zGAfglWlpaly5damhoUFRUZLFYGW2ePHni6+uL/kR+/vx59+7dCIJoaGhoamqihcLWrVvd3Nx+6TeuIwuFzMzMqKiowsLC9PR0HR0dX1/fMWPG6OrqIgIgOzu7ubn5r7/+wpY0NDSUlJQ0NDRAoSDghg0bZmpqih6JQKdbGT9+/MKFC1etWiUgn44uQ6fTy8rKWltbY2NjjY2NCwoKpKSkOuMDyGKx4LxxoDsQbYN2aNBvY29vb2JiEhQUpKysLC0tnZqampmZyTt0YsOGDbdv3x4zZsyQIUMGDBjQ2NiYn59vYmLyg/niOrJQUFBQGDVqVGhoqJiYmJOTk6ysrLKy8rebff36NSUl5QcfMy6XKysrO2LEiG4yKdUPcLncxMTE8vLyyMhItIdqXV3dsGHDiG3IZHL3fwigC1AoFE9PTxkZmby8vMLCwoyMjHPnzj148GD+/PkrVqwQFRUVExMjkUh4x+zxysrK7t69W1RUlJGRoaOjc+fOnZEjR5qZmX275X/+zMMBBdCztLS0YIfL58yZ4+joSKPR0Gkfs7KyMjIyjh079u7dO3SD2DYIgqipqTU1NVVXV+vr6/v7+w8ZMoRKpaJlB5FIxFodOrJQkGpz/vx5AwMD3v/W7ZSUlHz69OkHH0UOh6OiooIe5cVFfn5+YWHh4MGD/7ObApfLra+vr6qqio6O1tXVZbPZTU1N6CoymSwiIgKFAkD17dv3woULLBarsbHR19c3ICAgMzNz3759J0+e5HK548aNCwgIkJWVxTtmz6amprZ8+fJbt27Jy8tv3LhRU1MTPecvLxaLdfXq1fT09B98Njkcztg2nR8ZgD/F5XLfvn375s0bGo1ma2ubnJxcUlJSX19vY2PTv39/UVFRdOiEpaXl33//3dzcbGlpmZWV9fHjx6ysLKxDQ1pa2ujRo3V0dCwtLYcPHy4hIZGRkeHj47N8+XIpKakO7szY3Nz87t27CRMm/GCbEW2QbiwsLMzHxyc8PPw/+4sSiUQrK6u6uroTJ044ODg4OTm1Gzrxq30dQE9UWVn59OnTPn36WFpa5uXlxcfHi4qKWllZfTs2iUwmS0lJbd26dcaMGX5+ft7e3g0NDehbTkdHZ//+/UJCQjg9iN6ASCRSqdSPHz/269dPQ0MD/W/UDoFAMDIyUlVV/UGbAZfL7ZC+4gB0AbQJbeLEiQsWLEhKSpo7dy76x2PXrl0+Pj7YRD5KSkrnz5/H9mIwGNnZ2VFRUXv27Kmurkb7FKJzSPv4+KDbbN++vbq6+siRIx1cKOTk5JSWlvJt6+tB2Gz2Lx2D/PjxY2Njo7GxMe9CKpWqoKAA3/u9Xmtra0REhIaGxrFjx54/fz5gwABLS8vQ0NAdO3YcO3aM728VgiD9+/c/evSovr7+woUL0SXHjh2LiYlZv369ra2tsLAwoU3XPpTeoKGh4dOnT0OHDsXmvGuHRCL9Rq9vALqt9PR0TU1NAoFQW1s7fPhwc3NzBEGMjIw8PDwyMzOHDBnCdy8qlYr2adDV1Y2MjJw0aRKJRIqOjn7//n1mZiY66hJBkJiYGDab3cGFwocPH0RFRdEuWpWVlWJiYt+23j948ODmzZs/vp1+/fq5u7t/22zYNX71O/rdu3cKCgrtJuATFhbW1tbG6yGALpOdnY3+9hQWFvbt23fbtm0UCmX06NHnz59PSkoyMTGpr68nEAhEIpFEIrFYLLRlW0hISExMbM6cOehhdTab/enTp5iYmDdv3lhbW7u5uTGZTOwwFvh5BQUFhYWF6HclX1wut7a2lsFg/Ph2aG06ISAAHczQ0JBGoz18+JBCoYwcORJdWFlZ2djYyGQyQ0NDX758qaGhweFwPn/+rK6uLiMjU1RUJCEhsXr1alFR0fFt0L3GjBmDHnw/ePDg+fPnJSUlly5dSu7wn7G3b9/2799fVVWVTqffv39/6tSp3xYKo0ePHjhw4I+nPhQREflx3y4ikYh++SKdgMViMRiMn2xRYLFY796909fXRyfSaW1tpVAoBAJBVlZ23rx53/tDCXoTMzOzrKyspqYmR0dH9GBTa2trTU1NeXm5n59fTEzMkCFDGAzGtWvXhg0bNmjQoPz8/KysrMuXLwsLC2/ZsmX9+vVcLjcmJub48eOPHj0KDw+PjIxEECQ5OVlOTs7e3h7vx9eTJCUlcblctHkvPT1dQkKi3ZxLTCbTx8cHO0ff99jb28+cObOTwwLQAdCpemJjY7W0tLBpexITE2k0GolESkhImDdvXv/+/d+9e3fu3LmLFy+iA4KCgoK+9xunrq5+6tSp7OzsXbt2jRo1quMnXGKxWPLy8iwW6/Hjx2jl0uGlekNDQ2tra25ubkVFRX5+voiICI1G69gWfgsLCy6XKyUl9TMbl5aWZmVlubq6EgiE3Nzc7OxstCgjEAhQJQgCPT09BEH8/PwkJSWx8UU5OTl0Or25zenTp+Xk5JKSkq5cubJixQpjY+Pm5mYfHx9sLhC0trCysho+fHhkZOTRo0ffvHmDni7k7NmzNjY2cADr56WkpPTt21ddXb2hoeH169fflllCQkLbt2//w3tBzwWDDm76w5sC4M+1tLR8/PhxwoQJ6B/slpaW2NhYc3NzJpM5bNgw9OhDZmZmv379jI2NJSUlWSyWkZHRD3rrU6lUSUnJgQMHolc7uFBYsWLF7du3r127pq6ubmVl1bE3jrbZ3r9/PykpqbKycurUqdeuXZORkZkxY4ahoWEH3otlm5/cuLKysqWlZfDgwWw2+9WrVyYmJnBoWdBwOJx3794NGDBAUVERXRITE6OkpKSgoNCnTx/0fB/x8fFYJcFkMg0MDL7t6EqlUqdOnWpmZjZr1qyYmBh0mhQ4evVL1NXV09LSPn369PHjxyFDhnT4/LD19fXx8fHFxcWpqamioqJXr15VVla2sLBAx7IDgIu8vLyCggLsdz0uLi4/P9/d3V1fXx/9nkG/o/T19dFpRWg02vjx439Q5nLboIdKO75QMDIy0tXVbW1t7ahzUbRDJBKd2yDdhoaGxsiRI6Ojo1NSUtATh+OdCHS1mpqaz58/z5gxAy3nq6qqnj9/7ujoOGzYMOwIGvopRc8GS6PRRo8e/b12AhUVFS8vL1tbWzMzs3379sF/1l+Cnt8hNzfX3NwcbezpDGJiYsePH0e/TDvpLgD4eUlJSTU1NWVlZeicY35+fqtXr+adYqCysjIzM3PZsmXoVSqViv2r+Rnk7jD/cY8mKSl5+vTpr1+/ysrKwmkdBNPXNtgPv4+Pz4ABA1xdXbEl6LzmixYtQq+S2vzgBo2NjXV0dBwdHTv8nEa9Ho1Gs7Oz67zbl5CQwHp+AdBNxMbGjhgxQlpa+vbt28XFxegZpHg3yMzMrKurMzEx+b3bh1bNDiAiIgINCYIsKSlJQkKipaXl3r17ubm5TU1Np0+f5p09KTs7u76+HmsY/E9o94U/P9c5AKDXq6+vT05OtrKymjZtWl1dHdqHsd02nz59kpaWbjc07+dBocDHo0ePoqKiNm/e3BknwAS9z5s3b8zMzFauXJmbm2tkZNS3b992xwvevXsnKyurpaWFX0YAQO+Un59fXFyM9qv73plN4uLiTExMfrtLABz+5CM1NfX27dvolHkA/FhTU1NKSsqwYcNERET09PQ0NDR4qwQ6nV5TU/P8+XMVFRUikdipZz0FAAiagoKCa9eu1dTUFBcXV1VVtVvL4XCePHly8uTJpKSkioqKGzdu1NfXd3WLQm/t3k8ikYSEhHrrowMdiMFgvHjxIi8vT15ensFgfDsgNjExMSEhQVpaWkJCIiwszM7Oju+Y4XbgvfeT4IkCAq64uLh///6HDh1iMBj19fXtzhdDJBItLS3Nzc2xbow/2ajQ7pP1R4VCb+2Pjc7mBMPSwH9KS0uLiYmZMGFCamqqjo7Ot70Qhrf51ZslEonQnf4/cblcOOUmEHAWbX6wgUibX71ZEonEWyv8/m9hWlra1q1bFRQUellrKplMjo6Ozs3N3bNnj5ycXC97dKBjMZlMDocjKytbWVkZEBDQIeN9CAQCk8l89erVkiVLOiJj78TlcgsKCjZs2EClUqGoAqADEQiE1tbWDx8+YP+WCb/3GcvNzX337t3Pz3Pcs/w7tTWZ3FsfHehA7RroOvANIywsPHz4cFVV1Y66wV4mNzc3ISGBTqfDhxSAziAmJjZlyhS0VvjNQgEAAAAAgqB3djIAAAAAQIeAQgEAAAAA3wWFAgAAAAC+CwoFAAAAAHwXFAoAAAAA+C4oFAAAAADwXVAoAAAAAOC7oFDoVf5zVgyYNgMA0Hl+5tzo8C3U40Ch0HtER0ffv3//x9s0NDRcunSpqKioq0IBAARCbW1tQEBARUUFh8NhMpksFut7RUNERMSrV6+6PCD4fVAo9BKJiYl3794dNGjQjzej0Wja2tonTpwoKyvrqmgAgF6OTqefPXtWWlpaXl4+NTV148aNtra2z54947vx4MGDw8LCXr9+3eUxwW+CQqE3qK6u9vHxcXBwUFdX//GWRCJx+PDh/fr18/HxgQZAAECHuHnzJoPBmDJlCpFIVFdXt7S0/PDhQ3FxMd+NlZWVnZ2d/fz8SktLuzwp+B1QKPQGN2/eFBMT+/nTGc+aNSsxMTEhIaGTcwEAer+8vLzQ0FAnJyf0BEKSkpIWFhZycnI/2GXIkCHKysrBwcFdGBP8PigUeryGhobIyMjx48e3O5PhDygpKRkZGYWEhHRyNABA7xcRESEjI6Onp4ctYTKZ/9lgOWnSpGfPnlVXV3d+QPCnoFDo8T59+lRbW2tsbMy7kMPhfPr06fHjx+Hh4cnJyd/uNWLEiISEhIqKii5MCgDobVgs1rNnz8zMzIjE//NrQiAQiERidnZ2ZGTkkydPvv2qMTAwYDAYiYmJXZsX/A4oFHq8xMREGRmZdg19/v7+N27cQBCkpaXF3d3dz8+v3V5aWlr19fX5+fldGxYA0KuUl5fn5+fr6Oh8uyoqKurNmzcIguTn569Zs6Zd30YZGRl5efkPHz50YVjwm/49pAR6tK9fv8rLy1OpVGwJi8W6evUqh8M5dOgQgiDNzc179uyxsrLS1tbGtpGRkSGRSHl5eUOGDMEpOACgxyspKWEymfLy8u2WM5lMBQUFJycnEomEDotwc3O7c+cOdoSCQqEoKirm5ORwudyfP2wKcAEtCj0bl8utq6uj0Wi8C0kk0unTp0+cONHY2FhbWyshIdHU1NSu8UBYWJhMJtfV1XV5ZABA71FbW0sgEERERNotJxKJGhoaaJWAIMjEiRNbWlqCgoJ4t6HRaLW1tT8zRxPAF7Qo9HgcDufbo4MaGhpXrlwJDg7W0NAoLS1ls9ntPo1EIpFAILBYrC7PCwDoPdAvlnZfQbyrUJKSktLS0snJyUwmk0KhoAtJJBKLxYJx2t0fFAo9G4FAEBMTa2lp4V1YUVGxcOFCLpd77tw5TU3NuLg4tJBnsVgkEglt5WttbWWxWGJiYvhlBwD0eGJiYlwul8FgfLuK94ACm81mMpk0Gg1rY0B7UImJifEuAd0THHro8VRVVaurq3nbBp48efLixYv169dramoiCFJVVUWn08lk8o0bN7ADEA0NDUwmU0lJCb/gAIAeT1FRkUgk1tbWtlvO4XB4q4f8/Pzq6mpbW1us7YHL5VZVVSkrK0MHhe4PCoUez8DAoLKysr6+HlsiIyNDpVIzMjKam5srKio+fvwoKiqalpZWW1srLCyMblNQUEChUPr164dfcABAj6ekpCQrK5uXl8e7kMPhCAkJpaSkZGVlNTU1FRUVnT171traetq0adg2DQ0NJSUlhoaGeKQGvwYOPfR4ZmZmHA4nKyvL3NwcXTJhwgRvb++nT5/6+voKCQnZ2dlpaWk9fPjQyclJUVER3SYhIUFbW1tNTQ3X7ACAnk1MTMzc3Pz9+/eOjo7YQhqNtnPnTiMjoydPnjCZzPz8/JEjR86bN4+3z+PXr1/pdPrQoUNxCg5+AQE6kvR0XC5327ZtcnJymzdvbreqpqZGQkLi20OALS0tixcvnt6mC5MCAHqhuLi4Q4cOXbhwge+0zQ0NDcLCwlgHRsyZM2eysrJOnToFfRS6Pzj00OMRCIT58+cnJiZ+e/JoaWlpvh/CZ8+eiYmJTZo0qasyAgB6LVNTUwMDg7t37/JdKy4u/m2VUF5e/ubNmwULFkCV0CNAodAbGBgY2NjYBAYGtra2/ufGJSUlDx8+XLZsGQx5AAD8OTKZvHz58vfv3/OdLf5bLBYrMDDQysrKxMSk89OBDgCFQi8xZ84cBQWFx48f//hYEp1ODwkJmTZtmpmZWRemAwD0Zn379nV1dY2IiPh2+EM7XC732bNnNBpt0aJFMN6hp4A+Cr0Hk8ksLy9XUVH5wcePTqdXVVWpqqp2bTQAQO9XWlpKa/ODbbhcbklJiaysLO+s86Cbg0IBAAAAAN8Fhx4AAAAA8F1QKAAAAAAA+Z7/DxzroV2xdz/6AAAAAElFTkSuQmCC)

Remark 3. It is worth noting that the price curve defined in Proposition 2 only depends on the gain effect parameter η + . This property will be helpful in designing our learning algorithm in Section 4.

Proof Sketch of Proposition 2. The full proof is provided in Appendix A.3. Below we outline the ideas for proving the optimality of ˜ p ( r ) when customers are loss-neutral. We note that we can re-write the seller's problem P OPT using the following Bellman Equation:

<!-- formula-not-decoded -->

When customers are loss-neutral, namely η + = η -, the single-shot revenue function Rev ( p, r ) is differentiable in p, r . Thus, we are able to apply the first-order optimality condition and envelope theorem for the optimal price to the above program P OPT -BE to deduce a condition on the partial derivatives on the single-shot revenue function Rev ( p, r ) . In particular, we show that the optimal prices p ∗ t , p ∗ t +1 at time round t, t + 1 , repectively must satisfy (partial derivatives are denoted by corresponding subscripts):

<!-- formula-not-decoded -->

Substituting the revenue function Rev ( p, r ) with the base linear demand can lead to the price markdown rule in Proposition 2. Together with Theorem 1, we can deduce that the price curve defined in Proposition 2 is indeed optimal for loss-neutral customers.

/negationslash

For customers with η + = η -, we use Lemma 3.1 and Lemma 3.2 to show that the price curve ˜ p (¯ p ) is always near-optimal.

/negationslash

Computing a near-optimal markdown policy. Solving for an optimal pricing policy (i.e., the set of equations P OPT -BE ) amounts to solving a dynamic program with non-smooth and nonconcave objective function (as customers may have asymmetric reference effects, i.e., η + = η -) and time-variant transition function. Here, we show that, by leveraging the markdown structure for the price curve defined in Proposition 2, we are able to design an efficient algorithm that computes a near-optimal markdown pricing curve.

Proposition 3 (Computing near-optimal markdown) . For any ( η + , η -) , there exists an algorithm (see Algorithm 3) that solves only O (ln T ) linear systems to compute the price curve ˜ p ( r ) defined in Proposition 2 for any r ∈ [0 , ¯ p ] .

Proof Sketch of Proposition 3. The full proof is provided in Appendix A.3. Here we provide a proof sketch for the case η + = η -. When η + = η -, the single-round revenue function Rev ( · , · ) becomes smooth and concave. By leveraging the optimality condition we have derived in (4), we can show that the partial price sequence ( p t ) t ∈ [ t † ,T ] in price curve ˜ p ( r ) can be solved by solving a system of linear equations, i.e., a linear system A p = b where A is a ( T -t † +1) × ( T -t † +1) matrix and b is a ( T -t † +1) -dimensional vector. Both matrix A and vector b can be determined by model parameter I , time round t † , reference price r t † at time t † and sales horizon T (see formal definitions of A and b in Definition A.1). Thus, to determine the optimal policy for loss-neutral customers, it suffices to determine the time round t † (as reference price r t † = r +( t † -1)¯ p t † can be determined by t † ). Since this price curve ˜ p is a markdown curve, we can deduce that the time round t † is the smallest time round such that the corresponding linear system A p = b has a feasible solution. Thus, we can use a binary search to determine the location of time t † . As there at most O (ln T ) steps to binary search the time t † , the algorithm only solves at most O (ln T ) many of linear systems to obtain the price curve ˜ p ( r ) for any r ∈ [0 , ¯ p ] .

## 4 Learning and optimization under demand uncertainty

The proceeding section has analyzed the structure of the near-optimal pricing policy under the ARM when the seller has the complete information about the underlying demand function. This assumption obviously hinders effective application of the resulting pricing policies in practice, where

demand functions are typically unknown and have to be learned from sales data. In this section, we explore the design of the dynamic learning-and-pricing policies in the presence of demand model uncertainty and customer reference effects. In particular, we focus on a base linear demand model, namely, H ( p ) = b -ap , and initially, the seller does not know the model parameter I = ( a, b, η + , η -) . In Section 4.1, we first discuss the challenges in the seller's dynamic pricing and learning problem, and then in Section 4.2, we present solutions and proposed learning algorithm.

## 4.1 The Learning Challenges

When the seller has demand uncertainty, a key difficulty to design a low-regret learning algorithm is the dynamic nature of both the reference-price dynamics and the optimal pricing policy. Such nonstationarity creates unique challenges to our problem. The below mentioned two challenges point out two different technical difficulties that make the techniques from previous literature inapplicable to solve our problem.

/negationslash

Challenge one: unclear how to estimate the model parameter. One tempting dynamic pricing-and-learning algorithm is to first estimate the model parameter I = ( a, b, η + , η -) . If we had a good estimate for the parameter I , then we can compute the price curve ˜ p ( r ) with the estimated model parameters. By Proposition 2, this price curve is near-optimal when computed with true model parameters. Thus, if one can establish some kind of Lipschitz property, we may be able to extend the near-optimality to the estimated price curve. However, there are two main difficulties in implementing this idea: (1) A typical approach to estimate the model parameter is using the 'iterated least squares' (Keskin and Zeevi 2014), which charges a test price for certain rounds, and another different price for other certain rounds and then uses the observed demand to estimate the model parameter. This approach relies on a crucial assumption that the underlying demand is stationary and does not change over time. But in our setting, the underlying demand function is nonstationary and depends on the choices of the past prices through the reference price. Moreover, the underlying demand function may also be non-smooth as customers may have asymmetric reference effects, i.e., η + = η -. It is unclear how to use or modify this typical approach to account for the non-stationarity and reference effects to learn the model parameters ( a, b, η + , η -) , especially learn the reference effect parameters ( η + , η -) . (2) In addition, the near-optimal price curve ˜ p is highly non-stationary, and the price in this curve depends on its current time and the model parameters in a highly non-trivial way. It is unclear how this price curve changes w.r.t. the model parameter, and thus it is difficult to establish the Lipschitz property of seller's cumulative revenue function w.r.t. the model parameters.

Challenge two: inapplicable to apply restart mechanism. Under ARM , the optimal revenue V ∗ ( r t , t ) over the remaining rounds [ t, T ] can be quite sensitive to the starting reference price r t at time t . Indeed, one can see that when the reference price r t is not close to the reference price r ∗ t under the optimal pricing policy, then even though the seller can use an optimal pricing policy (i.e., p ∗ ( r t , t ) ) w.r.t. this reference price r t for the remaining rounds, the collected revenue V ∗ ( r t , t ) could be much smaller than the optimal revenue V ∗ ( r ∗ t , t ) : the difference of these two revenues could be as large as in the order of O ( t ( r ∗ t -r t ) ln T / t ) (see Lemma 3.3). Intuitively, in this problem, the seller needs to learn and follow not only the (near-)optimal price curve but also the reference price curve. One potential fix is one can periodically try to move the current reference price r t towards to a target reference price r t ′ that is close to the reference price r ∗ t ′ at time t ′ under optimal pricing policy. However, this fix does not work as (1) the seller does not actually know r ∗ t ′ (as she does not know the optimal pricing policy); (2) even if the seller knew r ∗ t ′ , one may need Ω( t | r t ′ -r t | )

rounds to reach to the reference price r t ′ from the reference price r t at time t , which could lead to linear regret when t is in the order of T . This challenge makes certain restart mechanisms, which is a common approach used to tackle the problem in learning with time-variant MDP (Besbes et al. 2014, Cheung et al. 2020), inapplicable under ARM .

## 4.2 Solution Ideas and the Proposed Learning Algorithm

In this section, we present our solution to the above challenges and our proposed algorithm details.

Reparameterizing the markdown price curve. The key observation in our algorithm design is that we can generalize and reparameterize the price curve ˜ p ( r ) defined in Proposition 2 in the following way: we can generalize the price curve ˜ p ( r, t 1 ) with an arbitrary starting time t 1 and starting reference price r at this time; moreover, instead of looking at the model parameter I , we can reparameterize the price curve such that it depends on the model parameter I only through a two-dimensional policy parameter θ ∈ Θ where Θ ⊆ R + × R + is a policy parameter space that will be defined later. In particular, we have the following generalized version of Proposition 2:

Lemma 4.1 (Generalized and reparameterized version of Proposition 2) . Given a policy parameter θ = ( C 1 , C 2 ) ∈ Θ , a starting reference price r at time t 1 ∈ [ T ] , we define price curve ˜ p ( r, t 1 , θ ) /defines ( p t ) t ∈ [ t 1 ,T ] as:

where p † and t † are some deterministic functions of ( θ, r, t 1 ) , and r t = t 1 r + ∑ t -1 s = t 1 p s t , t ∈ [ t 1 , T ] . Given an instance I = ( a, b, η + , η -) , let θ ∗ /defines ( C ∗ 1 , C ∗ 2 ) with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

/negationslash

Then the price curve ˜ p ( r, t 1 , θ ∗ ) is optimal when η + = η -, i.e., V ∗ ( r, t 1 ) = V ˜ p ( r,t 1 ,θ ∗ ) ( r, t 1 ) . Furthermore, the price curve ˜ p (¯ p, t 1 , θ ∗ ) is near-optimal when η + = η -, namely, for any starting reference price r , V ∗ ( r, t 1 ) -V ˜ p (¯ p,t 1 ,θ ∗ ) ( r, t 1 ) ≤ O ( t 1 ln T / t 1 ) .

One benefit of the above generalization is that it provides a way to obtain a price curve starting from any time t 1 that is near-optimal for the remaining rounds as long as t 1 is not very large (e.g., if t 1 is sublinear in T ). By reparameterization, we also note that even though the price curve ˜ p ( r, t 1 , θ ∗ ) heavily depends on the starting time t 1 , the universal constants C ∗ 1 and C ∗ 2 to characterize this curve are fixed irrespective of the starting time round t 1 and the starting reference price r at t 1 . This allows design of an algorithm that first does some price explorations for sublinear rounds in order to estimate these universal constants, and then implements the near-optimal price curve for the remaining rounds based on these estimated parameters.

One may recall that from Theorem 1 and Lemma 3.2, the optimal pricing policy is always a markdown pricing policy when the initial reference price equals to the price upper bound ¯ p . Thus, one may expect that the learning problem would become much easier if we assume r 1 = ¯ p at the very

beginning. However, we note that having r 1 = ¯ p does not simplify the structure of the optimal markdown and the previously mentioned two learning challenges still remain.

Remark 4. From Assumption 2.2 and Assumption 2.1, we know that the price upper bound ¯ p must satisfy b / 2 a &lt; ¯ p &lt; b / ( a + η + ) , which implies that one must have a &gt; η + . With this observation, we can deduce C ∗ 1 &lt; 1 / 4 , C ∗ 2 ≥ ¯ p / 2 . Thus, the policy parameter θ ∗ induced by all possible I ∈ I effectively satisfies that θ ∗ ∈ [0 , 1 / 4 ) × ( ¯ p / 2 , ∞ ) =: Θ .

Remark 5. As a sanity check, when η + = 0 and η -&gt; 0 , i.e., there is only loss reference effect, the price curve defined in Proposition 2 would become a fixed-price policy that keeps charging the price b / 2 a .

Learning the policy parameter via greedy price. The above observations motivate us to design an algorithm that could efficiently learn a good estimate for the policy parameter θ ∗ = ( C ∗ 1 , C ∗ 2 ) defined in Lemma 4.1. Interestingly, we observe that the greedy price for a certain range of reference prices r is also fully characterized by the policy parameter θ ∗ . In particular, let δ /defines ¯ p -max I ∈I b 2 a , we can show that given any reference price r ∈ (¯ p -δ, ¯ p ] , the greedy price p GR ( r ) that maximizes the single-shot revenue function Rev ( p, r ) subject to the constraint p ≤ r for this reference price r satisfies p GR ( r ) ∈ [0 , r ] , and moreover, it can be characterized as follows (see Lemma 5.1)

<!-- formula-not-decoded -->

The above observation implies that if we can learn two greedy prices p GR ( r a ) and p GR ( r b ) for two different reference prices r a and r b , respectively, then we can solve the following system of two linear equations to get the policy parameter θ ∗ = ( C ∗ 1 , C ∗ 2 ) :

<!-- formula-not-decoded -->

In addition, we observe that the single-shot revenue function Rev ( · , r ) over the price range [0 , r ] is strongly-concave and smooth. Thus, we can learn the greedy price by utilizing ideas from stochastic convex optimization techniques. Notably, in our problem, the seller can only learn a noisy demand value D t ( p t , r ) for the expected demand D ( p t , r ) at the chosen price p t . With this feedback structure, the problem of learning greedy price p GR ( r ) essentially reduces to stochastic convex optimization with bandit feedback.

## Algorithm 1 Dynamic pricing and learning under ARM

- 1: Input: Horizon T , starting reference price r 1 , price upper bound ¯ p .
- 2: Initialization: T 1 ; r a , r b .
- 3: /* Exploration phase 1: learning an estimate ˆ p GR ( r a ) for reference price r a */
- 4: Run LearnGreedy ( T 1 , r a , ¯ p ) to get ˆ p GR ( r a ) , and let t be the current time step.
- 5: /* Exploration phase 2: learning an estimate ˆ p GR ( r b ) for reference price r b */
- 6: Run LearnGreedy ( T 1 , r b , ¯ p ) to get ˆ p GR ( r b ) .
- 7: /* Exploitation phase */
- 8: Compute policy parameter estimate ˆ θ = ( ˆ C 1 , ˆ C 2 ) from (7) using ˆ p GR ( r a ) , ˆ p GR ( r b ) .
- 9: /* Suppose current time is T 2 */
- 10: Compute the price curve ˜ p (¯ p, T 2 , ˆ θ ) defined as in Lemma 4.1 with ˆ θ , and implement this price curve for remaining rounds.

The algorithm details. With the above challenges and solutions in mind, we design an 'explorethen-exploit'-style policy-parameter learning algorithm. At a high-level, our algorithm first learns estimates ˆ p GR ( r a ) , ˆ p GR ( r b ) of two greedy prices p GR ( r a ) , p GR ( r b ) for two carefully-set reference prices r a , r b , respectively. Then the algorithm uses ˆ p GR ( r a ) , ˆ p GR ( r b ) to construct an estimate ˆ θ for the policy parameter θ ∗ ∈ Θ by substituting these in (5). Finally, it implements the price curve defined in Lemma 4.1 with the estimated ˆ θ for the remaining rounds. We summarize our proposed algorithm in Algorithm 1.

- In both Exploration phase 1 and Exploration phase 2 : we design a stochastic convex optimization with bandit feedback algorithm to learn two greedy price estimates ˆ p GR ( r a ) (in Exploration phase 1 ) and ˆ p GR ( r b ) (in Exploration phase 2 ). We summarize the steps of learning greedy price in Algorithm 2, LearnGreedy ( T 1 , r, ¯ p ), which takes a time budget T 1 , a reference price r , and the price upper bound ¯ p as the input. In particular, we use T 1 to balance the estimation error | ˆ p GR ( r ) -p GR ( r ) | and the regret incurred in this algorithm; use reference price r to learn the greedy price p GR ( r ) ; and use the price upper bound ¯ p to control the learning rate. One difficulty here is while running the Algorithm 2, the reference price will not remain fixed, but evolve according to ARM . Thus, to learn the greedy price for a particular reference price r , before each learning round t in Algorithm 2, we use a subroutine ResetRef ( t, r t , r b ) (described as Algorithm 4 in Appendix B) to reset the current reference price to the reference price r . 89
- In Exploitation Phase : we first construct the estimate ˆ θ = ( ˆ C 1 , ˆ C 2 ) using (5) with the estimates ˆ p GR ( r a ) and ˆ p GR ( r b ) . Suppose T 2 is the time that the algorithm enters in this phase, we then compute the price curve ˜ p (¯ p, T 2 , ˆ θ ) (defined in Lemma 4.1) with the policy parameter estimate ˆ θ , and starting time T 2 (note that here we compute the price curve with starting reference price ¯ p , this price curve is near-optimal, by Lemma 4.1, for any ( η + , η -) if we had ˆ θ = θ ∗ ). Then we implement this price curve for remaining rounds.

```
Algorithm 2 LearnGreedy : Bandit stochastic convex optimization for learning greedy price 1: Input: Budget T 1 , reference price r , price upper bound ¯ p , current time round t . 2: Initialization: Initialize p 1 = p GR 1 ∈ [0 , ¯ p ] arbitrarily; let d = 1 2 ( r -max I ∈I b 2 a ) . 3: Initialization: Initialize s ← 0 . 4: /* Counter t records all time rounds */ 5: /* Counter s records learning rounds */ 6: while t < T and s < T 1 do 7: t ← t + ResetRef ( t, r t , r ) + 1 . /* Use ResetRef ( t, r t , r ) to reset reference price to r . */ 8: s ← s +1 . 9: Pick κ ∈ {-1 , 1 } uniformly at random. 10: Set the price p t ← p GR s + κd , and observe the realized demand D t ( p t , r ) . 11: Let p GR s +1 ← PROJ [ d,r -d ] ( p GR s + p t D t ( p t ,r ) 2¯ pds κ ) . 12: end while 13: Return ˆ p GR ( r ) = ∑ T 1 s =1 p GR s T 1 , and the current time t .
```

8 The subroutine ResetRef ( · , · , · ) is designed to set the prices within the price range [0 , ¯ p ] .

9 Notice that the seller knows the initial reference price r 1 , thus the seller knows when to stop when she reaches to r a or r b .

## 5 Regret Analysis

In this section, we derive an upper bound the regret of Algorithm 1 for any problem instance that has parameter I = ( a, b, η + , η -) in I . Recall that I ⊆ [0 , 1] 4 was defined as the set of all feasible parameter vectors satisfying Assumption 2.2.

<!-- formula-not-decoded -->

Theorem 2. With T 1 = ˜ Θ ( ¯ p 2 √ T √ 1+¯ p ) , for any instance I ∈ I , any starting reference price r ∈ [0 , ¯ p ] at t = 1 , Algorithm 1 has expected regret

As a comparison, for the setting where reference-price dynamics follow ESM , den Boer and Keskin (2022), which also focuses on linear base demand model, proposed a learning algorithm with a regret upper bound of ˜ O (¯ p 6 √ T/ (1 -ζ ) 2 ) for loss-averse customers 10 . Here, ζ is the constant averaging factor as described in Remark 1. Note that under ARM , the averaging factor ζ t is ζ t = t t +1 so that 1 / (1 -ζ t ) = t +1 . Thus naive application of their algorithm in our setting with ARM would incur a linear regret. This again highlights the previously mentioned observation that our ARM is fundamentally different from ESM .

Also note that our regret upper bound in Theorem 2 depends polynomially on the price upper bound ¯ p . One may wonder if we can first consider a setting with price range [0 , 1] (instead of [0 , ¯ p ] ), and then recover the general regret upper bound (for arbitrary price upper bound ¯ p ) simply by scaling. In doing so, one may hope to obtain a final regret bound that has quadratic dependency of ¯ p (since the instant revenue quadratically depends on the price). However, we would like to point out that the price upper bound ¯ p in our setting appears beyond just scaling, it also regulates the range of model parameters. Moreover, the instant revenue quadratically depending on price does not imply that the optimal total revenue also quadratically depends on ¯ p due to the nature of the nonstationarity of our demand function. When there are no reference effects, Keskin and Zeevi (2014) have shown that no learning algorithm can have regret growing slower than Ω( √ T ) , even when restricted to instances satisfying Assumption 2.2. We build upon their proof to derive the following more detailed lower bound that shows that for any learning algorithm, the dependence of regret on price upper bound ¯ p is unavoidable even when there are no reference effects.

Proposition 4. Given a price upper bound ¯ p ≥ 1 , consider the following problem instances I ′ : let η + = η -≡ 0 , b ≡ 1 , and a ∈ [ 3 4¯ p , 1 ¯ p ] . Then clearly all instances in I ′ satisfy Assumption 2.2. And the expected regret of any algorithm π ALG satisfies that sup I ∈I ′ REG π ALG [ T, I ] ≥ ¯ p √ T -1 3 √ 1+36 π 2 . 11

The proof of above Proposition 4 is in Appendix C.1. We below provide a proof outline of Theorem 2. All missing proofs in the following subsection are in Appendix C.2 to Appendix C.4.

## 5.1 Proof of Theorem 2

In this subsection, we provide an outline of different steps involved in proving Theorem 2.

## Step 1: Bounding the estimation error of the policy parameter.

10 The dependence of regret bound on ¯ p has not been explicitly mentioned in the results of this paper. We have derived it here to the best of our understanding of their proof.

11 The lower bound we derive here scales linearly w.r.t. price upper bound ¯ p , it is an interesting open question to study whether one can tighten this gap between the linear dependency in lower bound and the cubic-ish dependency of ¯ p in our upper bound.

In this step, we bound the estimation error ‖ ˆ θ -θ ∗ ‖ of the policy parameter estimate ˆ θ , which is computed using the outputs of Algorithm 2 for two reference prices r a , r b . Given any reference price r ∈ [0 , ¯ p ] , we consider the following greedy price p GR ( r ) , which maximizes the single round revenue function among all prices in the range [0 , r ] :

<!-- formula-not-decoded -->

We first show that when the reference price r satisfies r ∈ (¯ p -δ, ¯ p ] , we always have p GR ( r ) ∈ [0 , r ] , and it can be fully characterized by the policy parameter θ ∗ .

Lemma 5.1. Fix any reference price r ∈ (¯ p -δ, ¯ p ] , then greedy price p GR ( r ) = C ∗ 1 r + C ∗ 2 ∈ (0 , r ) .

With the above Lemma 5.1, we can see that on fixing two different reference prices r a , r b ∈ (¯ p -δ, ¯ p ] , the greedy prices p GR ( r a ) , p GR ( r b ) satisfy Equation (5). And therefore, given the estimates ˆ p GR ( r a ) , ˆ p GR ( r b ) to the greedy price p GR ( r a ) and p GR ( r b ) , we can use Equation (5) to compute a policy parameter estimate ˆ θ = ( ˆ C 1 , ˆ C 2 ) as follows

<!-- formula-not-decoded -->

We then have the following estimation error on the policy parameter estimate ˆ θ :

Proposition 5. Given two reference prices r a , r b ∈ (¯ p -δ, ¯ p ] where r a &lt; r b , let ˆ θ = ( ˆ C 1 , ˆ C 2 ) be the estimated parameter obtained from Line 12 in Algorithm 1, then the following holds with probability at least 1 -2 δ for δ ∈ (0 , 1 / e ) :

The key step to prove Proposition 5 is the following characterization on how the estimated greedy price ˆ p GR ( r ) is close to the true greedy price p GR ( r ) .

<!-- formula-not-decoded -->

Lemma 5.2. Let δ ∈ (0 , 1 / e ) and let r be the reference price to the input of Algorithm 2. For any T 1 ≥ 4 , we have the following holds with probability at least 1 -δ :

where d is the parameter defined in Line 2 in Algorithm 2.

<!-- formula-not-decoded -->

Our Algorithm 2 is built on Shamir (2013), which proposed an algorithm for stochastic stronglyconvex optimization with bandit feedback with assuming that the distance between the maximizer and the domain boundary is at least d . We satisfy this assumption by carefully choosing the distance parameter d such that the greedy price p GR ( r ) is always in the range [ d, r -d ] (see Lemma 5.1).

Intuitively, Algorithm 2 utilizes a well-known 1 -point gradient estimate technique, to get an unbiased estimate of the gradient at the chosen price by randomly querying for a (noisy) value of the function around it. For general strongly-concave-and-smooth function (i.e., not necessarily being quadratic), the attainable estimation error of the maximizer is Θ( T -1 / 4 1 ) (Shamir 2013,

Agarwal et al. 2010, Jamieson et al. 2012). Instead, by leveraging the quadratic structure of the revenue function Rev ( · , r ) , the gradient estimates we construct have much smaller variance compared to the one for general function, which allows us to obtain an improved estimation error O ( T -1 / 2 1 ) .

With the estimation error bound in Lemma 5.2, Proposition 5 follows immediately by the triangle inequality. The missing steps to prove Lemma 5.2 are provided in Appendix C.2.

## Step 2: Bounding the total resetting regret.

In this step, we bound the total regret incurred from running the subroutine ResetRef ( · , · , ) (i.e., Algorithm 4) before each of the ( 2 T 1 ) learning round in Algorithm 2. In particular, we upper bound these regrets by upper bounding the total number of rounds used in invoking ResetRef ( · , · , ) . The main results in this subsection are summarized in the following lemma:

Lemma 5.3. In all Exploration phase 1 and Exploration phase 2 of Algorithm 2, the number of rounds used in running ResetRef ( · , · , ) is O ( T 1 ) .

## Step 3: Bounding the Lipschitz error in policy parameter space.

In this step, we establish a Lipschitz property of the revenue function V ˜ p ( r,t 1 ,θ ) ( r, t 1 ) with respect to the policy parameter θ for the case of symmetric reference effects, i.e., when η + = η -≡ η . Since in this case V ˜ p ( r,t 1 ,θ ∗ ) ( r, t 1 ) = V ∗ ( r, t ) , this allows us to bound the gap V ∗ ( r, t 1 ) -V ˜ p ( r,t 1 ,θ ) ( r, t 1 ) in terms of policy parameter estimation error ‖ θ ∗ -θ ‖ . Later, we describe how we use this result in the next step to establish regret for arbitrary η + , η -.

Proposition 6 (Bounding the Lipschitz error) . Assume η + = η -≡ η . Fix a starting time t 1 ∈ [ T ] , a starting reference price r at time t 1 . Then, the following holds for all θ ∈ Θ ,

Proof Sketch of Proposition 6. A typical way to establish the Lipschitz property of a function is to bound the gradient of this function, namely ∇ θ V ˜ p ( r,t 1 ,θ ) ( r, t 1 ) . In our problem, the dependence of the total revenue function V ˜ p ( r,t 1 ,θ ) ( r, t 1 ) over the policy parameter θ is through the corresponding price curve ˜ p ( r, t 1 , θ ) (defined in Lemma 4.1), and this price curve depends on the policy parameter in a highly non-trivial way. Thus, it is not clear how to directly compute and bound the gradient ∇ θ V ˜ p ( r,t 1 ,θ ) ( r, t 1 ) . Instead, we use a two-step approach to establish the Lipschitz property of function V ˜ p ( r,t 1 ,θ ) ( r, t 1 ) .

<!-- formula-not-decoded -->

- Bounding the revenue gap ∣ ∣ ∣ V p ( r, t 1 ) -V p ′ ( r, t 1 ) ∣ ∣ ∣ via the distance on the price sequences ‖ p -p ′ ‖ : We first show that when the optimal price curve ˜ p ( r, t 1 , θ ∗ ) (i.e., the price curve defined in Lemma 4.1 computed with true policy parameter θ ∗ ) is a strict markdown price curve, then the seller's total revenue function V p ( r, t 1 ) is strongly concave over the price sequence p (see below Lemma 5.4). We prove this result by bounding the eigenvalues of the Hessian matrix ∇ 2 p V p ( r, t 1 ) . We note that this result does not require that the price sequence p is the price curve computed in Lemma 4.1.

Lemma 5.4 (Strong concavity of function V p ) . Fix a starting time t 1 ∈ [ T ] and starting reference price r . If the price curve ˜ p ( r, t 1 , θ ∗ ) is a strict markdown price curve (i.e., t † = t 1 ), then there exists two positive, finite constants c 1 and c 2 where c 1 &gt; c 2 , such that the eigenvalue λ V of the Hessian matrix ∇ 2 p V p ( r, t 1 ) satisfies λ V ∈ [ -2( a + η ) c 1 , -2( a + η ) c 2 ] .

- Bounding the price curve distance ‖ ˜ p ( r, t 1 , θ 1 ) -˜ p ( r, t 1 , θ 2 ) ‖ via the parameter distance ‖ θ 1 -θ 2 ‖ : We next establish a 'Lipschitz property' for the price curve ˜ p ( r, t 1 , θ ) in terms of the differences between the policy parameters (see Lemma 5.5).

Lemma 5.5 (Lipschitz error on the price curve) . Fix a starting time t 1 ∈ [ T ] and starting reference price r . If price curves ˜ p ( r, t 1 , θ 1 ) and ˜ p ( r, t 1 , θ 2 ) are both strict markdown price curves, then we have ‖ ˜ p ( r, t 1 , θ 1 ) -˜ p ( r, t 1 , θ 2 ) ‖ ≤ O ( ¯ p ‖ θ 1 -θ 2 ‖ √ T -t 1 ln T / t 1 ) .

Combining above two results can prove Proposition 6 when the both price curves ˜ p ( r, t 1 , θ ∗ ) and ˜ p ( r, t 1 , θ ) are strictly markdown. On the other hand, as we can see from Lemma 4.1, the curve ˜ p ( r, t 1 , θ ∗ ) (or ˜ p ( r, t 1 , θ ) ) is not necessarily a strict markdown curve, as it may keep charging the same price ¯ p for initial certain rounds. For this case, notice there must exist a time round t † ∈ [ t 1 , T ] such that the partial price sequence ˜ p ( r t † , t † , θ ∗ ) = ( p t ) t ≥ t † in ˜ p ( r, t 1 , θ ∗ ) must still be a strict markdown price curve (and similarly, it holds true for ˜ p ( r, t 1 , θ ) ). Thus, in the analysis, we also bound the gap between the time rounds at which the two price curves start to strictly markdown their prices. Notably, we show that the revenue gap incurred due to such time round gap is negligible compared to the revenue gap incurred due to the policy parameter distance ‖ θ -θ ∗ ‖ .

Step 4: Putting it all together. We now put all pieces together to show the regret bound of Algorithm 1.

Proof of Theorem 2. Let ( r ∗ t ) t ∈ [ T ] be the resulting reference price sequence under the optimal policy p ∗ . Let π ALG be the pricing policy implemented by Algorithm 1, and let p be the realized price sequence from algorithm π ALG . Suppose T 2 ∈ [ T ] is the first time round (notice that this is a random variable) that the Algorithm 1 enters in the Exploitation phase. According to our algorithm design, we know that the pricing decisions over the time window [ T 2 , T ] is the price curve ˜ p (¯ p, T 2 , ˆ θ ) with the starting reference price ¯ p and parameter ˆ θ . Let ¯ R /defines max p,r Rev ( p, r ) . For any instance I ∈ I , the total regrets can be decomposed and bounded as follows:

<!-- formula-not-decoded -->

where inequality (a) holds true by observing that the regret incurred in first T 2 rounds is at most T 2 ¯ R ; inequality (b) holds true by lemma 3.3 where r ∗ T 2 ≤ ¯ p ; inequality (c) holds true by Lemma A.1 which bounds the revenue gap when a same price curve is applied with two different starting reference prices.

We now observe that by Lemma 3.2, the optimal pricing policy to achieve V ∗ (¯ p, T 2 ) is a markdown pricing policy, thus, V ∗ (¯ p, T 2 ) exactly equals to the optimal revenue V ∗ (¯ p, T 2 | ( η + , η + )) when the customers have symmetric reference effects η + = η -with starting reference price ¯ p . Here, we slightly abuse the notation and let V ∗ ( r, t | ( η + , η + )) feature that this cumulative revenue is computed with reference effect parameter ( η + , η + ) . Moreover, the cumulative revenue V ˜ p (¯ p,T 2 , ˆ θ ) (¯ p, T 2 ) is obtained by applying the markdown price curve ˜ p (¯ p, T 2 , ˆ θ ) with the starting reference price ¯ p at time T 2 . Thus, it only depends parameter η + , and V ˜ p (¯ p,T 2 , ˆ θ ) (¯ p, T 2 ) also equals to the cumulative revenue

V ˜ p (¯ p,T 2 , ˆ θ ) (¯ p, T 2 | ( η + , η + )) when customers have η + = η -with the starting reference price ¯ p . With this observation, we can further bound the regret as follows:

<!-- formula-not-decoded -->

where inequality (a) holds by the fact that we are able to use Proposition 6 to bound the revenue gap via the policy parameter estimation error that we obtain in Proposition 5 for symmetric reference effects η + = η -; inequality (b) holds true by Lemma 5.3 where we have T 2 = O ( T 1 ) , and by noticing that ¯ R ≤ ¯ p (1 + ¯ p ) , and we optimize T 1 = Θ ( ¯ p 2 √ T (log log T +1)log T 1+¯ p ) = ˜ Θ ( ¯ p 2 √ T √ 1+¯ p ) and δ = 1 / T to get the desired regret.

## 6 Conclusions and Future Directions

In this work, we study dynamic pricing problem where customer response to the current price is impacted by a reference price, which is formed by following an averaging-reference mechanism ( ARM ). We demonstrate that a fixed-price policy is highly suboptimal in this setting, which sets it distinctively apart from the well-studied ESM dynamics for reference price effects. We also establish the (near-)optimality of markdown pricing in ARM models. We show that under ARM with gainseeking customers, markdown pricing is optimal, and for loss-averse customers, markdown pricing is near-optimal in the sense that the revenue achieved is within O (log( T )) of the optimal revenue.

Investigating this problem further for a linear base demand model, we provide a detailed structural characterizations of a near-optimal markdown pricing policy for both gain-seeking and loss-averse customers, along with an efficient algorithm for computing such policies. We then study the dynamic pricing and learning problem, where the demand model parameters are apriori unknown. We provide an efficient learning algorithm with an asymptotically optimal revenue performance.

Below we mention a few possible avenues for future research, from the perspective of algorithm design and customer behavior modeling, respectively.

From algorithm design perspective. What is the general characterization of the optimal pricing policy when the underlying base demand model is beyond linear? We notice that for general base demand model, the condition (4) for optimal prices that we derive here still holds. It would be interesting to explore further what additional structural characterizations we can infer from this condition. Meanwhile, on the algorithmic side, the learning part of our work also considers a linear base demand model. Though it is already interesting and challenging enough to develop efficient learning algorithm for this case, it would be interesting to generalize our idea to more general or non-parametric demand models. In addition, our learning algorithm is an explore-first-then-exploit

type algorithm. Though this simple algorithm can already guarantee us a regret bound that has optimal dependency on the sales horizon, it would be interesting to explore whether a learning algorithm with 'adaptive exploration' (e.g., UCB-type algorithm, Thompson Sampling) can further improve the bound, e.g., tighten the regret gap on the ¯ p dependency.

From customer behavior modeling perspective. Almost all the reference price models (including ours) in current literature assume that the reference price updates depend only on the offered price (and its offered time), and not the customer demand response to those prices. These mechanisms could lead to an (unsatisfied) pricing strategy that the seller can set a single very large price (especially when the price upper bound is very high) to increase the reference price and lead the customer to purchase more. One potential approach to address this is to consider models where reference price update depends on the sales that happen at the offered price.

Secondly, in our current ARM model (3), the averaging factor is ζ t = 1 / t , while in ESM , we have ζ t ≡ ζ for some constant ζ . An interesting direction is to consider an intermediate setting where the averaging factor ζ t = 1 / t α is parameterized by some rate parameter α ≥ 0 that interpolates between the ARM ( α = 1 ) and ESM ( α = 0 ). One can explore the same set of questions asked in this work for this more general setting. For example, how does the fixed-price policy perform? One may conjecture that the total revenue from the fixed-price policy may approach the optimal total revenue gracefully as α goes to 0 .

Lastly, we have implicitly considered a setting where customers are myopic, i.e., they are not forwardlooking and not strategically timing their purchasing decisions. Yet the markdown nature of the (near-)optimal pricing policy that we characterize may incentivize the customers to strategically decide when to enter the market and make the purchase decision. It thus would be interesting to explore the design of optimal pricing policies in the presence of long-term reference effects and strategic customer behavior.

## Acknowledgement

The authors would like to thank the reviewers of EC'24 for helpful comments. This work was supported in part by NSF 2147361, NSF 2040971 and NSF CAREER 1846792.

## References

- Agarwal, A., Dekel, O., and Xiao, L. (2010). Optimal algorithms for online convex optimization with multipoint bandit feedback. In Colt , pages 28-40. Citeseer.
- Agrawal, S., Feng, Y., and Tang, W. (2023). Dynamic pricing and learning with bayesian persuasion. arXiv preprint arXiv:2304.14385 .
- Agrawal, S. and Jia, R. (2017). Optimistic posterior sampling for reinforcement learning: worst-case regret bounds. Advances in Neural Information Processing Systems , 30.
- Agrawal, S., Yin, S., and Zeevi, A. (2021). Dynamic pricing and learning under the bass model. In Proceedings of the 22nd ACM Conference on Economics and Computation , pages 2-3.
- Araman, V. F. and Caldentey, R. (2009). Dynamic pricing for nonperishable products with demand learning. Operations research , 57(5):1169-1188.
- Arslan, H. and Kachani, S. (2010). Dynamic pricing under consumer reference-price effects. Wiley encyclopedia of operations research and management science .
- Auer, P., Jaksch, T., and Ortner, R. (2008). Near-optimal regret bounds for reinforcement learning. Advances in neural information processing systems , 21.

- Besbes, O., Gur, Y., and Zeevi, A. (2014). Stochastic multi-armed-bandit problem with non-stationary rewards. Advances in neural information processing systems , 27.
- Besbes, O. and Zeevi, A. (2009). Dynamic pricing without knowing the demand function: Risk bounds and near-optimal algorithms. Operations research , 57(6):1407-1420.
- Besbes, O. and Zeevi, A. (2011). On the minimax complexity of pricing in a changing environment. Operations research , 59(1):66-79.
- Besbes, O. and Zeevi, A. (2015). On the (surprising) sufficiency of linear models for dynamic pricing with demand learning. Management Science , 61(4):723-739.
- Birge, J. R., Chen, H., and Keskin, N. B. (2023). Markdown policies for demand learning with forwardlooking customers. Operations Research .
- Broder, J. and Rusmevichientong, P. (2012). Dynamic pricing under a general parametric choice model. Operations Research , 60(4):965-980.
- Cao, P., Zhao, N., and Wu, J. (2019). Dynamic pricing with bayesian demand learning and reference price effect. European Journal of Operational Research , 279(2):540-556.
- Chen, B., Simchi-Levi, D., Wang, Y., and Zhou, Y. (2022). Dynamic pricing and inventory control with fixed ordering cost and incomplete demand information. Management Science , 68(8):5684-5703.
- Chen, B., Wang, Y., and Zhou, Y. (2024). Optimal policies for dynamic pricing and inventory control with nonparametric censored demands. Management Science , 70(5):3362-3380.
- Chen, X., Hu, P., and Hu, Z. (2017). Efficient algorithms for the dynamic pricing problem with reference price effect. Management Science , 63(12):4389-4408.
- Chen, Y., Wen, Z., and Xie, Y. (2019). Dynamic pricing in an evolving and unknown marketplace. Available at SSRN 3382957 .
- Cheung, W. C., Simchi-Levi, D., and Zhu, R. (2020). Reinforcement learning for non-stationary markov decision processes: The blessing of (more) optimism. In International Conference on Machine Learning , pages 1843-1854. PMLR.
- Den Boer, A. V. (2015a). Dynamic pricing and learning: historical origins, current research, and new directions. Surveys in operations research and management science , 20(1):1-18.
- Den Boer, A. V. (2015b). Tracking the market: Dynamic pricing and learning in a changing environment. European journal of operational research , 247(3):914-927.
- den Boer, A. V. and Keskin, N. B. (2022). Dynamic pricing with demand learning and reference effects. Management Science , 68(10):7112-7130.
- den Boer, A. V. and Zwart, B. (2014). Simultaneously learning and optimizing using controlled variance pricing. Management science , 60(3):770-783.
- Fei, Y., Yang, Z., Wang, Z., and Xie, Q. (2020). Dynamic regret of policy optimization in non-stationary environments. Advances in Neural Information Processing Systems , 33:6743-6754.
- Fibich, G., Gavious, A., and Lowengart, O. (2003). Explicit solutions of optimization models and differential games with nonsmooth (asymmetric) reference-price effects. Operations Research , 51(5):721-734.
- Gill, R. D. and Levit, B. Y. (1995). Applications of the van trees inequality: a bayesian cramér-rao bound. Bernoulli , pages 59-79.
- Goldenshluger, A. and Zeevi, A. (2009). Woodroofe's one-armed bandit problem revisited. The Annals of Applied Probability , pages 1603-1633.
- Golrezaei, N., Jaillet, P., and Liang, J. C. N. (2020). No-regret learning in price competitions under consumer reference effects. Advances in Neural Information Processing Systems , 33:21416-21427.
- Google (2023). Transforming specialty retail with ai. pages Online; accessed 26-Nov-2023.
- Hu, Z., Chen, X., and Hu, P. (2016). Dynamic pricing with gain-seeking reference price effects. Operations Research , 64(1):150-157.
- IBM (2023). Ibm markdown optimization. pages Online; accessed 26-Nov-2023.
- Jamieson, K. G., Nowak, R., and Recht, B. (2012). Query complexity of derivative-free optimization. Advances in Neural Information Processing Systems , 25.

- Ji, S., Yang, Y., and Shi, C. (2023). Online learning and pricing for multiple products with reference price effects. Available at SSRN 4349904 .
- Jia, S., Li, A., and Ravi, R. (2021). Markdown pricing under unknown demand. Available at SSRN 3861379 .
- Jia, S., Li, A., and Ravi, R. (2022). Dynamic pricing with monotonicity constraint under unknown parametric demand model. Advances in Neural Information Processing Systems , 35:19179-19188.
- Keskin, N. B. and Zeevi, A. (2014). Dynamic pricing with an unknown demand model: Asymptotically optimal semi-myopic policies. Operations research , 62(5):1142-1167.
- Keskin, N. B. and Zeevi, A. (2017). Chasing demand: Learning and earning in a changing environment. Mathematics of Operations Research , 42(2):277-307.
- Kleinberg, R. and Leighton, T. (2003). The value of knowing a demand curve: Bounds on regret for online posted-price auctions. In 44th Annual IEEE Symposium on Foundations of Computer Science, 2003. Proceedings. , pages 594-605. IEEE.
- Lattin, J. M. and Bucklin, R. E. (1989). Reference effects of price and promotion on brand choice behavior. Journal of Marketing research , 26(3):299-310.
- Lazear, E. P. (1986). Retail pricing and clearance sales. The American Economic Review , 76(1):14-32.
- Mazumdar, T., Raj, S. P., and Sinha, I. (2005). Reference price research: Review and propositions. Journal of marketing , 69(4):84-102.
- Nasiry, J. and Popescu, I. (2011). Dynamic pricing with loss-averse consumers and peak-end anchoring. Operations research , 59(6):1361-1368.
- Popescu, I. and Wu, Y. (2007). Dynamic pricing strategies with reference effects. Operations research , 55(3):413-429.
- Rajendran, K. N. and Tellis, G. J. (1994). Contextual and temporal components of reference price. Journal of marketing , 58(1):22-34.
- Rakhlin, A., Shamir, O., and Sridharan, K. (2012). Making gradient descent optimal for strongly convex stochastic optimization. In Proceedings of the 29th International Coference on International Conference on Machine Learning , pages 1571-1578.
- Shamir, O. (2013). On the complexity of bandit and derivative-free stochastic convex optimization. In Conference on Learning Theory , pages 3-24. PMLR.
- Wu, S., Liu, Q., and Zhang, R. Q. (2015). The reference effects on a retailer's dynamic pricing and inventory strategies with strategic consumers. Operations Research , 63(6):1320-1335.

## A Missing Proofs in Section 3

## A.1 Missing Proofs of Section 3.1

Proposition 1. There exists an ARM problem instance with linear base demand model, i.e., H ( p ) = b -ap and loss-neutral customers (i.e., η + = η -), and an initial reference price r 1 such that for any fixed-price policy p , we have V ∗ ( r 1 ) -V p ( r 1 ) = Ω( T ) .

Proof of Proposition 1. We consider following problem instance: η -= η + ≡ η , r 1 = 0 , and the base demand is a linear demand H ( p ) = b -ap . Let p ( p ) = ( p, . . . , p ) denote a fixed-price policy that keeps charging the price p throughout the sales horizon. The total revenue under the fixed-price policy is p :

<!-- formula-not-decoded -->

Let p ∗ , fixed = arg max p ∈ [0 , ¯ p ] V p ( r 1 ) be the optimal fixed-price, and V ∗ , fixed ( r 1 ) be its corresponding total revenue, then we have

<!-- formula-not-decoded -->

where inequality (a) is due to r 1 = 0 . Given a time round T 1 ∈ [ T ] , and let α = T T 1 . We now consider the following non-fixed-price policy p = ( p t ) p t ∈ [ T ] where p t = p u 1 { t ≤ αT 1 } + p d 1 { t ≥ αT 1 +1 } , where p u , p d are determined later. Under the policy p , the total revenue is

<!-- formula-not-decoded -->

Given the value of α , we choose p u and p d as follows

<!-- formula-not-decoded -->

Essentially, we choose the above p u , p d such that it maximizes the total revenue V p ( r 1 ) under the value α . With the above value of p u , p d , let A ( α ) /defines αp u ( b -ap u ) + (1 -α ) p d ( b -ap d ) + ηp d α ( p u -p d ) ln 1 α . With the above definitions, we have

<!-- formula-not-decoded -->

Notice that it is easy to find values for a, b, α, η such that we have A ( α ) -b 2 4 a ≥ C for some positive constant C &gt; 0 . For example, let b = 2 , a = 1 , η = 0 . 5 , α = 0 . 3 , we have A ( α ) -b 2 4 a = 0 . 0318 . Moreover, under the above choices of a, b, η, α , we also have p u = 1 . 2787 and p d = 0 . 926 , and consider ¯ p = b a + η = 1 . 3333 . This implies that the above defined pricing policy p is indeed a feasible pricing policy. Thus, we can conclude that V ∗ ( r 1 ) -V ∗ , fixed ( r 1 ) = Ω( T ) .

## A.2 Missing Proofs of Section 3.2

Lemma 3.1. Fix any starting time t 1 ∈ [ T ] and a starting reference price r t 1 = r , when η + ≥ η -, the optimal pricing policy starting from time t 1 is a markdown pricing policy.

Proof of Lemma 3.1. Consider a pricing policy p = ( p t ) t ∈ [ t 1 ,T ] where the reference price at time t is r t and p t &lt; p t +1 . Now consider a new pricing policy p ′ = ( p ′ s ) s ∈ [ t 1 ,T ] where: (1) p ′ t ← p t +1 , p ′ t +1 ← p t ; (2) p ′ s ← p s for all s ∈ [ t 1 , T ] \{ t, t +1 } . Let r ′ t , r ′ t +1 be the induced reference price at rounds t, t +1 under policy p ′ , respectively. By definition, we have r ′ t = r t , moreover, we also have the following observation

<!-- formula-not-decoded -->

Notice that the revenue difference between the policy p ′ and policy p is

<!-- formula-not-decoded -->

To analyze whether ∆ ≥ 0 , we below consider two cases

1. When r t ≥ p t . Under this case, we know that

Thus, we have

<!-- formula-not-decoded -->

We further consider following two sub-cases

- (a) When p t +1 ≥ r t . Under this sub-case, we have

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (b) When p t +1 &lt; r t . Under this sub-case, we have

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

Now suppose we have r t +1 ≤ p t +1 , then

<!-- formula-not-decoded -->

where in inequality (a) we use the fact that p t &lt; p t +1 .

Now suppose we have r t +1 &gt; p t +1 , then we have

<!-- formula-not-decoded -->

Thus, under this sub-case, we also have ∆ ≥ 0 .

2. When r t &lt; p t . Under this case, we have r t &lt; p t ≤ p t +1 , and moreover

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

Now suppose we have r ′ t +1 ≤ p t , then

<!-- formula-not-decoded -->

Now suppose we have r ′ t +1 &gt; p t , then

<!-- formula-not-decoded -->

where in inequality (a), we use the fact that η + ≥ η -. Thus, under this case, we always have ∆ ≥ 0

Putting all pieces together, we can prove the statement.

Lemma 3.3 (Optimal revenue gap w.r.t. different starting reference price) . Fix any starting time t 1 ∈ [ T ] , the optimal revenue function V ∗ ( r, t 1 ) is increasing w.r.t. reference price r . Moreover, for any ( η + , η -) , we have V ∗ ( r ′ , t 1 ) -V ∗ ( r, t 1 ) ≤ O (¯ pt 1 ( r ′ -r )( η -+ η + ) ln T / t 1 ) for any r ′ ≥ r .

To prove Lemma 3.3, we first show the following lemma which bound the revenue gap when implementing a same pricing policy under different starting reference prices.

<!-- formula-not-decoded -->

Proof of Lemma A.1. Fix any pricing policy p = ( p t ) t ∈ [ t 1 ,T ] , let ( r t ) t ∈ [ t 1 ,T ] (resp. ( r ′ t ) t ∈ [ t 1 ,T ] ) be the resulting reference price path under the starting reference price r (resp. r ′ ). Then by definition, for any t ∈ [ t 1 , T ] , we have

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

where the last inequality is due to the fact that r ′ t ≥ r t for all t ∈ [ t 1 , T ] . Moreover,

<!-- formula-not-decoded -->

We now prove Lemma 3.3.

Proof of Lemma 3.3. Fix any pricing policy p = ( p t ) t ∈ [ t 1 ,T ] , let ( r t ) t ∈ [ t 1 ,T ] (resp. ( r ′ t ) t ∈ [ t 1 ,T ] ) be the resulting reference price path under the starting reference price r (resp. r ′ ).

Let p ∗ ( r, t 1 ) (resp. p ∗ ( r ′ , t 1 ) ) be the optimal pricing policy under the starting reference price r (resp. r ′ ). Then,

<!-- formula-not-decoded -->

where last inequality is by Lemma A.1. Moreover,

<!-- formula-not-decoded -->

where last equality is by Lemma A.1, thus completing the proof.

Lemma 3.2. Fix any starting time t 1 ∈ [ T ] and a starting reference price r t 1 = ¯ p , if η + &lt; η -, then the optimal pricing policy starting from time t 1 is a markdown pricing policy.

Proof of Lemma 3.2. We prove by contradiction. Let us fix r t 1 = ¯ p . Suppose under a pricing policy p = ( p t ) t ∈ [ t 1 ,T ] , there exists a time step k ∈ [ t 1 , T ] such that the resulting reference price r k satisfies r k &lt; p t 1 . Then it implies that there exists a time step s &lt; k such that (i) p s &lt; p k ; (ii) p t ≥ p k for all t &lt; s . We now define a new pricing policy p = ( p ′ t ) t ∈ [ t 1 ,T ] such that it satisfies (1) p ′ s = p k , p ′ k = p s ; (2) p ′ t = p t , ∀ t ∈ [ t 1 , T ] \ { s, k } . Then we consider

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where inequality (a) is due to the fact that for any t ∈ [ s +1 , k -1] , r ′ t -r t = p k -p s t &gt; 0 , and p ′ t = p t , thus we have Rev ( p ′ t , r ′ t ) ≥ Rev ( p t , r t ) , and equality (b) is due to r s ≥ p k &gt; p s and r k &lt; p k . Let A /defines ∑ k -1 s = s +1 p t . Then we notice that

<!-- formula-not-decoded -->

Let ∆ denote the right-hand-side of the above equation (b). We below consider two possible cases:

1. When r ′ k ≥ p s . Under this case, we know that

<!-- formula-not-decoded -->

2. When r ′ k &lt; p s . Under this case, we know that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

When p k ( r s -p k ) -p s ( r s -p s ) ≥ 0 , we have where the last inequality is due to p k &gt; p s , r k &lt; r ′ k .

<!-- formula-not-decoded -->

Putting all pieces together, we can prove the statement.

Theorem 1 (Near optimality of markdown pricing policy) . Fix any starting reference price r 1 ∈ [0 , ¯ p ] at time t = 1 ,

- 1a when η + ≥ η -, i.e., when customers are gain-seeking, optimal pricing policy is a markdown policy;
- 1b when η + &lt; η -, i.e., when customers are loss-averse, there exists a markdown policy p that is near-optimal, namely, V ∗ ( r 1 ) -V p ( r 1 ) = O (¯ p (¯ p -r 1 )( η -+ η + ) ln T ) .

Proof of Theorem 1. When η + ≥ η -, Theorem 1 holds true due to Lemma 3.1. When η + &lt; η -, for this case, let the policy p ∗ (¯ p ) be the optimal pricing policy under the reference effect ( η -, η + ) and under the starting reference price ¯ p . From Lemma 3.2, we know that policy p ∗ (¯ p ) is a markdown pricing policy. Thus,

<!-- formula-not-decoded -->

where inequality (a) is by Lemma 3.3 with ¯ p ≥ r 1 , and inequality (b) is by Lemma A.1.

## A.3 Missing Proofs of Section 3.3

To prove Proposition 2, we prove the following generalized and reparameterized version presented in Section 4:

Lemma 4.1 (Generalized and reparameterized version of Proposition 2) . Given a policy parameter θ = ( C 1 , C 2 ) ∈ Θ , a starting reference price r at time t 1 ∈ [ T ] , we define price curve ˜ p ( r, t 1 , θ ) /defines ( p t ) t ∈ [ t 1 ,T ] as:

where p † and t † are some deterministic functions of ( θ, r, t 1 ) , and r t = t 1 r + ∑ t -1 s = t 1 p s t , t ∈ [ t 1 , T ] . Given an instance I = ( a, b, η + , η -) , let θ ∗ /defines ( C ∗ 1 , C ∗ 2 ) with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

/negationslash

Then the price curve ˜ p ( r, t 1 , θ ∗ ) is optimal when η + = η -, i.e., V ∗ ( r, t 1 ) = V ˜ p ( r,t 1 ,θ ∗ ) ( r, t 1 ) . Furthermore, the price curve ˜ p (¯ p, t 1 , θ ∗ ) is near-optimal when η + = η -, namely, for any starting reference price r , V ∗ ( r, t 1 ) -V ˜ p (¯ p,t 1 ,θ ∗ ) ( r, t 1 ) ≤ O ( t 1 ln T / t 1 ) .

/negationslash

<!-- formula-not-decoded -->

Proof of Lemma 4.1. We first prove the optimality of price curve ˜ p ( r, t 1 , θ ∗ ) when η + = η -. Then we show the near optimality of price curve ˜ p (¯ p, t 1 , θ ∗ ) when η + = η -. The optimality of ˜ p ( r, t 1 , θ ∗ ) when η + = η -. In the proof, we show that the optimal pricing policy p ∗ ( r, t 1 ) = ˜ p ( r, t 1 , θ ∗ ) . Let η + = η -≡ η . Fix a time window [ t 1 , T ] and an starting reference price r t 1 = r at time t 1 . Recall that seller's program P OPT -BE

We denote partial derivatives by using subscripts. By first-order optimality condition, we know that the optimal price p ∗ t must satisfy

<!-- formula-not-decoded -->

By envelope theorem, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From (10), we have V ∗ r ( tr + p ∗ t t +1 , t +1 ) = -( t +1) Rev p ( p ∗ t , r ) , substituting it in (11) and we get

Thus, we can also deduce that

<!-- formula-not-decoded -->

Finally, substitute the above formula into (10) and obtain a condition which does not depend on the value function anymore:

<!-- formula-not-decoded -->

where we have used the fact that Rev r ( p, r ) = ηp . For base linear demand H ( p ) = b -ap , the above equality gives us

<!-- formula-not-decoded -->

From Theorem 1, we know that when η + = η -, the optimal pricing policy is a markdown pricing policy. Together with the above observation, we can deduce that the optimal pricing policy p ∗ = ( p ∗ t ) t ∈ [ t 1 ,T ] must keep charging the price as the highest possible price ¯ p for until time round t † , and then markdowns its prices according to (12) for the remaining rounds.

It now remains to characterize the time t † and the price p † . From Theorem 1 and (12), we know optimal pricing policy p ∗ = ( p ∗ t ) must satisfy that there exists a time round t † ∈ [ t 1 , T ] and a price p † at time t † such that:

<!-- formula-not-decoded -->

where ( r ∗ t ) is the reference price sequence from optimal pricing policy.

Let r † /defines r ∗ t † . We can now roll out the above relation for the price p ∗ t +1 until we can write the price p t +1 as a function of the initial price p † :

<!-- formula-not-decoded -->

where ( A t ( C ∗ 1 ) , B t ( C ∗ 1 , r † )) t ∈ [ t † ,T -1] are defined as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From (12), we can also deduce

<!-- formula-not-decoded -->

Plugging in the relation p ∗ t = A t ( C ∗ 1 ) p † + B t ( C ∗ 1 , r † ) and p ∗ T = C ∗ 1 r T + C ∗ 2 , we can pin down the price p † as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we intentionally feature the [ t † , T ] , r † , ( C ∗ 1 , C ∗ 2 ) dependence of p ( [ t † , T ] , r † , ( C ∗ 1 , C ∗ 2 ) ) prominently and

Clearly, the time step t † ∈ [ t 1 , T ] in optimal pricing policy should satisfy that

<!-- formula-not-decoded -->

We finish the proof of this part by showing the existence of t † . Let us look at the final time round T with the starting reference price r T = t 1 r +( T -t 1 )¯ p T ∈ [0 , ¯ p ] , then according to the above definition, we have

<!-- formula-not-decoded -->

where inequality (a) is due to the fact that ¯ p -( C ∗ 2 + C ∗ 1 r T ) ≥ ¯ p -( C ∗ 2 + C ∗ 1 ¯ p ) = 2 a ¯ p -b + η ¯ p 2( a + η ) ≥ 0 due to ¯ p &gt; b / 2 a from Assumption 2.2.

/negationslash

The near optimality of ˜ p (¯ p, t 1 , C ∗ 1 ) when η + = η -. In this part, we also write V ∗ ( r, t 1 | ( η -, η + )) to emphasize that it is the optimal value under the starting reference price r and customers reference effect parameters ( η -, η + ) .

When η + ≥ η -, let p ∗ (¯ p, t 1 ) = ( p ∗ t, ¯ p ) t ∈ [ t 1 ,T ] be the optimal pricing policy under starting reference price ¯ p and customers' reference effects ( η + , η -) . Let ( r ∗ t, ¯ p ) t ∈ [ T ] be the resulting reference price sequence under policy p ∗ (¯ p, t 1 ) and starting reference price ¯ p . We first note that

<!-- formula-not-decoded -->

where equality (a) is by Theorem 1 where we know that policy p ∗ (¯ p, t 1 ) must be a markdown pricing policy, and given the starting reference price ¯ p , we then must have p ∗ t, ¯ p ≤ r ∗ t, ¯ p for all t ∈ [ t 1 , T ] . Since the optimal pricing policy when customers have symmetric reference effects η + = η -is also a markdown policy, we can conclude that policy p ∗ (¯ p, t 1 ) also maximizes V p ( r, t 1 | ( η + , η + )) . In other words, we have policy p ∗ (¯ p, t 1 ) = ˜ p (¯ p, t 1 , θ ∗ ) .

Then,

<!-- formula-not-decoded -->

When η -&gt; η + , follow Lemma 3.2 and the similar argument above, we know that policy p ∗ (¯ p, t 1 ) also maximizes V p (¯ p, t 1 ) = ˜ p (¯ p, t 1 , θ ∗ ) . Then we have where inequality (a) is by Lemma 3.3 with r ≤ ¯ p , and inequality (b) is by Lemma A.1.

<!-- formula-not-decoded -->

where the last inequality is by Lemma A.1.

Then the proof of Proposition 2 follows immediately.

Proof of Proposition 2.

Proposition 2 follows by noting that

## Algorithm 3 Computing the pricing curve ˜ p ( r )

```
1: Input: starting reference price r , time horizon T , θ ∗ ← ( C ∗ 1 , C ∗ 2 ) . 2: Initialization: t 1 ← 1 . 3: while t 1 < T do 4: t † ← t 1 + T 2 . 5: Let r t † ← r +( t † -1)¯ p t † . 6: Solve the linear system A [ t † ,T ] ( θ ∗ ) p = b [ t † ,T ] ,r t † ( θ ∗ ) where A [ t † ,T ] ( θ ∗ ) , b [ t † ,T ] ,r t † ( θ ∗ ) are defined as in Definition A.1. 7: Let t 1 ← t 1 + t † 2 if p ∈ [0 , ¯ p ] T -t † +1 , otherwise let t 1 ← t † + T 2 . 8: end while 9: Return ˜ p ( r ) = ( p t ) t ∈ [ T ] where p t ← ¯ p 1 { t < t † } + p [ t -t † +1] 1 { t ≥ t † } for all t ∈ [ T ] .
```

Proposition 3 (Computing near-optimal markdown) . For any ( η + , η -) , there exists an algorithm (see Algorithm 3) that solves only O (ln T ) linear systems to compute the price curve ˜ p ( r ) defined in Proposition 2 for any r ∈ [0 , ¯ p ] .

Proof of Proposition 3. Notice that the price curve ˜ p ( r ) is optimal to the loss-neutral customers. Thus, To prove Proposition 3, it suffices to show that there exists an algorithm that can solve for an optimal pricing policy for loss-neutral customers by solving at most O (ln T ) linear systems.

t

1

= 1

in Lemma 4.1.

When η + = η -= η , the derived optimality condition (12) can be reformulated as follows: for any t ∈ [ t † , T ]

<!-- formula-not-decoded -->

As we can see, given a time round t † and the reference price r † at this round, the above condition essentially says that the optimal price p ∗ t can be represented by a linear combination over all other prices ( p ∗ s ) s ∈ [ t † ,T ] \{ t } . In other words, the partial price sequence ( p ∗ t ) t ∈ [ t † ,T ] forms a linear system where the matrix and vector in this system depend on model parameters C ∗ 1 , C ∗ 2 , t † , r † , T . In particular, the linear system can be defined as follows:

Definition A.1 (Linear system) . Given a time window [ t † , T ] and an starting reference price r t † = r † at time t † . We define the following matrix A [ t † ,T ] ( θ ∗ ) and the vector b [ t † ,T ] ,r † ( θ ∗ ) which takes parameter θ ∗ = ( C ∗ 1 , C ∗ 2 ) (defined as in (4.1) ) as input:

<!-- formula-not-decoded -->

So if we pin down the time round t † and the reference price r † at this time round, then the optimal price sequence for the remaining times ( p ∗ t ) t ∈ [ t † ,T ] , which satisfies the condition (12), must also be the solution to the linear system A [ t † ,T ] ( θ ∗ ) p = b [ t † ,T ] ,r † ( θ ∗ ) . Recall that by Theorem 1, we know optimal pricing policy must be a markdown pricing policy, by Proposition 2, the reference price r † satisfies r † = r +¯ p ( t † -1) t † . Thus, to solve the optimal pricing policy ( p ∗ t ) t ∈ [ T ] , it suffices to determine the time round t † . Recall that by Theorem 1, we know optimal pricing policy must be a markdown pricing policy. Thus, the time round t † is smallest time index such that the solution A [ t,T ] ( θ ∗ ) p = b [ t,T ] , r +( t -1)¯ p t ( θ ∗ ) is a feasible solution in [0 , ¯ p ] T -t +1 . That is,

<!-- formula-not-decoded -->

With the above observation, we can have a binary search algorithm to pin down the time round t † . We summarize our algorithm as in Algorithm 3.

## B Missing Algorithms in Section 4

## Algorithm 4 ResetRef ( t, r t , r ) : Reset to the target reference price

- 1: Input: Current time step t , the reference price r t for this time t , and a target reference price r
- 2: if r t &lt; r then
- 3: Let N ( t, r t , r ) /defines min { N ∈ N : ( t + N +1) r -tr t -N ¯ p ∈ (0 , ¯ p ) } .
- 5: Return N ( t, r t , r ) + 1 .
- 4: Keep setting the price ¯ p with N ( t, r t , r ) rounds and then set the price ( t + N ( t, r t , r ) + 1) r -tr t -N ( t, r t , r )¯ p for 1 round.
- 6: else if r t &gt; r then
- 7: Let N ( t, r t , r ) /defines min { N ∈ N : ( t + N +1) r -tr t ∈ (0 , ¯ p ) } .
- 8: Keep setting the price 0 with N ( t, r t , r ) rounds and then set the price ( t + N ( t, r t , r )+1) r -tr t for 1 round.
- 9: Return N ( t, r t , r ) + 1 .
- 10: else
- 11: Return 0 .
- 12: end if

## C Missing Proofs in Section 5

## C.1 Proof for Appendix C.1

Proposition 4. Given a price upper bound ¯ p ≥ 1 , consider the following problem instances I ′ : let η + = η -≡ 0 , b ≡ 1 , and a ∈ [ 3 4¯ p , 1 ¯ p ] . Then clearly all instances in I ′ satisfy Assumption 2.2. And the expected regret of any algorithm π ALG satisfies that sup I ∈I ′ REG π ALG [ T, I ] ≥ ¯ p √ T -1 3 √ 1+36 π 2 . 12

Proof of Proposition 4. Our lower bound proof is based on the proof of Theorem 1 in Keskin and Zeevi (2014) with certain necessary modifications. We first show that all instances in I ′ satisfy Assumption 2.2. Notice that this assumption implies

<!-- formula-not-decoded -->

Since b ≡ 1 , above condition implies that 1 2¯ p &lt; a ≤ 1 ¯ p . By construction, we know a ∈ [ 3 4¯ p , 1 ¯ p ] for all I ∈ I ′ . Thus given ¯ p ≥ 1 , all instances I ∈ I ′ satisfy Assumption 2.2.

Given an instance I ∈ I ′ , let p ( I ) = b 2 a = 1 2 a . Let Λ be an absolutely continuous density on I ′ , taking positive values on the interior of I ′ and zero on its boundary. Then, the multivariate van Trees inequality (cf. Gill and Levit 1995) implies that

<!-- formula-not-decoded -->

˜ 12 The lower bound we derive here scales linearly w.r.t. price upper bound ¯ p , it is an interesting open question to study whether one can tighten this gap between the linear dependency in lower bound and the cubic-ish dependency of ¯ p in our upper bound.

where ˜ F (Λ) is the Fisher information for the density Λ , and E Λ [ · ] is the expectation operator with respect to density Λ . Notice that for all I = ( a, b ) ∈ I ′ , we have C ( I ) ( ∂p ( I ) ∂I ) /latticetop = -p ( I ) 2 a , and F π ALG t -1 ( I ) = E π ALG I [ J t -1 ] / σ 2 where denotes the empirical Fisher information, ( p s ) s ≥ 1 are the pricing choices realized from algorithm π ALG . Using these identities and adding up inequality (14) for t = 2 , . . . , T , we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since E Λ [ · ] is a monotone operator, we further have

Note due to the fact that C ( I ) = [ -p ( I )1] , we have C ( I ) E π ALG I [ J t -1 ] C ( I ) /latticetop = ∑ t -1 s =1 E π ALG I [ ( p s -p ( I )) 2 ] . Thus, (15) implies that

<!-- formula-not-decoded -->

Recall the definition of our regret, we then have

Thus,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Rearranging the above inequality we further have

<!-- formula-not-decoded -->

Now observe that

<!-- formula-not-decoded -->

where a min /defines 3 4¯ p . Since we know that b ≡ 1 , ∀ I ∈ I ′ , following a standard choice of Λ (cf. Goldenshluger and Zeevi 2009, p. 1632 for a choice of Λ ) over the model parameter space I ′ can give us F (Λ) = π 2 (¯ a -a min ) -2 where ¯ a /defines 1 ¯ p . Thus, we further have

<!-- formula-not-decoded -->

Recall that by our construction, we have

<!-- formula-not-decoded -->

which finishes the proof by considering bounded variance noises

## C.2 Missing Proofs in Step 1

In the analysis of this step, for notation simplicity we omit the superscript GR in price p GR s , and we ignore all time steps used to reset the reference price (i.e., Line 7 in Algorithm 2).

To prove Lemma 5.1, we show the following stronger result:

Lemma C.1. Fix any reference price r ∈ (¯ p -δ, ¯ p ] , the greedy price p GR ( r ) for this reference price satisfies that p GR ( r ) = C ∗ 1 r + C ∗ 2 ∈ [ d, r -d ] where d = 1 2 ( r -max I ∈I b 2 a ) and d &lt; r -d .

Proof of Lemma C.1. By the first-order condition for the maximization problem in (6), we know that

Below we show that when r ∈ (¯ p -δ, ¯ p ] , we always have C ∗ 1 r + C ∗ 2 &lt; r . To see this

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For an input reference price r ∈ (¯ p -δ, ¯ p ] , we thus always have p GR ∈ [ d, r -d ] . The proof then completes.

Proposition 5. Given two reference prices r a , r b ∈ (¯ p -δ, ¯ p ] where r a &lt; r b , let ˆ θ = ( ˆ C 1 , ˆ C 2 ) be the estimated parameter obtained from Line 12 in Algorithm 1, then the following holds with probability at least 1 -2 δ for δ ∈ (0 , 1 / e ) :

Proof of Proposition 5. By triangle inequality, we know that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma C.2. Let g t /defines p t D t ( p t ,r ) d κ , where p t is the chosen price in Line 10 in Algorithm 2, then we have E [ g t ] = ∂ Rev ( p t ,r ) ∂p t .

where equality (a) holds true by Lemma 5.2 and holds with probability at least 1 -2 δ . Similarly, we can also derive the high-probability bound for the estimation error ∣ ∣ ∣ ˆ C 2 -C ∗ 2 ∣ ∣ ∣ .

Proof of Lemma C.2. By the way κ is chosen, we have that E [ κ ] = 0 , E [ κ 2 ] = 1 and E [ κ 3 ] = 0 . Recall that by the design of the algorithm, we must have p t ≤ r . Thus, we have D ( p t , r ) = b -ap + η + ( r -p t ) .

<!-- formula-not-decoded -->

Lemma C.3. Fix a r as the input of Algorithm 2, the one-shot revenue function Rev ( · , r ) is L -Lipschitz where L /defines b + ¯ η ¯ p . And moreover for any time t , we have that ∣ ∣ p t -p GR ( r ) ∣ ∣ ≤ 2 L / λ holds with probability 1 where λ /defines 1 / 2¯ p .

Proof of Lemma C.3. We first show that fix any reference price r ∈ [0 , ¯ p ] , the one-shot revenue function Rev ( · , r ) is ( ¯ b + ¯ η ¯ p ) -Lipschitz. Notice that

<!-- formula-not-decoded -->

where inequality (a) holds true by triangle inequality; inequality (b) holds true by Assumption 2.2 which implies that ¯ p &gt; b / 2 a , and thus 2( a + η + )¯ p -b &gt; 0 ; and equality (c) holds true by b -(2( a + η + )¯ p -b ) = 2( b -( a + η + )¯ p ) ≥ 0 by D (¯ p, r ) ≥ 0 .

For any reference price r ∈ [0 , ¯ p ] , we know that the function Rev ( p, r ) is a -λ -strongly-concave function. Using strong concavity, we have

<!-- formula-not-decoded -->

Lemma C.4. For any t ≥ 2 , we have the following holds

If | p GR ( r ) -p t | = 0 , then the statement trivially holds true, otherwise we have ∣ ∣ p t -p GR ( r ) ∣ ∣ ≤ 2 L / λ by dividing ∣ ∣ p t -p GR ( r ) ∣ ∣ in both sides of the above inequality.

<!-- formula-not-decoded -->

where δ ( g ) i /defines g i -g i for all i ∈ [ t ] .

Proof of Lemma C.4. By the strong concavity of the revenue function Rev ( p, r ) and p GR ( r ) is the maximizer of the function Rev ( p, r ) , we have and moreover,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Notice that for any price p ′ and any p ∈ [0 , ¯ p ] we have ∣ ∣ PROJ [ d, ¯ p -d ] ( p ′ ) -p ∣ ∣ ≤ | p ′ -p | . We also define δ ( g ) t /defines g t -g t . Recall that from Lemma C.2, we know E [ δ ( g ) t ] = 0 . With these inequalities

and observations, we have the following:

<!-- formula-not-decoded -->

where inequality (a) holds true by (16); and inequality (b) holds true by (17). We unfold the above recursive inequality till t = 2 , we then have for any t ≥ 2 , where equality (a) uses the observation that ∏ t j = i +1 ( 1 -2 j ) = i ( i -1) t ( t -1) , thus completing the proof.

<!-- formula-not-decoded -->

To finish the proof of Lemma 5.2, we also need the following technique lemma:

Lemma C.5 (See Rakhlin et al. 2012) . Let δ 1 , . . . , δ T be a martingale difference sequence with a uniform bound | δ i | ≤ b for all i . Let V s = ∑ s t =1 Var t -1 [ δ t ] be the sum of conditional variances of δ t 's. Further, let σ s = √ V s . Then we have, for any δ ≤ 1 / e and T ≥ 4 ,

<!-- formula-not-decoded -->

We are now ready to prove Lemma 5.2.

Lemma 5.2. Let δ ∈ (0 , 1 / e ) and let r be the reference price to the input of Algorithm 2. For any T 1 ≥ 4 , we have the following holds with probability at least 1 -δ :

where d is the parameter defined in Line 2 in Algorithm 2.

<!-- formula-not-decoded -->

Proof of Lemma 5.2. In below proof, we fix a reference price r ∈ [0 , ¯ p ] as the input to the Algorithm 2. We start our analysis from Lemma C.4. Let z t /defines δ ( g ) t ( p t -p GR ( r )) . Notice that by definition, we have E [ z t | ( p s , ˜ p s , g s , D s ) s ∈ [ t -1] ] = 0 . Recall that | δ ( g ) t | = | g t -g t | ≤ | g t | + | g t | ≤ | g t | +( ¯ b + ¯ η ¯ p ) . For notation simplicity, let L /defines ¯ b + ¯ η ¯ p . By Cauchy-Schwartz inequality, we also know that the conditional variance Var t -1 [ z t ] /defines E [ ( z t -E [ z t ]) 2 | ( p s , ˜ p s , g s , D s ) s ∈ [ t -1] ] ≤ ( | g t | + L ) 2 ( p t -p GR ( r )) 2 . Thus, we have the following holds about the sum of conditional variances:

For i ∈ [ t ] , we also have the following uniform bound

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where inequality (a) holds true by Lemma C.3. Thus with Lemma C.5, when T ≥ 4 , δ &lt; 1 / e , we have with probability at least 1 -δ , the following holds for all t ≤ T :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where inequality (a) uses Lemma C.5. When the noise { ε i } follows a bounded distribution with uniform bound ¯ ε , then we have

<!-- formula-not-decoded -->

where ¯ R /defines max p,r Rev ( p, r ) . Back to (18), we have

<!-- formula-not-decoded -->

where we have ¯ G /defines ¯ g ∨ L = √ 2 d 2 ( ¯ R 2 + ¯ p 2 ¯ ε 2 ) ∨ ( ¯ b + ¯ η ¯ p ) ≤ √ 2¯ p (1+¯ p ) d ∨ (1 + ¯ p ) , and in the last step, we use an induction argument to prove the desired inequality. Notice that by Assumption 2.2, we know ¯ p &gt; a 2 a . Thus we have a ≥ b 2¯ p for all a, b . Since we know max I ∈I b = 1 , and max I ∈I a = 1 , we know that ¯ p ≥ 1 / 2 . Thus we have ¯ G ≤ 3¯ p / d .

## C.3 Missing Proofs in Step 2

Proof of Lemma 5.3. We bound the total number of rounds used in Algorithm 4 in Exploration phase 1, similar analysis can be carried over to the Exploration phase 2. When r t &lt; r a , we have the following inequality on the number of rounds used N ( t, r t , r a ) :

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

/negationslash where in equality (a) we have used the fact that whenever we have r t = r a , one must have that r t -1 = r a , and in last inequality, we have used the fact that r a = ¯ p / 4 .

When r t &gt; r a , we have the following inequality on the number of rounds used N ( t, r t , r a ) :

<!-- formula-not-decoded -->

Thus we have

<!-- formula-not-decoded -->

/negationslash where in equality (a) we have used the fact that whenever we have r t = r a , one must have that r t -1 = r a , and in last inequality, we have used the fact that r a = ¯ p / 4 . Thus, in both cases, we have that N ( t, r t , r a ) = 0 for any t in Exploration phase 1. Similarly, we can also show that N ( t, r t , r a ) = Θ(1) for any t in Exploration phase 1.

We next bound the number rounds used in ResetRef for resetting reference price r a to be r b . Let t be the time round that enters in the smoothing phase. From Lemma 5.3, we know that t = Θ( T 1 ) . Since r b &gt; r a , we have the following inequality on the number of rounds used N ( t, r a , r b ) :

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

which completes the proof.

## C.4 Missing Proofs in Step 3

Lemma 5.4 (Strong concavity of function V p ) . Fix a starting time t 1 ∈ [ T ] and starting reference price r . If the price curve ˜ p ( r, t 1 , θ ∗ ) is a strict markdown price curve (i.e., t † = t 1 ), then there exists two positive, finite constants c 1 and c 2 where c 1 &gt; c 2 , such that the eigenvalue λ V of the Hessian matrix ∇ 2 p V p ( r, t 1 ) satisfies λ V ∈ [ -2( a + η ) c 1 , -2( a + η ) c 2 ] .

Proof of Lemma 5.4. Given a pricing sequence p = ( p t ) t ∈ [ t 1 ,T ] , we can write the value function as follows

<!-- formula-not-decoded -->

Taking the first-order derivative of function V p ( r, t 1 ) over each price p t ,

<!-- formula-not-decoded -->

where S t /defines r t + ∑ T s = t +1 p s s = t 1 r + ∑ t -1 s = t 1 p s t + ∑ T s = t +1 p s s . The Hessian matrix of V p ( r, t 1 ) equals to

<!-- formula-not-decoded -->

In below, we will show that the matrix -H V ( p ) is a strictly diagonally dominant matrix. Notice that in matrix -H V ( p ) , the sum of all non-diagonal entries in a row is decreasing when row index increases, and all diagonal entries have the value 1 . Thus, to show the strictly diagonal dominance of matrix -H V ( p ) , it suffices to show that

<!-- formula-not-decoded -->

To prove the above inequality, notice that the Hessian matrix H V ( p ) is exactly the matrix A [ t 1 ,T ] ( θ ∗ ) . Let p ∗ = ( p ∗ t ) t ∈ [ t 1 ,T ] be the solution to the following linear systems: A [ t 1 ,T ] ( θ ∗ ) p = b [ t 1 ,T ] ,r ( θ ∗ ) . By definition, we then have

<!-- formula-not-decoded -->

Since p t ≤ [0 , ¯ p ] for every t ∈ [ t 1 , T ] , we know that

<!-- formula-not-decoded -->

Moreover, from Theorem 1, we know that the pricing policy p ∗ is a markdown pricing policy. Thus, together with the above observations, from (20), we have

<!-- formula-not-decoded -->

where inequalities (a), (d) hold true by C ∗ 2 = b 2( a + η ) ≥ 1 2 ¯ p , inequality (b) holds true p ∗ t 1 ≤ ¯ p , and inequalities (c), (e) hold true by p ∗ t ≥ p ∗ T , ∀ t ∈ [ t 1 , T ] . From the above inequalities, we can deduce the following inequality

<!-- formula-not-decoded -->

Thus, matrix -H V ( p ) is a strictly diagonally dominant matrix with positive diagonal entries. This implies that the Hessian matrix H V ( p ) is strictly negative definite. By Gershgorin Circle Theorem, we know that any eigenvalue λ V of the Hessian matrix H V ( p ) must satisfy that

<!-- formula-not-decoded -->

which implies that the value function V p ( r, t 1 ) is strongly concave.

Lemma 5.5 (Lipschitz error on the price curve) . Fix a starting time t 1 ∈ [ T ] and starting reference price r . If price curves ˜ p ( r, t 1 , θ 1 ) and ˜ p ( r, t 1 , θ 2 ) are both strict markdown price curves, then we have ‖ ˜ p ( r, t 1 , θ 1 ) -˜ p ( r, t 1 , θ 2 ) ‖ ≤ O ( ¯ p ‖ θ 1 -θ 2 ‖ √ T -t 1 ln T / t 1 ) .

Proof of Lemma 5.5. In below proof, we fix a starting time t 1 ∈ [ T ] and the initial reference price r . Given the policy parameter θ = ( y, z ) ∈ [0 , 1 / 2 ) × ( ¯ p / 2 , ∞ ] , let the matrix A [ t 1 ,T ] ( θ ) and b [ t 1 ,T ] ,r ( θ ) be defined as in (13). Given two policy parameters θ 1 = ( y 1 , z 1 ) and θ 2 = ( y 2 , z 2 ) , we can also express the matrix A [ t 1 ,T ] ( θ 1 ) = A [ t 1 ,T ] ( θ 2 ) + ∆ A and the vector b [ t 1 ,T ] ,r ( θ 1 ) = b [ t 1 ,T ] ,r ( θ 2 ) + ∆ b as the perturbed matrix of A [ t 1 ,T ] ( θ 2 ) and the perturbed vector of b [ t 1 ,T ] ,r ( θ 2 ) where perturbation matrix ∆ A and the perturbation vector ∆ b depend on the error θ 1 -θ 2 . For notation simplicity, let ˜ p ( r, t 1 , θ ) = p ( θ ) . With the above definitions, we then have

<!-- formula-not-decoded -->

where p ( θ 1 ) = p ( θ 2 )+∆ p . Expanding the above equality and using the exact solution A [ t 1 ,T ] ( θ 2 ) p ( θ 2 ) = b [ t 1 ,T ] ,r ( θ 2 ) , we then have

<!-- formula-not-decoded -->

Taking the infinite norm on both sides, we then have

<!-- formula-not-decoded -->

Rearranging the terms, we have

Observe that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In below, we provide the upper bound of ∥ ∥ ∥ ( A [ t 1 ,T ] ( θ 2 ) ) -1 ∥ ∥ ∥ ∞ . By the definition of p ( θ 2 ) where A [ t 1 ,T ] ( θ 2 ) p ( θ 2 ) = b [ t 1 ,T ] ,r ( θ 2 ) , we have

<!-- formula-not-decoded -->

which gives us

<!-- formula-not-decoded -->

where inequalities (a), (d) holds true by z 2 ≥ 1 2 ¯ p , inequality (b) holds true by assumption that p t 1 ( θ 2 ) ≤ ¯ p , and inequalities (c), (e) holds true by the observation that p t ( θ 2 ) ≥ p T ( θ 2 ) , ∀ t ∈ [ t 1 , T ] . From the above inequalities, we have

<!-- formula-not-decoded -->

The above inequality implies that the matrix A [ t 1 ,T ] ( θ 2 ) is the strictly diagonally dominant matrix. Thus, by Neumann Series Theorem, we know that

<!-- formula-not-decoded -->

where I is the identity matrix, and the matrix A ′ [ t 1 ,T ] ( θ 2 ) has all zero diagonal values and all positive values in all non-diagonal entries. With the above observation, we thus have

<!-- formula-not-decoded -->

Notice that every entry in matrix A ′ [ t 1 ,T ] ( θ 2 ) is non-negative, and every entry in the vector b [ t 1 ,T ] ,r ( θ 2 ) is no smaller than y 2 rt 1 / T + z 2 . Thus, we have

Thus, we can deduce that

<!-- formula-not-decoded -->

Thus, back to (22), we know that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then from the above characterizations, we know that

<!-- formula-not-decoded -->

The proof then completes.

Proposition 6 (Bounding the Lipschitz error) . Assume η + = η -≡ η . Fix a starting time t 1 ∈ [ T ] , a starting reference price r at time t 1 . Then, the following holds for all θ ∈ Θ ,

Proof of Proposition 6. Let ( r ∗ t ) t ∈ [ t 1 ,T ] (resp. ( r t ) t ∈ [ t 1 ,T ] ) be the reference price sequence under the pricing policy ˜ p ( r, t 1 , θ ∗ ) (resp. ˜ p ( r, t 1 , θ ) ). For pricing policy ˜ p ( r, t 1 , θ ∗ ) (resp. ˜ p ( r, t 1 , θ ) ), we use τ ∗ /defines t † ( θ ∗ ) (resp. τ /defines t † ( θ ) ) to denote the time index such that for every t ∈ [ τ ∗ , T ] (resp. t ∈ [ τ, T ] ), we have p ∗ t &lt; ¯ p (resp. p t &lt; ¯ p ). Let ˜ p ( r, t 1 , θ ∗ ) = p ( θ ∗ ) = ( p ∗ t ) t ∈ [ t 1 ,T ] , and ˜ p ( r, t 1 , θ ) = p ( θ ) = ( p t ( θ )) t ∈ [ t 1 ,T ] . For notation simplicity given a time s , we also define p ∗ s : T /defines ( p ∗ t ) t ∈ [ s,T ] , p s : T /defines ( p t ( θ )) t ∈ [ s,T ] .

<!-- formula-not-decoded -->

In below proof, we bound the Lipschitz error based on two possible cases: (1) τ ∗ = τ ; (2) τ ∗ = τ .

/negationslash

Case 1 - When τ ∗ = τ :. In this case, the Lipschitz error can be decomposed as follows:

<!-- formula-not-decoded -->

where equality (a) is by the fact that r ∗ s = r s , ∀ s ∈ [ t 1 , τ ] since p ∗ s = p s = ¯ p, ∀ s ∈ [ t 1 , τ ] . Notice that given the reference price r τ at time τ , the price sequence p ∗ τ : T is the optimal pricing curve for the time window [ τ, T ] , namely, it equals to the pricing curve ˜ p ( r τ , τ, θ ∗ ) . Since p ∗ τ ∈ [0 , ¯ p ) by definition of τ , from Lemma 5.4, we know that the value function V p ( r τ , τ ) is strongly concave over the price sequence p and its Hessian matrix has bounded eigenvalues. Thus, together with Lemma 5.5, we have

<!-- formula-not-decoded -->

where in last inequality we have τ ≥ t 1 .

Case 2a - When τ ∗ &lt; τ :. In this case, by definition, we know that p ∗ t = ¯ p for all t ∈ [ t 1 , τ ∗ ] , and p t = ¯ p for all t ∈ [ t 1 , τ ] . Thus, we know that r ∗ τ ∗ = r τ ∗ .

Let p † t ( θ ) be the first price of the solution to the linear system A [ t,T ] ( θ ) p = b [ t,T ] ,r t ( θ ) over the time window [ t, T ] where r t = t 1 r +( t -t 1 )¯ p t . Recall that p ∗ τ ∗ : T = ( p ∗ t ) t ∈ [ τ ∗ ,T ] is the solution the linear system A [ τ ∗ ,T ] ( θ ∗ ) p = b [ τ ∗ ,T ] ,r τ ∗ ( θ ∗ ) . Thus, follow the similar analysis in the proof of Lemma 5.5, i.e., (24), we also have

<!-- formula-not-decoded -->

where ε 1 (resp. ε 2 ) is the estimation error for C ∗ 1 (resp. C ∗ 2 ). When p † τ ∗ ( θ ) &lt; 0 , then from the above inequality, we know that

<!-- formula-not-decoded -->

From Theorem 1, we know that ∥ ∥ ∥ p ∗ [ τ ∗ ,T ] ∥ ∥ ∥ ∞ ≤ p ∗ τ ∗ . Consider a new pricing policy p ‡ = ( p ‡ t ) t ∈ [ τ ∗ ,T ] where p ‡ t ≡ 0 for all t ∈ [ τ ∗ , T ] . Then,

<!-- formula-not-decoded -->

where in equality (a) we have V p ‡ ( r ∗ τ ∗ , τ ∗ ) = 0 , and inequality (b) is due to ∥ ∥ p ∗ τ ∗ : T -p ‡ ∥ ∥ ∞ ≤ p ∗ τ ∗ and the Hessian matrix of the value function V p ( r ∗ τ ∗ , τ ∗ ) has bounded eigenvalues Lemma 5.4 and the results in Lemma 5.5.

When p † τ ∗ ( θ ) &gt; ¯ p , then follow the similar analysis in the proof of Lemma 5.5 for any t ∈ [ τ ∗ , T ] ,

<!-- formula-not-decoded -->

where equality (a) is due to the definition of r t , inequality (b) is due to the fact that | ¯ p -p ∗ t -1 | ≤ | p † t -1 ( θ ) -p ∗ t -1 | . Thus, we know that ‖ p ( θ ) -p ∗ ‖ ∞ ≤ O ( ¯ p ¯ ε ln T t 1 ) . Consequently, we can bound the Lipschitz error V ∗ ( r, t 1 ) -V ˜ p ( r,t 1 ,θ ) ( r, t 1 ) = O ( ¯ p 2 ‖ θ ∗ -θ ‖ 2 ( T -τ ∗ ) ( ln T τ ∗ ) 2 ) with using Lemma 5.4 for value function V p ( r ∗ τ ∗ , τ ∗ ) , and the results in Lemma 5.5.

Case 2b - When τ ∗ &gt; τ :. In this case, we also define p † t ( θ ∗ ) as the first price of the solution to the linear system A [ t,T ] ( θ ∗ ) p = b [ t,T ] ,r ∗ t ( θ ∗ ) over the time window [ t, T ] where r ∗ t = t 1 r +( t -t 1 )¯ p t . As we have r τ = r ∗ τ ,

<!-- formula-not-decoded -->

and similarly, we also have for any t ∈ [ τ, τ ∗ -1]

We can roll out the value of p t ( θ ) for t ∈ [ τ, τ ∗ -1]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Notice that at time t = τ ∗ -1 , we have

<!-- formula-not-decoded -->

Thus, we can deduce the following condition on the time round τ ∗ :

<!-- formula-not-decoded -->

Now we can bound the Lipschtiz error as follows:

<!-- formula-not-decoded -->

where inequality (a) is due to Lemma 3.3 and ( p ∗ t,r τ ∗ ) t ∈ [ τ ∗ ,T ] is the optimal pricing policy for the time window [ τ ∗ , T ] given the initial reference price r τ ∗ , inequality (b) follows similarly as in the case 1 where the value function V p ( r τ ∗ , τ ∗ ) is a strongly-concave function, and the fact that r ∗ τ ∗ -r τ ∗ ≤ ¯ p ( τ ∗ -τ ) τ ∗ , and inequality (c) is to due to the upper bound of the time round τ ∗ established in (25) and we thus have ( τ ∗ -τ ) ln( T -τ ∗ ) = O (1) .