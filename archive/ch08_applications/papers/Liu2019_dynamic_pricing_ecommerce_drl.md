## DYNAMIC PRICING ON E-COMMERCE PLATFORM WITH DEEP REINFORCEMENT LEARNING: A FIELD EXPERIMENT

## A PREPRINT

## Jiaxi Liu

Alibaba Supply Chain Platform galiliu.ljx@alibaba-inc.com

## Xiaoqing Wang

Alibaba Supply Chain Platform robin.wxq@alibaba-inc.com

## Yuming Deng

## Yidong Zhang ∗

Alibaba Supply Chain Platform tanfu.zyd@alibaba-inc.com

## Xingyu Wu

Alibaba Supply Chain Platform yuming.dym@alibaba-inc.com

December 2019

## ABSTRACT

In this paper we present an end-to-end framework for addressing the problem of dynamic pricing (DP) on E-commerce platform using methods based on deep reinforcement learning (DRL). By using four groups of different business data to represent the states of each time period, we model the dynamic pricing problem as a Markov Decision Process (MDP). Compared with the state-of-the-art DRL-based dynamic pricing algorithms, our approaches make the following three contributions. First, we extend the discrete set problem to the continuous price set. Second, instead of using revenue as the reward function directly, we define a new function named difference of revenue conversion rates (DRCR). Third, the cold-start problem of MDP is tackled by pre-training and evaluation using some carefully chosen historical sales data. Our approaches are evaluated by both offline evaluation method using real dataset of Alibaba Inc., and online field experiment starting from July 2018 with thousands of items, lasting for months on Tmall.com. To our knowledge, there is no other DP field experiment using DRL before. Field experiment results suggest that DRCR is a more appropriate reward function than revenue, which is widely used by current literature. Also, continuous price sets have better performance than discrete sets and our approaches significantly outperformed the manual pricing by operation experts.

K eywords Reinforcement learning · Dynamic pricing · E-commerce · Revenue management · Field experiment

Dynamic pricing, to adjust prices according to inventories left and demand response observed, has drawn great attentions during the past decades since the deregulation of the airline industry in the 1970s. [1] and [2] gave overviews of the research that has been done in the field of perishable-asset revenue management, which is a field that combines the areas of yield management, overbooking, and pricing.

During the recent development of business, many industries have become more active in revenue management. Ridesharing platforms like Uber has implemented dynamic pricing strategy, known as 'surge' pricing and [3] showed that it has significant impact on motivations for more driving times. Retailers like Zara have implemented systematic dynamic markdown pricing strategy [4]. Kroger is now testing electronic price tag at one store in Kentucky ([5]).

Online retailers have a stronger desire for dynamic pricing strategies due to the requirement of more complex operations. For example, Amazon.com sells 356 million products (562 million now). Walmart.com sells 4.2 million products according to a 2017 estimate 2 . Taobao.com, the biggest E-commerce platform in China, sells billions of products at present. Operation specialists have to set prices for these items periodically to remain competitive while maximize revenue, which will be mission impossible when the number of items goes this high. As a result, Amazon has

∗ Correspondence: 969 West Wen Yi Road, Yu Hang District, Hangzhou, China, 311121

2 https://www.scrapehero.com/how-many-products-does-walmart-com-sell-vs-amazon-com

Alibaba Supply Chain Platform zhuyang.wxy@alibaba-inc.com

implemented automatic pricing systems and it is reported that Amazon.com can change prices every 15 minutes 3 . [6] studied the pricing strategies for Amazon.com.

In this paper, we proposed a reinforcement learning approach to address the dynamic pricing problem for online retailers. The scenario we consider is how to dynamically price for different products on Tmall.com, the largest business-to-consumer retailer in China, spun off from Taobao.com. There are many difficulties for pricing on such an E-commerce platform. First, the market environment is impossible to be quantified. Demand for the same product could change dramatically due to unpredictable fluctuation of the daily customer traffic, the change of other products' prices or even comments from the previous buyers. Second, it would lead to non-convergence policies if the reward function is not set properly under such complicated environment. Third, it is not appropriate to apply a learning model online directly, since a slightly inappropriate price online could quickly cause large capital loss. Fourth, unlike recommendation system[7], it is impossible to do online A/B testing, because it is illegal to expose different prices at the same time to different customers. That is an obstacle to evaluate the performances of different pricing policies during the field experiment.

To overcome these difficulties, we propose a framework for dynamic pricing with DRL to optimize long-term revenue. This paper has several contributions. First of all, we are the first to apply DRL for both discrete and continuous pricing problem on real-world E-commerce platform, rather than traditional discrete one using Q-learning ([8]) or within simulated environment ([9], [10], [8], [11], [12]). To achieve this, the real-world market environment is precisely described with four groups of features we defined. Second, we found that the revenue conversion rate and the difference of the revenue conversion rate are more suitable as the reward function rather than the revenue ([10], [8], [13]) due to its concave nature. Third, aiming the cold start problem, we designed pre-training and evaluation procedures within our pricing framework. Fourth, we defined simi-products on E-commerce platform, which we proved in our experiments that they could be used for evaluating different pricing policies. Finally, we carry out large-scale field experiments lasting for months with thousands of SKUs of products priced by our DRL models. Our field experiments indicate the effectiveness of our dynamic pricing framework on E-commerce platform, which is the first work of this kind with such achievement.

The remaining of this paper is organized as follows: The next section lists some related work in dynamic pricing problem. Section 2 introduces approaches we designed for dynamic pricing, where the problem is modeled as a Markov Decision Process model. Both discrete pricing action model and continuous pricing action model are proposed. In section 3, the results from both offline and online experiments are introduced, which validate our approach of the problem. The conclusions and future work directions are summarized in section 4.

## 1 Literature review

Much research has been done in dynamic pricing for decades. We refer to [14] for a comprehensive review for recent developments. It combines two research fields: (1) statistical learning, specifically applied to the problem to estimate demand and (2) price optimization. Most of previous research has focused on the cases where a functional relationship between price and demand is assumed to be known to decision makers. [15] is acknowledged to be the first to mathematically describe the price-demand relation of products and solve the mathematical problem to achieve the optimal revenue. However, it assumed that the relationship is static over time which usually does not hold in reality. [16] assumed that the demand is not only a function of price, but also the time-derivative of price, leading to a dynamic demand function of price over time. [17] introduced a stochastic model to capture demand uncertainty while optimizing the prices. [18] considered constraints like limited inventories and a finite planning horizon.

In practice, it is often difficult to describe the demand beforehand. Much recent research focuses on the dynamic pricing with unknown demand function. Some researchers addressed the problem by parametric approaches. [19] assumed parametric families of demand functions to be learned over time. [20] proposed an approach to learn from the historical purchase data. [21] utilized Bayesian dynamic pricing policies to address demand uncertainties. However, revenue may depart from the optimal due to mis-specifying the demand family. Therefore, much recent research mainly revolves around non-parametric approaches. [22], [23], [24] looked deep inside learning while earning approaches. However, they all assumed the revenue function is strictly concave and differentiable, which could not hold in E-commerce retail industry as shown in section 3.1.

With the development of computation, reinforcement learning (RL) is introduced to address dynamic problems ([25], [26]). [9] demonstrated the possibility of using Q-learning to express anticipated future-discounted profits for possible prices, forming a so called pricebot to adjust prices in response to changing market conditions. [13] used Temporal Difference for Information products' dynamic pricing from a yield management view. [11] formulated single seller and

3 https://www.whitehouse.gov/sites/default/files/docs/Big\_Data\_Report\_Nonembargo\_v2˙ pdf

Historical Sales Data

•Pre-training-

Demonstrations

Agent

—Pricing Action-

Reward &amp; Observation

Market

Environment two sellers dynamic pricing problems and employ different RL algorithms in a simulated context. [27] used different types of asynchronous multi-agent RL methods to determine the competitive pricing strategy in the market scenario. [10] and [12] utilized reinforcement learning to optimize prices in the energy market. [8] suggested Q-learning with neural networks approximation to maintain revenue while improving fairness in a simulated environment. All these previous works, however, are carried out in simulations with simplified market settings, where the reward defined with revenue work out well. Meanwhile, DNNs are only used for approximation of discrete prices, which, however, is not the case for real-world market.

## 2 Methodology

In our study we first represent the dynamic pricing problem as a Markov Decision Process (MDP). The agent periodically changes prices of the products as its action after observing environment state. The new environment state could then be observed and the reward could also be received. Each pricing episode reaches its end if the product is out of stock. The model is pre-trained by historical sales data and previous specialists' pricing actions, which is also used for offline evaluation. The framework is shown in Figure 1.

Figure 1: Dynamic pricing framework using DRL with demonstrations on E-commerce platform.

<!-- image -->

## 2.1 Problem formulation

We mainly consider two kinds of dynamic pricing applications for E-commerce platform, named markdown pricing and daily pricing in the rest of this work. For both markdown pricing and daily pricing, we define n products labeled by i = 1 , 2 , ..., n individually. The pricing process is formed as Markov Decision Process (MDP) referring works from [10], [11], [12] etc. Prices are decided to be modified or kept at discrete time steps t = 1 , 2 , ..., T referring to the model of market in [27], [28] and [29]. The distance between two time steps is defined by a hyper-parameter d . At each time step t , the pricing agent observes m -dimensional state vector s i,t ∈ S ⊆ R m describing the state of product i , and takes an action a i,t . Then the agent receives the reward signal r i,t ∈ R for that action as well as the observation for the new state s i,t +1 . These four elements form the transition ( s i,t , a i,t , r i,t , s i,t +1 ) , which could be simplified as ( s, a, r, s ′ ) . For markdown pricing, the supply is limited, so the pricing process for certain product reaches its end if it is out of stock. For daily pricing application, the supply is regarded as unlimited.

Intuitively we want a small d to make pricing actions reacting in time or archiving continuous pricing. However, precisely describing the change of the environment may need a certain time period for observing. Changing price rapidly could also break price image for the product and even cause credit issue on E-commerce platform. Since our experiments would change the prices online, we set this period carefully after discussions with professional pricing managers. In the rest of this work, pricing period d is settled as one day. Therefore, time step t represents day t .

State space. Here in our model, each product i is priced separately, with four different groups of features at time step t to describe the state s i,t : price features, sales features, customer traffic features and competitiveness features. Price features contain the actual payment for this product, the discount rate, the coupons, etc. Sales features contain the sales volume, revenue, etc. Customer traffic features contain the time the page of the product i has been viewed, the number of unique visitors viewed the product (UV), the number of buyers for product i , etc. The comments and the states of the similar products contribute to competitiveness features.

Action space. We also define the action space for each product i separately. We use the maximum price P i,max and the minimum price P i,min of the product i during a certain number of periods in the history to define the upper bound and lower bound. It assumes that the pricing framework should not output a price out of this area. The pricing space could be discrete or continuous for different applications. When it is discrete, the pricing space is divided from P i,min to P i,max into K separated areas, as K discrete actions. The price p i,k ∈ [ P i,min + ( P i,max -P i,min ) K · ( k -

Revenue

0.9

0.8

0.7

0.5

0.4

0.3

0.2

0.0

5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100

Price level

1.0

0.9

0.8

0.7

06

0.5

0.4

0.3

0.2

0.1

Figure 2: (a) and (b) show the average of re-scaled revenue for 1400 SKUs of jewelries and 4800 SKUs of candies respectively in different price levels (with the highest revenue re-scaled to 1). Price level 1 stands for the lowest price in three months while price level 100 stands for the highest. (c) and (d) show the average revenue conversion rate for jewelries and candies respectively (with the highest revenue conversion rate re-scaled to 1).

<!-- image -->

1) , P i,min + ( P i,max -P i,min ) K · k ) will be regarded as the k th ( k ∈ [1 , K ] ) pricing action for product i . And then each action a i,t ∈ { 1 , 2 , ..., K } stands for a price range. When it is continuous, action a i,t ∈ A ⊆ R represents an specific price.

Reward function. We compared revenue with respect to revenue conversion rate to define the immediate reward r i,t . The customer traffic could fluctuate drastically and then dominate the total revenue. Therefore, there may not be a clear and explainable relationship between price and revenue like in traditional retail industry. On the other hand, mapping price with revenue could be regarded as a special case of our research assuming the customer traffic is steady. But on E-commerce platform, there are links between prices and revenue conversion rates, r i,t = revenue i,t /uv i,t , where revenue i,t and uv i,t represent the total revenue of product i and the number of unique visitors viewed the product i between time step t and t -1 respectively. In some part of this work, we also use profit conversion rate, dividing profit i,t the profit of product i by uv i,t , if we have the knowledge of the inventory cost. To prove this idea, we analyze different categories of products about their distribution of revenues and conversion rates on different price levels. Here, we demonstrate the results for over 1400 SKUs of jewelries and over 4800 SKUs of candies selling on Tmall.com in Figure 2.

As shown in Figure 2 that, the revenue conversion rate is more concave than revenue itself. In the field experiments, using revenue conversion rate as reward function works well in markdown pricing application when 1) most of the markdown products are low-sales-volume luxuries, having low but sensitive revenue conversion rates with prices; 2) the revenue conversion rates for these products are relatively steady; 3) there is a very clear and accurate stock determining the length of the pricing process, and the total discounted revenue conversion rate is finite. However, in another pricing application in this work, daily pricing for fast moving customer goods (FMCGs), the supply is adequate and the stock could be regarded as unlimited. A clear end point for each pricing process could hardly be defined. More importantly, the average sales volumes for these FMCGs are much higher than the luxuries and the revenue conversion rates are very unstable. In this case, despite the relationship between prices and revenue conversion rates may not keep steady. Another investigation for two different kinds of products about the trends of their price level with revenue conversion rate for 90 days (shown in Figure 3) reveals this phenomena.

We could see from Figure 3 (a) that, for these luxuries, when the price drops, the revenue conversion rate goes up, especially around day 25, 45, 65 and 85. However, for FMCGs in (b) there is no such relationship, but the revenue conversion rate fluctuates with an individual frequency. Within this period, the correlation coefficients between price levels and revenue conversion rates are -0.57 and 0.15 for luxuries and FMCGs respectively. Therefore, if the revenue conversion rate is used as reward function for these FMCGs, the convergence of the model could not be guaranteed. Comparing two phenomenon in Figure 2 and Figure 3, we define a different reward function, using the difference of the revenue conversion rates (DRCR) in Eq.(1):

<!-- formula-not-decoded -->

where τ represents the length of the time to compare the revenue conversion rates. The idea behind this definition is that, we hope to give the agent an positive signal, if it relatively raise the revenue conversion rate with its pricing action. Our experiments indicate that, this definition of reward function solves the convergence problem in FMCGs daily sales, while could also work out fine in markdown pricing.

Revenue

N

Revenue conversion rate

08

0.4

0.3

0.2

0.1

Revenue conversion rate

1.0

0.7

0.6

0.5

0.4

0.3

0.2

Revenue conversion rate/price level

1.0

0.9

0.8

0.7

0.6

0.5

0.4

0.3

0.2

0.1

0.0

--

5

Luxuries

Price level

Revenue conversion rate

10 15 20 25 30 35 40 45 50

1.

0.

0.

.8

0.7

0.

. б

Figure 3: The average of re-scaled revenue conversion rate and price level for 2000 SKUs of luxuries (in sub-figure (a)) and 4000 SKUs of FMCGs (in sub-figure (b)) respectively through 90 days. Here 0 stands for the lowest price level through 90 days and 1 stands for the highest. The revenue conversion rate is re-scaled by dividing the maximum value.

<!-- image -->

## 2.2 Discrete pricing action models

To solve the dynamic pricing MDP we defined above, we first use Q-learning ([30]) to find the optimal pricing policy. Q-learning is a value iteration method to compute the optimal policy. It starts with randomly initialed Q value and recursively iterates using the transitions t = ( s, a, r, s ′ ) to get the optimal Q ∗ as well as the optimal policy:

<!-- formula-not-decoded -->

where α ∈ (0 , 1] is the learning rate and γ is the discount factor. Due to the high dimension of the state space, we use a deep network to map the Q-values from the state space, which follows the idea of deep Q-networks (DQN, [25]). To update the action value network, a one-step off-policy evaluation is used to minimize the loss function:

<!-- formula-not-decoded -->

where D is a distribution over transitions contained in a replay buffer working, θ are the parameters of the Q-network and θ ′ are the network parameters used to compute the target. The target network parameter θ ′ are only updated with the Q-network parameters every C steps.

## 2.3 Continuous pricing action model

Pricing on discrete action space encounters an obvious conflict setting the number of discrete actions, i.e. the hyperparameter K in this work. If K is too small, different prices in a large pricing area will be regarded as the same price action for the agent. In return, the discrete output action would also be a large pricing area, making the policy imprecise. On the other hand, if K is too large, a lot of actions will not be explored in the history and the exploration in the future could also be inefficient and expensive. Therefore, we consider to build up model pricing on continuous space to output an exact price instead of a pricing area. We apply the actor-critic algorithm ([31]), which combines value-iteration methods and policy-iteration methods and have been proposed and performed well on different problems([32], [33]). Specifically, we apply deep deterministic policy gradient (DDPG, [34]) as our actor-critic method. The actor part of this model maintains a policy-network π ( a | s ; θ µ ) , taking the environment state as its input and output continuous actions a = µ θ ( s ) . And the critic part takes both the state and action as input and estimates the action value function Q ( s, a | θ Q ) . θ µ and θ Q are the network parameters. So, the loss function would be:

<!-- formula-not-decoded -->

In this work, we also apply the experience replay and separate target network techniques, so θ µ ′ and θ Q ′ are the targetnetwork parameters for actor and critic respectively. And it takes the gradient of Q-value to update the policy-network:

<!-- formula-not-decoded -->

The idea is to adjust the parameters θ µ of the policy-network in the direction of the performance gradient, underlying these algorithms is the policy gradient theorem ([35]).

Revenue conversion rate/price level

0

FMCGS

## 2.4 Pre-training

If we directly apply reinforcement learning algorithms to E-commerce dynamic pricing, the cold start problem occurs, which starts with very poor performance and may cause capital loss. In some other areas like robotics [36] and games [37], there may be accurate simulators, within which the agent could learn policy. However, there is no such a simulator for dynamic pricing problem. Instead, we have enough data of the environment and the pricing decisions made by some previous controllers. These controllers could be some specialists or some rules, and some of their pricing decisions may be reasonable. The records of their decision facing the environment could be regarded as demonstrations, which has been proved to be effective for pre-training the agent from [38].

As mentioned before, the pricing actions were taken periodically. Thus, the state of the environment as well as the rewards could be represented by the data collected within these periods between actions. Therefore, we form the demonstration in tuples &lt; s t , a t , r t , s t +1 &gt; and use them for pre-training. Specifically, we refer the ideas of Deep Q-learning from Demonstration (DQfD) [38] as our pre-training method for DQN and Deep Deterministic Policy Gradient from Demonstration (DDPGfD) [39] for DDPG.

## 2.5 Offline evaluation

As we discussed above, we need to evaluate the model during pre-training before pricing online. We will introduce the methodology for offline evaluation in this part, while the methods for online evaluation will be discussed in detail in section 3.2. We first use the latest T periods of records to form tuples &lt; s t , a t , r t , s t +1 &gt;,t ∈ [1 , T ] . And then, we divide these tuples into two parts: the first D tuples, D &lt; T , will be used for pre-training, where t ∈ [1 , D ] . And for D &lt; t &lt; T , the tuples will be used for evaluation. The idea is that we sum the reward r t only if the action a t is close to the output of the policy a t -glyph[epsilon1] &lt; π ( s t ) &lt; a t + glyph[epsilon1] . The detail of the evaluate algorithm is sketched in Algorithm 1.

```
input : T > 0 : number of demonstration tuples for evaluation. π : the policy to evaluate; output : R π : average reward form policy π ; begin R ← 0 , N ← 0 ; for step t ∈ { 1 , 2 , ..., T } do repeat Get next tuple < s t , a t , r t , s t +1 > ; until a t -glyph[epsilon1] < π ( s t ) < a t + glyph[epsilon1] ; R ← R + r t , N ← N +1 ; if N > 0 then R π = R/N else R π = 0
```

Algorithm 1: Policy evaluation with demonstrations.

## 3 Experimental results

In this part, we first introduce the offline experiments, using historical data from Tmall.com. Then we introduce the field experiments. During these field experiments, we changed the selling prices for products on Tmall.com in a markdown scenario and a daily basis starting from July 2018.

## 3.1 Offline experiments

For offline experiments, we carefully chose over 40,000 SKUs of FMCGs from hundreds of different categories. We used 60 days selling records to form about 2,400,000 tuples. The first 59 days' records were used for pre-training and the last day's for evaluation. We first evaluate two different reward functions we discussed above, revenue conversion rate (RCR) and the difference of revenue conversion rate (DRCR). We set α = 0 . 01 , τ = 1 and gradually increased γ from 0.5 to 0.99. We use DQN model with K = 100 for this evaluation and the result is shown in Figure 4a. We could see that,the policy with DRCR as reward function performs better and steadier than using RCR.

Reward

0.6

0.5

0.4

0.3

Rescaled DRCR

0.2

0.1

0.0

-0.1

-0.2

-0.3]

10

100

Reward

Reward

0.5

0.4

0.3

0.2

Then we evaluate DQN model with different parameter K . we set K = 10 , 20 , 50 , 100 and 200 . The result is shown in Figure 4b. We could see that, for our experiment setup, DQN with K = 100 performs best. For DQN and DDPG comparison, we set glyph[epsilon1] = 0 . 05 and K = 100 . The result is shown in Figure 4c. DDPG performs better than DQN after pre-trained with a certain number of demonstrations. -0.1 10 100 300 400 200 Time step (x1000) 500

Group one

Group two

Figure 4: (a) Offline evaluation for different reward functions. (b) Offline evaluation for DQN with different K values. K stands for the number of output discrete actions for DQN models. (c) Offline evaluation for DQN and DDPG comparison.

<!-- image -->

## 3.2 Online experiments

To evaluate different pricing policies online, we need to eliminate the fluctuation of the market environment, causing by seasonality of demand or marketing operations. Since online A/B testing is not suitable for E-commerce platform as we discussed, we follow the idea of the difference-in-differences (DID) techniques ([40],[41]), to evaluate the effects of different policies on relevant outcome. We first defined simi-products on E-commerce platform for our experiment, the products with the same brand, same category and similar selling behaviours. We have found that, using the same pricing policy, two groups of simi-products could have very close DRCR with certain parameter τ , even if two groups have different total revenues or revenue conversion rates. We set τ = 365 here, making DRCR also represents the year-on-year growth of the revenue conversion rates, in order to eliminate the season fluctuation and influence from annual marketing strategies. Figure 5 demonstrates that, two different groups of simi-products pricing by same policy have very close DRCR within 30 days. It is worth mentioning that, the DRCR with τ = 365 here is an index for DID evaluation, rather than a reward function.

Figure 5: The DRCR for different groups of simi-products under same pricing policy within 30 days. The average of DRCR from group one is re-scaled to 1.00 and the average of group two becomes 0.99 after re-scaling.

<!-- image -->

Pricing for markdown season. During the first part of online experiment, our agent prices 500 SKUs of luxury products (mainly handbags and clothes). Each product has around 10 items in stock and the aim of this markdown season is to maximize the total profit conversion rate. Therefore we use the profit conversion rate as our reward function, r i,t = profit i,t /uv i,t . Another set-up in this online experiment is that, the agent is required to output a discrete value form 1 to 9 representing the discount rate from 10% to 90% in markdown season. So we applied discrete action model,

0.2

rate

Rescaled revenue conversion

1.0

0.9

0.8

0.7

0.6

0.5

0.4

0.3

0.2

0.1

0.0

" g

4

6

8

0.6

0.5

0.4

0.3

0.2

0.1

0.0

088.

DQN with K = 9 here for these luxury products markdown pricing. We set τ = 1 for reward function and D = 90 , α = 0 . 01 and γ = 0 . 99 as pre-training parameters.

There is another group containing 2000 SKUs of simi-products pricing manually in the same period, which is regarded as the benchmark group for DID evaluation. The result of the experiment is demonstrated in Figure 6a and Figure 6b.

-0.7 -

-0.8

-0.9

<!-- image -->

10

Figure 6: (a) Re-scaled revenue conversion rate from products priced by DQN and managers manually. From day 1 to day 15, DQN group and manual group both had a revenue conversion rate of 0.04. From day 16 to day 30, DQN group achieved a revenue conversion rate of 0.22 on average and manual group's average conversion rate is 0.16 for this period (with day 16 manual group's revenue conversion rate re-scaled to 1). (b) Re-scaled profit conversion rate from products priced by DQN and managers manually. From day 1 to day 15, the average profit conversion rate for both DQN group and manual group are 0.06. From day 16 to day 30, DQN group obtained an average profit conversion rate of 0.16, while manual group's average profit conversion rate dropped to -0.04 (with day 16 manual group's profit conversion rate re-scaled to -1)

In the first 15 days of this online experiment, these products were in their daily prices. It shows that two groups of products performs alike, which confirms our investigation of the simi-product above. Then the markdown season started at the 16th day and lasted for 15 days. We could see that both groups boosted the revenue conversion rate at the beginning of the markdown season (day 16). Then in day 21 and 22 as well as day 25 to 30, DQN group successfully pull up the revenue conversion rate again beating manual group. Comparing with the profit conversion rate in Figure 6b, it is more clear that, manual pricing method pull up the revenue by setting the prices lower than the cost, causing negative profit. DQN pricing policy successfully keeps positive profit at most of the markdown season. It is interesting that, at day 26, DQN also priced the products to negative profit rate, and got negative immediate reward. But this action pulled up both the revenue and profit conversion rate in the rest of the markdown season, achieving good total profit.

Pricing for daily sales. To verify the effectiveness of reward function in Eq.(1), we set up another online experiment, to price supply unlimited FMCGs. The whole experiment contains two parts and each part lasted for 30 days. We chose 1000 SKUs of FMCGs from 200 different categories selling on Tmall.com as our experiment group (group two in Figure 5). As for DID evaluation, we matched 3000 SKUs of simi-products also selling on the platform as the control group (group one in Figure 5). The control group was pricing by some managers. Due to the huge amount of the products to be priced manually, their policy is usually set up a price at the beginning of the month, and then change some of them if their revenues are below their expectations.

We set K = 100 for DQN model and set α = 0 . 01 and γ = 0 . 99 . But as the FMCGs' behaviour changes more rapidly than the luxury products', we set D = 30 during pre-training, shorter than we set in markdown pricing experiment. In the first 30 days of experiment, we investigated the behaviour of DQN pricing policy by comparing it with the control group. DRCR within 30 days for two groups are shown in Figure 7a. We could see that, DQN group outperformed control group.

Then we divided the experiment group randomly into two groups for testing DQN and DDPG, while kept the group one still as the control group. The result is shown in Figure 7b. During this part of experiment, we encountered some daily management activities (with some coupons given out). These activity only influenced less than 10% of the total revenue. As the relationship between simi-products we mentioned above could still be observed, we did not stop the

1-----

DQN

• • Manual|

DON

Rescaled DRCR

5

10

Figure 7: (a) The DRCR for DQN pricing group and manual pricing group within 30 days. The averages of DQN groups is 5.10, with the average of the control group re-scaled to 1.00. (b) Comparing the DRCR for DQN pricing group, DDPG pricing group and manual pricing group. The averages of DDPG and DQN pricing groups are 6.07 and 5.03 respectively with the average of the control group re-scaled to 1.00.

<!-- image -->

field experiment but kept observing the behavior of two models encountered the fluctuation of the environment. Both DRL methods outperformed control group while DDPG performed better.

## 4 Conclusion and discussion

In this work, we proposed a deep reinforcement learning framework for dynamic pricing on E-commerce platform. We defined the pricing process as a Markov Decision Process and then defined the state space, discrete and continuous action space, and different reward function for different pricing applications. We applied our methods for pricing policies and applied to online pricing in real time. We first apply deep reinforcement learning method for pricing products in a markdown season. The field experiment showed that it outperformed the manual markdown pricing strategy. As daily pricing for FMCGs, we design a systematic mechanism for online pricing policy evaluation, to address the legal issue of A/B testing for different pricing strategies. We showed that pricing policies from DDPG and DQN outperformed other pricing policies significantly.

This work is the first to use deep reinforcement learning for dynamic pricing problem on E-commerce platform, pricing thousands of SKUs of products in real-time. In this work, there are a few constraints could be removed. First, our pricing framework trains each product separately. As a result, the low-sales-volume products may not have sufficient training data. This could be solved by clustering similar products and using transfer learning to price the products in the same cluster. Meta-learning may also help for this problem. Second, our framework outputs pricing policy for each product separately. However, sometimes we hope to price different products together to form certain marketing strategies. This may be solved by a combinatorial action space. Third, in our pricing framework, we take only the features related to the products to describe the environment state. In the future, we would try to take more kinds of features into consideration for pricing under more specific scenarios, e.g., promotion pricing or membership pricing.

## Acknowledgement

We would like to thank Sentao Miao, Huiqiang Mao, Miaolan Xie, Yangsheng Ji and others at Alibaba Supply Chain Platform for their helpful comments.

## References

- [1] Lawrence R Weatherford and Samuel E Bodily. A taxonomy and research overview of perishable-asset revenue management: Yield management, overbooking, and pricing. Operations research , 40(5):831-844, 1992.

Rescaled DRCR

- [2] Kalyan T Talluri and Garrett J Van Ryzin. The theory and practice of revenue management , volume 68. Springer Science &amp; Business Media, 2006.
- [3] MKeith Chen and Michael Sheldon. Dynamic pricing in a labor market: Surge pricing and flexible work on the uber platform. In EC , page 455, 2016.
- [4] Felipe Caro and Jérémie Gallien. Clearance pricing optimization for a fast-fashion retailer. Operations Research , 60(6):1404-1422, 2012.
- [5] JACK Nicas. Now prices can change from minute to minute. Wall Street Journal , 2015.
- [6] Le Chen, Alan Mislove, and Christo Wilson. An empirical analysis of algorithmic pricing on amazon marketplace. In Proceedings of the 25th International Conference on World Wide Web , pages 1339-1349. International World Wide Web Conferences Steering Committee, 2016.
- [7] James Davidson, Benjamin Liebald, Junning Liu, Palash Nandy, Taylor Van Vleet, Ullas Gargi, Sujoy Gupta, Yu He, Mike Lambert, Blake Livingston, et al. The youtube video recommendation system. In Proceedings of the fourth ACM conference on Recommender systems , pages 293-296. ACM, 2010.
- [8] Roberto Maestre, Juan Duque, Alberto Rubio, and Juan Arévalo. Reinforcement learning for fair dynamic pricing. arXiv preprint arXiv:1803.09967 , 2018.
- [9] Jeffrey O Kephart, James E Hanson, and Amy R Greenwald. Dynamic pricing by software agents. Computer Networks , 32(6):731-752, 2000.
- [10] Byung-Gook Kim, Yu Zhang, Mihaela Van Der Schaar, and Jang-Won Lee. Dynamic pricing and energy consumption scheduling with reinforcement learning. IEEE Transactions on Smart Grid , 7(5):2187-2198, 2016.
- [11] CVL Raju, Y Narahari, and K Ravikumar. Reinforcement learning applications in dynamic pricing of retail markets. In E-Commerce, 2003. CEC 2003. IEEE International Conference on , pages 339-346. IEEE, 2003.
- [12] David Vengerov. A gradient-based reinforcement learning approach to dynamic pricing in partially-observable environments. 2007.
- [13] Michael Schwind and Oliver Wendt. Dynamic pricing of information products based on reinforcement learning: A yield-management approach. In Annual Conference on Artificial Intelligence , pages 51-66. Springer, 2002.
- [14] Arnoud V den Boer. Dynamic pricing and learning: historical origins, current research, and new directions. Surveys in operations research and management science , 20(1):1-18, 2015.
- [15] Antoine Augustin Cournot. Researches into the Mathematical Principles of the Theory of Wealth . Macmillan, 1897.
- [16] G. C. Evans. The dynamics of monopoly. The American Mathematical Monthly , 31(2):77-83, 1924.
- [17] Bardia Kamrad, Shreevardhan S Lele, Akhtar Siddique, and Robert J Thomas. Innovation diffusion uncertainty, advertising and pricing policies. European Journal of Operational Research , 164(3):829-850, 2005.
- [18] Guillermo Gallego and Garrett Van Ryzin. Optimal dynamic pricing of inventories with stochastic demand over finite horizons. Management science , 40(8):999-1020, 1994.
- [19] Dimitris Bertsimas and Georgia Perakis. Dynamic pricing: A learning approach. In Mathematical and computational models for congestion charging , pages 45-79. Springer, 2006.
- [20] Vivek F Farias and Benjamin Van Roy. Dynamic pricing with a prior on market response. Operations Research , 58(1):16-29, 2010.
- [21] J Michael Harrison, N Bora Keskin, and Assaf Zeevi. Bayesian dynamic pricing policies: Learning and earning under a binary prior distribution. Management Science , 58(3):570-586, 2012.
- [22] Omar Besbes and Assaf Zeevi. Dynamic pricing without knowing the demand function: Risk bounds and near-optimal algorithms. Operations Research , 57(6):1407-1420, 2009.
- [23] Omar Besbes and Assaf Zeevi. On the (surprising) sufficiency of linear models for dynamic pricing with demand learning. Management Science , 61(4):723-739, 2015.
- [24] Zizhuo Wang, Shiming Deng, and Yinyu Ye. Close the gaps: A learning-while-doing algorithm for single-product revenue management problems. Operations Research , 62(2):318-331, 2014.
- [25] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature , 518(7540):529, 2015.

- [26] David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. Nature , 550(7676):354, 2017.
- [27] Erich Kutschinski, Thomas Uthmann, and Daniel Polani. Learning competitive pricing strategies by multi-agent reinforcement learning. Journal of Economic Dynamics and Control , 27(11-12):2207-2218, 2003.
- [28] Ananth Madhavan. Market microstructure: A survey. Journal of financial markets , 3(3):205-258, 2000.
- [29] Maureen O'hara. Market microstructure theory , volume 108. Blackwell Publishers Cambridge, MA, 1995.
- [30] Christopher John Cornish Hellaby Watkins. Learning from delayed rewards . PhD thesis, King's College, Cambridge, 1989.
- [31] Ian H Witten. An adaptive optimal controller for discrete-time markov environments. Information and control , 34(4):286-295, 1977.
- [32] Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International conference on machine learning , pages 1928-1937, 2016.
- [33] Kyriakos G Vamvoudakis and Frank L Lewis. Online actor-critic algorithm to solve the continuous-time infinite horizon optimal control problem. Automatica , 46(5):878-888, 2010.
- [34] Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971 , 2015.
- [35] Richard S Sutton, David A McAllester, Satinder P Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Advances in neural information processing systems , pages 1057-1063, 2000.
- [36] Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuomotor policies. The Journal of Machine Learning Research , 17(1):1334-1373, 2016.
- [37] David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. nature , 529(7587):484, 2016.
- [38] Andrew Sendonaris and COM Gabriel Dulac-Arnold. Learning from demonstrations for real world reinforcement learning. arXiv preprint arXiv:1704.03732 , 2017.
- [39] Matej Vecerík, Todd Hester, Jonathan Scholz, Fumin Wang, Olivier Pietquin, Bilal Piot, Nicolas Heess, Thomas Rothörl, Thomas Lampe, and Martin A Riedmiller. Leveraging demonstrations for deep reinforcement learning on robotics problems with sparse rewards. CoRR, abs/1707.08817 , 2017.
- [40] Marianne Bertrand, Esther Duflo, and Sendhil Mullainathan. How much should we trust differences-in-differences estimates? The Quarterly journal of economics , 119(1):249-275, 2004.
- [41] Alberto Abadie. Semiparametric difference-in-differences estimators. The Review of Economic Studies , 72(1):1-19, 2005.