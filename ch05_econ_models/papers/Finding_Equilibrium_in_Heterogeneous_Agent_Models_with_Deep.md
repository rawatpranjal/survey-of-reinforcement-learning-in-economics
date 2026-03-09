## ANALYZING MICRO-FOUNDED GENERAL EQUILIBRIUM MODELS WITH MANY AGENTS USING DEEP REINFORCEMENT LEARNING

Michael Curry ∗ University of Maryland curry@cs.umd.edu

## Alexander Trott, Soham Phade, Yu Bai, Stephan Zheng

Salesforce Palo Alto, CA (atrott,sphade,yu.bai,stephan.zheng)@salesforce.com

## ABSTRACT

Real economies can be modeled as a sequential imperfect-information game with many heterogeneous agents, such as consumers, firms, and governments. Dynamic general equilibrium (DGE) models are often used for macroeconomic analysis in this setting. However, finding general equilibria is challenging using existing theoretical or computational methods, especially when using microfoundations to model individual agents. Here, we show how to use deep multi-agent reinforcement learning (MARL) to find glyph[epsilon1] -meta-equilibria over agent types in microfounded DGE models. Whereas standard MARL fails to learn non-trivial solutions, our structured learning curricula enable stable convergence to meaningful solutions. Conceptually, our approach is more flexible and does not need unrealistic assumptions, e.g., continuous market clearing, that are commonly used for analytical tractability. Furthermore, our end-to-end GPU implementation enables fast real-time convergence with a large number of RL economic agents. We showcase our approach in open and closed real-business-cycle (RBC) models with 100 worker-consumers, 10 firms, and a social planner who taxes and redistributes. We validate the learned solutions are glyph[epsilon1] -meta-equilibria through best-response analyses, show that they align with economic intuitions, and show our approach can learn a spectrum of qualitatively distinct glyph[epsilon1] -meta-equilibria in open RBC models. As such, we show that hardware-accelerated MARL is a promising framework for modeling the complexity of economies based on microfoundations.

## 1 INTRODUCTION

Real-world economies can be modeled as general-sum sequential imperfect-information games with many heterogeneous agents (Mas-Colell et al., 1995), such as consumer-workers, firms, and governments (or other social planners). Dynamic general equilibrium models (DGE) are workhorse models that describe the economic incentives, interactions, and constraints of these agents, which are often assumed to be rational. 1 In particular, we are interested in DGE models with a large number of heterogeneous, strategic agents , based on appropriate microfoundations Archibald et al. (1970); Smets and Wouters (2007).

By finding the strategic equilibria in such games, one can study macroeconomic outcomes, such as productivity, equality, and growth (Heer and Maussner, 2009). However, at this scale and gametheoretic complexity, existing theoretical and computational methods often struggle to find the

∗ Research conducted while Michael Curry was an intern at Salesforce.

1 One can also model more human-like behavior through instances of bounded rationality , e.g., agents that act suboptimally or whose affordances or objectives encode cognitive biases or limits (Kahneman, 2003). In this work, we focus on rational agents, but our framework can be generalized to using boundedly rational agents.

strategic equilibria Nisan et al. (2007); Bai et al. (2021). For a detailed exposition on DGE models and solution methods related to our work, see Section 2. We emphasize that we focus on the methodological challenge of finding equilibria, rather than the question of what constitute appropriate microfoundations.

In this work, we focus on reinforcement learning as a powerful and flexible methodological framework to analyze DGE models with many agents . We propose using deep multi-agent reinforcement learning (MARL) Sutton and Barto (2018) as a constructive solution to explicitly find their (approximate) equilibria. Using MARL provides many benefits: 1) large-scale deep RL has proven capable of finding (near-)optimal behavioral policies in complex multi-agent games in various domains Vinyals et al. (2019a); OpenAI (2018); Silver et al. (2017); 2) RL is flexible: it has few analytical requirements on the structure of the game, e.g., it can optimize policies for any scalar objective, e.g., consumer utility function or social welfare, which does not need to be differentiable; and 3) it can optimize rich behavioral models, e.g., deep neural networks, that can imitate complex human behaviors, e.g., given multi-agent behavioral data Zheng et al. (2016); Zhan et al. (2018). As such, deep MARL holds promise as a framework to model macroeconomic outcomes based on microfoundations.

Economic Interactions Pose Learning Challenges. Although RL offers many conceptual benefits for economic modeling, training each agent in a DGE model independently using RL often fails due to economic interactions. This is because economic interactions introduce mechanism design problems between agents. For example, firms choose how to set prices when interacting with a population of consumers. Furthermore, a consumer's purchasing power changes when firms change wages or governments change taxes.

As such, economic interactions imply that the actions of one economic agent can dramatically change the reward function and constraints (constituting a mechanism) of other agents. Because RL agents learn through exploration , agents sample (sub)-optimal policies during training, which may overly distort the rewards or constraints of other RL agents. As a result, RL agents that learn independently (and do not account for the learning process of other agents) often fail to learn non-trivial behaviors in DGE models. This non-stationary learning dynamic becomes especially challenging for DGE models with a large number of heterogeneous RL agents.

Moreover, which equilibrium agents converge to ('equilibrium selection') may depend on, e.g., the 1) world dynamics, 2) initial conditions and policies, 3) learning algorithm, and 4) policy model class. Our framework allows us to study how these factors relate to which equilibria can be found.

## 1.1 OUR CONTRIBUTIONS

To address these challenges, we show how to effectively apply MARL to macroeconomic analysis:

1. To enable stable convergence of MARL with many agents, we generalize the MARL curriculum learning method from Zheng et al. (2020) to structured curricula that train multiple agent types across multiple phases. This approach yields non-trivial solutions more stably compared to using independent training of RL agents, which often fails. We also show that an RL social planner can improve social welfare vs fixed baselines.
2. We show that our MARL approach can explicitly find local glyph[epsilon1] -equilibrium strategies for the meta-game over agent types, without approximating the DGE model. Previous solutions only could do so implicitly or for approximations of the DGE dynamics. Here, the meta-game equilibrium is a set of agent policies such that no agent type can unilaterally improve its reward by more than glyph[epsilon1] (its best-response ). Furthermore, we learn a spectrum of solutions in open RBC economies,
3. Our approach yields new economic insights . Our approach is more flexible and can find stable solutions in a variation of real-business-cycle (RBC) models (Pierre Danthine and Donaldson, 1993), a family of DGE models. We find solutions in closed and open RBC economies, the latter has a price-taking export market. After training, the behavior of our agents shows sensible economic behavior - for example, negative correlations of prices with consumption, and positive correlation of wages with hours worked, over a range of possible outcomes. Within each outcome, firms and consumers display different strategies in response to their conditions - for example, firms with capital-intensive production functions invest in more capital.

Enabling Economic Analysis with Many RL Agents. We are particularly interested in DGE models with a large number of heterogeneous strategic agents. To enable MARL at this scale, we ran both simulation and RL training on a GPU using the WarpDrive framework (Lan et al., 2021). WarpDrive accelerates MARL by orders of magnitude, e.g., by avoiding copying data unnecessarily between CPU and GPU. In effect, this enables MARL to converge in hours (rather than days). Such system design and implementation choices are key to enable richer economic analysis, without resorting to approximations like representative agents Kirman (1992); Hartley and Hartley (2002) or mean-field methods Yang et al. (2018). In contrast, prior computational work often was limited to training a small number of independent, strategic agents.

Using hardware acceleration to train machine learning models at larger scales has repeatedly resulted in qualitatively new capabilities in machine learning, including some of the most spectacular results in language modeling Brown et al. (2020) and computer vision Krizhevsky et al. (2012) in the past decade. An essential part of each of those breakthroughs was optimizing system design and implementation. While we don't claim to match the impact of these results, we want to emphasize that we see improved engineering to enable economic modeling at greater complexity and scale as a key contribution. In the context of economic modeling, being able to learn and analyze across a spectrum of possible outcomes and solutions is key for policymaking Haldane and Turrell (2019).

Code. The code for this paper will be publicly available for reproducibility.

## 2 PRIOR WORK

We now discuss how our MARL approach contrasts with or complements prior work.

## 2.1 THE ECONOMICS OF DGE MODELS

DGE models study the behaviors of and interactions between consumers, firms, and perhaps a government. The markets for goods, labor, and capital are often assumed to be competitive, with prices set to clear the markets at each time step. Typically, consumers may work, save, and consume, and balance these choices to maximize their total time-discounted future utility. In other words, their decisions represent the solution to a Bellman equation. There are DGE models without Smets and Wouters (2007) or with Kaplan et al. (2018) heterogeneity among agents.

Variations of DGE models also consider open economies with trade and flows with a (large) external economy; accordingly, closed economies lack such an external connection. Typically, they have the same structure of market-clearing for wages, prices, and capital, but also allow consumers to invest in foreign capital Mendoza (1991); De Groot et al. (2019). We consider both closed and open economies: we choose to model the open case by giving firms access to an export market with an inelastic price for goods.

## 2.2 MACROECONOMIC MODELING AND MICROFOUNDATIONS

Microfoundations. Lucas and Sargent (1981) argued for an approach to macroeconomic modeling based on microfoundations Smets and Wouters (2007). Broadly construed, microfoundations analyze macroeconomics through models with individual self-interested strategic agents; macroeconomic outcomes are then the aggregation of their behaviors. In particular, microfoundations address the Lucas critique, which points out the issue of basing macroeconomic policy on historical data. That is, historically observed aggregate patterns of behavioral responses to changes in economic policy may not extrapolate to the future, as individuals continuously adapt their behavior to the new policy.

Conceptual Limitations of Prior DGE models. Stiglitz (2018) discusses several limitations of previous DGE models, including the use of inappropriate microfoundations. For instance, agents are often assumed to be rational, have stylized decreasing marginal utilities, and maximize their expected value in the presence of uncertainty ( the rational expectations assumption ), in the face of perfectly clearing markets. However, modern DGEs often lack agent models based on behavioral economics, for example. We emphasize that our use of MARL is agnostic to the specifications of the individual agents, which may be rational or boundedly rational.

Moreover, while DGE models are based on microeconomic theory for individuals, they often do not actually model the large number of agents as individuals. Rather, many models use representative agents as a proxy for the aggregate behavior of, e.g., all consumers. These and other simplifying, but restrictive assumptions make it possible to find analytic solutions, but lead to unrealistic outcomes, e.g., no trading. Moreover, the use of representative agents does not lead to uniqueness and stability of equilibria Kirman (1992), a popular desideratum.

Challenges of Solving DGE Models. Existing analytical and computational methods often struggle to find explicit DGE equilibria, as DGE models are typically highly nonlinear. More generally, enumerating and selecting equilibria in general-sum games is an unsolved challenge (Bai et al., 2021).

Analytical work often simplifies the DGE model's dynamics Lucas and Sargent (1981), e.g., through linearization. However, a linearized DGE model may have fewer or different equilibria compared to the full DGE model, and a linearized solution may prefer one equilibrium arbitrarily when many are possible. Furthermore, linearization may only be a good approximation around steady-state equilibria and not be valid in the presence of large shocks, e.g., when agents change their behavior dramatically in time (Stiglitz, 2018; Boneva et al., 2016; Atolia et al., 2010).

Formal methods, e.g., backwards induction and other dynamic programming methods, often don't yield explicit solutions and only implicitly characterize optimal policies (Stokey et al., 1989). For example, in taxation, one can analyze the distortion in consumer savings or capital, the asymptotic behavior of optimal taxes in the far future, or the tax rate on the highest-skill agent (Golosov et al., 2011; Acemoglu et al., 2010). However, these methods generally cannot find the full tax policy explicitly or only describe part of it.

Computational Methods and Agent-Based Models. A related body of work has studied agentbased models (ABM) Bonabeau (2002); Sinitskaya and Tesfatsion (2015); Haldane and Turrell (2019). ABMs may include a much larger range of heterogeneous agents (compared to typical DGE models), with possibly similar microfoundations, but whose affordances and dynamics are often highly stylized. Through repeated simulation, ABMs typically study the emergent (macroeconomic) phenomena given simple behavioral rules for the agents, across a range of parameters. However, these studies mostly do not address questions of finding the equilibria and optimal behaviors. Moreover, computational methods more generally still pose technical and conceptual challenges for economic analysis Judd (1997). Libraries exist to use numerical methods, e.g. to find fixed-point solutions Holden (2017); Mendoza and Villalvazo (2020). However, it can be hard to distill simple or explainable economic principles from numerically-found optimal behaviors and simulations.

## 2.3 REINFORCEMENT LEARNING AND ECONOMICS

Modeling Economic Agents using RL. In the economics literature, Haldane and Turrell (2019) observed that RL is a natural fit for economic modeling. Rational economic agents maximize their total discounted future reward, which is equivalent to the definition of an RL agent. Their optimal behavioral policy (or simply 'policy') are characterized by the Bellman equation. The economics literature traditionally solves such Bellman equations with methods that are only usable in fully-known environments, e.g., value or policy function iteration Coleman (1990); Judd (1992)

In contrast, in the machine learning literature, many gradient-based RL techniques, e.g., REINFORCE Williams (1992), implement a form of approximate dynamic programming without requiring explicit knowledge of every environment state transition. Instead, they only require access to a simulation of the environment and learn through trial-and-error and continuous feedback loops, making them compatible with a larger range of modeling assumptions. Moreover, RL is compatible with modeling policies using deep neural networks, which can model a large class of functions and asymptotically are universal function approximators. Deep neural network policies can learn nonlinear predictive patterns over high-dimensional state-action spaces, while deep value functions can model complex reward function landscapes over long time horizons.

Combining these features has led to significant successes. Notably, deep RL has achieved superhuman performance in high-dimensional, sequential games (Silver et al., 2017), including in multi-agent settings Vinyals et al. (2019b); OpenAI (2018). These results suggest deep MARL can learn (approximate) strategic equilibria in complex settings. Pertinent to our setting, we do not have

to approximate non-linear DGE dynamics, as deep RL has been successful in highly nonlinear environments (Tassa et al., 2018).

Economic Analysis using RL. A small but growing number of works have explored the use of RL for economic analysis, although deep RL is still not commonly used in economics . To our knowledge, our work is the first application of deep multi-agent RL to DGE models where all agents learn.

The AI Economist used two-level RL to design optimal taxes to improve social welfare in spatiotemporal simulations, where both agents and governments use RL policies (Zheng et al., 2020). Two-level RL also yields interpretable economic and public health policies in pandemic simulations (Trott et al., 2021). Danassis et al. (2021) studies harvesters who work in a common fishery and a centralized price setter, and demonstrate that an RL price setter can outperform prices found through market equilibrium on a number of metrics, including social welfare, fairness, and sustainability. Radovic et al. (2021) simulate oil company investments while transitioning away from hydrocarbons; they find that good policies for oil companies involve rapidly investing in renewable energy.

Other contemporary work includes Chen et al. (2021), which studies monetary policy with a single representative household RL agent, and Hill et al. (2021), which learns the value function of consumers in RBC models. These works are more limited, compared to ours. First, they use RL for one agent type only, while the other agents (e.g., firms) use simple and fixed policies. Second, they assume markets always clear at each time step, i.e., prices are manually set to ensure supply and demand are balanced. However, this is an unrealistic assumption and causes slow simulations, requiring solving a nonlinear optimization problem at each timestep.

Perhaps most similar to our work is Sinitskaya and Tesfatsion (2015), which studies an economy with homogeneous consumer-workers and price- and wage-setting firms (but no government). Here, a few consumers and firms learn using tabular Q-learning. Even in this small-scale setting, they observe that learning dynamics can collapse to trivial solutions with no production or consumption, similarly to our work. Furthermore, they partially enforce market clearing by matching supply and demand for labor and goods in a double auction, but do not enforce constraints on labor availability requirements to produce demanded goods. In contrast, we do not enforce market clearing constraints at all, simply rationing goods if there are not enough. Moreover, in our DGE model, consumers and firms are heterogeneous, e.g., firms have different production functions and produce distinct goods.

Finding Equilibria in Games using Machine Learning. Several streams of work have explored using ML techniques to find equilibria in games. Leibo et al. (2017) studied meta-game equilbria in sequential social dilemmas, showing that RL policies can be segmented into those who effectively cooperate or defect. In the domain of imperfect-information games, counterfactual regret minimization has yielded superhuman poker bots (Brown and Sandholm, 2018). Empirical game-theoretic analysis studies equilibria through agent simulations (Wellman, 2006), but is limited to games with interchangable agents with identical affordances (Tuyls et al., 2018) and does not easily scale to settings with heterogeneous agents.

In extensions of deep RL, higher-order gradients (Foerster et al., 2017) and first-order gradient adjustments (Balduzzi et al., 2018) have been studied to promote convergence to non-trivial equilibria. However, these methods make strong assumptions on what agents know, e.g., that agents can see the policy weights of other agents and/or know the full reward function, and may even change the equilibria of the game.

## 2.4 THEORETICAL ANALYSIS

Different Equilibrium Types. DGEmodels may support many forms of equilibria. First, assuming all agents act simultaneously, one may analyze (repeated) Nash equilibria, where no rational agent is incentivized to deviate unilaterally. There may also be asymmetric Stackelberg-like equilibria, where a leader (e.g., the government) acts first (e.g., sets taxes), and the followers (e.g., consumers and firms) respond Zheng et al. (2020). Generally, finding all equilibria of any type in general-sum, sequential, imperfect-information games with many agents is an open challenge (Bai et al., 2021). For instance, with 2 or more (heterogeneous) followers, finding the Stackelberg best-response to a fixed leader requires finding multi-agent equilibria for the followers. This is computationally expensive and there is no known provably convergent algorithm.

Government

• Optimize Social Welfare

Set Income Tax Rates

Set Corporate Tax Rates

Income Tax

Subsidy

Corporate Tax

Consumer-Workers

• Maximize Utility

Firms

• Maximize Profit

Figure 1: RBC model with consumers, firms, and governments. Arrows represent money flow. Consumer-workers earn wages through work and consume goods from firms. They also strategically choose which firm to work for and which basket of goods to buy, but this is not explicitly visualized. Firms produce goods, pay wages, and set a price for their goods. They also invest a fixed fraction of profits to increase capital. The government taxes labor income and firm profits, and redistribute the tax revenue through subsidies to the consumer-workers. Firms can also sell goods to an external export market, which acts as a price-taker that is willing to consume goods at any price.

<!-- image -->

Instead, we use MARL to converge to a stable solution and analyze best-responses to evaluate to what extent it is an equilibrium. Several works have studied the use of RL techniques in such settings (Wang et al., 2019; Trejo et al., 2016; Kamra et al., 2018; Bai et al., 2021), although do not consider the complexity of DGE models.

Theoretical Guarantees for Convergence and Bounds on Approximate Equilibria. Guarantees of convergence for model-free policy optimization are difficult to come by. Some theorems only apply to tabular policies (as in Srinivasan et al. (2018), and where convergence to Nash is not even guaranteed due to a non-sublinear regret bound). Other cases also apply only to very limited families of games, as in Zhang et al. (2019), which deals only with zero-sum linear-quadratic games. We study general-sum games with a large number of agents with neural network behavioral policies. All three of these properties mean that convergence guarantees to an equilibrium are beyond the reach of theory. To test whether a learned strategy profile is at an glyph[epsilon1] -Nash-equilibrium, one can attempt to compute a best-response for each agent and measure glyph[epsilon1] . However, the non-convexity and dimensionality of neural network loss landscapes means that there are no perfect best-response oracles in practice. In our work, we use single-agent RL and gradient-based optimization to see whether individual agents can improve their utility unilaterally and hence measure glyph[epsilon1] . Although the found solutions and best-responses align with economic intuitions, it is beyond the scope of this work to establish theoretical guarantees on RL as a best-response oracle, e.g., deriving upper bounds on glyph[epsilon1] .

## 3 REAL-BUSINESS-CYCLE MODELS

RBCs are a representative DGE model in which consumers earn income from labor and buy goods, firms produce goods using capital and labor, and the government taxes income and profits (Pierre Danthine and Donaldson, 1993), see Figure 1. RBC models are stylized and may not fully describe reality (Summers et al., 1986). However, RBC models are a suitable environment to validate the use of RL, as they feature heterogenous agents with nonlinear economic interactions, making it challenging to find equilibria. Below, we describe the RBC dynamics.

At a high level, our model includes worker-consumers, price-and-wage setting firms, and a government who sets tax rates and redistributes. A key point about our model is that we do not assume as part of the environment that prices and wages are set so that markets clear at each time step - that is, goods may be overdemanded and firms are free to set prices and wages lower or higher than would balance supply and demand. These assumptions are an essential part of the techniques used to derive analytic solutions - avoiding this modeling choice requires using other techniques.

Agent Types. Formally, our RBC model can be seen as a Markov Game (MG) (Littman, 1994) with multiple agent types and partial observability. An MG has finite-length episodes with T timesteps, where each timestep represents a quarter of 3 months. At each timestep t , we simulate

Wages

Consumption

- firms and their goods who use the labor of the worker-consumers to each produce a different good, indexed by i ∈ I ,
- consumers, indexed by j ∈ J , who work and consume goods, and
- a government who sets a tax rate on income from labor and on revenue from selling goods.

Each agent iteratively receives an observation o i,t of the world state s t , executes an action a i,t sampled from its policy π i , and receives a reward r i,t . The environment state s consists formally of all agent states and a general world state. At each timestep t , all agent simultaneously execute actions. However, some actions only apply to the next timestep t +1 . For example, the government sets taxes that will apply at the next timestep t +1 . These tax rates are observable by the firms and consumer-workers at timestep t ; hence, they can condition their behavioral policy in response to the government policy. Similarly, at timestep t , firms set prices and wages that will be part of the global state at the next t +1 . This setup is akin to, though not exactly corresponding to, a Stackelberg leader-follower structure, where, for example, the government (leader) moves first, and the firms and consumer (followers) and move second (von Stackelberg et al., 2010). This typically gives a strategic advantage to the followers: they have more information to condition their policy on.

Consumer-Workers. Individual people both consume and work; we will refer to them as consumer-workers . At each timestep t , person j works l j,t hours and consumes c i,j,t units of good i . Each person chooses to work for a specific firm i at each timestep. Consumer-workers also (attempt to) consume ˆ c i,j,t . Each firm's good has a price p i (set by the firms, described below), and the government also sets an income tax rate. However, consumers cannot borrow or outspend their budget: if the cost of attempted consumption exceeds the budget, then we scale consumption so that ∑ i p i ˆ c t,i,j = B j .

Moreover, the realized consumption depends on the available inventory of goods ( y i,t , described below). The total demand for good i is ˆ c i,t = ∑ j ˆ c i,j,t . If there is not enough supply, we ration goods proportionally:

<!-- formula-not-decoded -->

Consuming and working change a consumer's budget B j,t . Consumer j has labor income z j,t = ∑ i l i,j,t w i,t ; each firm pays a wage w i,t . The cost of consumption is ∑ i p i,t · c i,j,t Moreover, with tax rate τ t , workers pay income tax τ t · z j,t ; the total tax revenue R t (which also includes taxes on the firms, described below) is redistributed evenly back to workers. In all, consumer budgets change as:

<!-- formula-not-decoded -->

Each consumer optimizes its behavioral policy to maximize utility:

<!-- formula-not-decoded -->

where γ c is the consumer's discount factor. The utility function is a sum of isoelastic utility over consumption and a linear disutility of work with coefficient θ j that can vary between workers.

Firms. At each timestep t , a firm receives labor from workers, produces goods, sells goods, and may invest in capital. At each timestep t , it sets a price p t +1 ,i for its good and chooses a wage w t +1 ,i to pay, both effective at the next timestep t +1 . If a firm invests ∆ k i,t in capital, its capital increases as k i,t +1 = k i,t +∆ k i,t . Using its capital k i,t and total labor L i,t = ∑ j l i,j,t (hours worked), a firm i produces Y units of good i , modeled using the production function :

<!-- formula-not-decoded -->

where 0 ≤ α ≤ 1 sets the importance of capital relative to labor. Also, consumers buy C i,t = ∑ j c i,j,t units of good i . Accordingly, inventories change as y t +1 ,i = y i,t + Y i,t -C i,t . Inventories are always positive, as only actually produced goods can be consumed. The firms receive a profit (or loss) P , pay taxes on their profit, and experience a change in their budget B :

<!-- formula-not-decoded -->

where σ is the corporate tax rate. The government receives σ t P i,t . Firms may borrow and temporarily be in debt (negative budget), but should have non-negative budget at the end of an episode ( no-Ponzi condition ). This may allow firms to invest more, which may lead to higher future economic growth. Each firm optimizes its behavioral policy to maximize profits as in Equation 5:

<!-- formula-not-decoded -->

where γ f is the firm's discount factor. The firm gets a negative penalty if it violates no-Ponzi .

Government. The government, or social planner, indexed by p , sets corporate and income tax rates, and receives total tax revenue R t = σ t ∑ j z j,t + τ t ∑ i P i,t . As a modeling choice, this revenue is redistributed evenly across consumer-workers; as such, the government's budget is always 0. The government optimizes its policy π p to maximize social welfare:

<!-- formula-not-decoded -->

where swf ( s t ) is the social welfare at timestep t , and γ p is the government's discount factor. In this work, we consider two definitions of social welfare (although many other definitions are possible): 1) total consumer utility (after redistribution), or 2) total consumer utility and total firm profit .

Open and Closed Economies via Export Markets. We consider both open and closed economies. In the open economy, firms can also sell goods to an external market which acts as a price-taker : their demand does not depend on the price of a good. Operationally, export happens after workerconsumers consume. The export market has a minimum price p export and a cap q export. If the price of good i is greater than the minimum price ( p i,t &gt; p export) then the additional export consumption is c t, export = min( q export , y i,t -C i,t ) , at price p i,t , i.e., the exported quantity is insensitive to the price.

From a learning perspective, the export market prevents firms from seeing extremely low total demand for their good, e.g., when prices are exorbitantly high and consumers do not want or cannot consume the good. In such cases, an on-policy learner that represents a firm may get stuck in a suboptimal solution with extremely high prices and no production as consumers cease to consume in response.

## 4 KEY SIMULATION IMPLEMENTATION DETAILS

In general, experimental outcomes can depend significantly on the implementation details; we outline several key implementation details hereafter. In addition, all simulation and training settings can be found in Table 1 and Table 2.

Budget Constraints. We implement budget constraints on the consumers by proportionally scaling down the resulting consumption of all goods to fit within a consumer's budget. Thus, the consumer actions really represent attempted consumption - if the budget is small or stock is limited, the actual consumption enjoyed by the consumer may be lower. Firm budgets are allowed to go negative (borrowing money). However, because the firm's goal is to maximize profit, they are incentivized to take actions will will be profitable, increasing their budget.

Scaling of Observables. The scales of rewards and state variables can vary widely in our simulation, even within time steps in a single episode. If the scales of loss functions or input features are very large or small, learning becomes difficult. We directly scale rewards and some state features by constant factors. For certain state features which have very large ranges (item stocks and budgets) we encode each digit of the input as a separate dimension of the state vector.

GPU Implementation We followed the WarpDrive framework (Lan et al., 2021) to simulate the DGE model and run MARL on a single GPU. We implemented the DGE dynamics as a CUDA kernel to leverage the parallelization capabilities of the GPU and increase the speed at which we can collect samples. We assigned one thread per agent (consumer, firm, or government); the threads communicate and share data using block-level shared memory. Multiple environment replicas run in

Table 1: Simulation Parameters.

| Parameter                                | Symbol   | Values                        |
|------------------------------------------|----------|-------------------------------|
| Labor disutility                         | θ        | 0.01                          |
| Pareto quantile function scale parameter | -        | 4.0                           |
| Initial firm endowment                   | B        | 2200000                       |
| Export market minimum price              | -        | 500                           |
| Export market maximum quantity           | -        | 100                           |
| Production function values               | α        | 0 . 2 , 0 . 4 , 0 . 6 , 0 . 8 |
| Initial capital                          | K        | 5000 or 10000                 |
| First round wages                        | w        | 0                             |
| First round prices                       | p        | 1000                          |
| Initial inventory                        | y        | 0                             |

Table 2: Training Hyperparameters.

| Parameter                             | Values     |
|---------------------------------------|------------|
| Learning Rate                         | 0.001      |
| Learning Rate (Government)            | 0.0005     |
| Optimizer                             | Adam       |
| Initial entropy                       | 0.5        |
| Minimum entropy annealing coefficient | 0.1        |
| Entropy annealing decay rate          | 10000      |
| Batch Size                            | 128        |
| Max gradient norm                     | 2.0        |
| PPO clipping parameter                | 0.1 or 0.2 |
| PPO updates                           | 2 or 4     |
| Consumer reward scaling factor        | 5          |
| Firm reward scaling factor            | 30000      |
| Government reward scaling factor      | 1000       |

multiple blocks, allowing us to reduce variance by training on large mini-batches of rollout data. We use PyCUDA (Kl¨ ockner et al., 2012) to manage CUDA memory and compile kernels. The policy network weights and rollout data (states, actions, and rewards) are stored in PyTorch tensors; the CUDA kernel reads and modifies these tensors using pointers to the GPU memory, thereby working with a single source of data and avoiding slow data copying.

Implementation Details. Furthermore, we outline several key implementation details.

- For consumers, consumption choices range from 0 to 10 units for each good and work choices from 0 to 1040 hours in increments of 260.
- Consumers have a CRRA utility function with parameter 0.1, and a disutility of work of 0.01.
- For firms, price choices range from 0 to 2500 in units of 500; wage choices from 0 to 44 in units of 11.
- The 10 firms are split into two groups, receiving either 5000 or 10000 units of capital. Within these groups, firms receive a production exponent ranging from 0.2 to 0.8 in increments of 0.2. Thus each firm has a different production 'technology'.
- Firms invest 10% of their available budget (if positive) in each round to increase their capital stock.
- Government taxation choices range from 0 to 100% in units of 20%, for both income tax and corporate tax rates.
- The government can either value only consumers when calculating its welfare ('consumeronly') or value welfare of both consumers and firms ('total'), with firm welfare downweighted by a factor of 0.0025 (to be commensurate with consumers).
- We set the minimum price at which firms are willing to export to be either 500 or 1000, and the quota for each firm's good to a variety of values: 10, 50, 100, or 1000.
- For consumers, consumption choices range from 0 to 10 units for each good and work choices from 0 to 1040 hours in increments of 260.

Agent Observations. The environment state s consists formally of all agent states and a general world state. Each agent observes can observe their own information and the global state:

<!-- formula-not-decoded -->

Here y i,t is the available supply of good i , p i,t is the price, w i,t is the wage. The extra information o i,t includes whether good i was overdemanded at the previous timestep and tax information.

In addition, consumer-workers observe private information about their own state: ( B i,t , θ ) A firm i also observes its private information: ( B i,t , k i,t , (0 , . . . , 1 , . . . , 0) , α ) , including a one-hot vector encoding its identity and its production function shape parameter α . The government only sees the global state.

## 5 REINFORCEMENT LEARNING ALGORITHM FOR A SINGLE AGENT

Firms and governments use 3-layer fully-connected neural network policies π ( a | s ) , each layer using 128-dim features, that map states to action distributions. Consumer policies are similar, using separate heads for each action type, i.e., the joint action distribution is factorized and depends on a shared neural network feature ϕ t ( s t ) : π ( a 1 , a 2 , . . . | s ) = π ( a 1 | ϕ, s ) π ( a 2 | ϕ, s ) . . . (omitting t and s t for clarity). Any correlation between actions is modeled implicitly through ϕ t .

There is a single policy network for each agent type, shared across the many agents of that type. To distinguish between agents when selecting actions, agent-specific features (parameters like the disutility of work, production parameters, and for firms, simply a one-hot representation of the firm) are included as part of the policy input state. Thus, despite a shared policy for each agent type, we model some degree of heterogeneity among agents. We also learn a value function V ( ϕ t ) for variance reduction purposes. We compare policies trained using policy gradients (Williams, 1992) or PPO Schulman et al. (2017).

RL parameter updates We now describe the RL parameter updates for any given agent type.

Given a sampled trajectory of states, actions, and rewards s t , a t , r t for t from 0 to the end time step T , we have the empirical return G t = ∑ T -t k =0 γ k r t + k +1 , the total discounted future rewards from time step t . The goal of the value function network is to accurately predict G t from a given state s t . We use a Huber loss function glyph[lscript] to fit the value network, glyph[lscript] ( V β ( s t ) -G t ) , and the value weights are updated as β t +1 = β t -η ∇ β ∑ T t =0 glyph[lscript] ( V β ( s t ) -G t ) , where η is the step size for gradient descent. Given the value function network's predictions, we can then define the advantages A t = G t -V ( s t ) and their centered and standardized versions ˆ A t = ( A t -E π [ A ]) / std ( A ) .

For the policy gradient approach, the optimization objective for the policy is:

<!-- formula-not-decoded -->

where H ( π ) = E π [ -log π ] is the entropy of π , α is the weight for entropy regularization (which may be annealed over time), and π θ is the policy with parameters θ . The true policy gradient for the first term is E π θ [ A t ∇ θ log π θ ] , which we estimate using sampled trajectories. For a single sampled trajectory, the full policy weight update is:

<!-- formula-not-decoded -->

where H ( π ( ·| s t )) is the entropy of the action distribution at a particular state. In practice, we sample multiple trajectories and compute a mini-batch mean estimate of the true policy gradient. In addition, we use proximal policy optimization (PPO), a more stable version of the policy gradient which uses a surrogate importance-weighted advantage function A PPO in the policy objective:

<!-- formula-not-decoded -->

which uses the current π θ and policy before the last update π old. Extreme values of the importance weights π θ /π old are clipped for stability. Moreover, in practice, using the standardized advantages ˆ A t

Consumer-workers are training

Consumer-workers entropy annealing

Labor disutility magnitude

Available firm actions (prices, wages)

Firms are training

Firm entropy annealing

Figure 2: Structured learning curricula. Colored bars show how annealing activates over time. Consumers are always training, use a decaying entropy regularization, and increasingly experience their labor disutility. Firms start with fixed prices and wages, and are slowly allowed to change these. Once the full price and wage range is available, firms start to train. Similarly, tax rates start at zero and are slowly allowed to change. Once the full tax range is available, the government starts to train.

<!-- image -->

<!-- image -->

and clipping gradients to bound their glyph[lscript] 2 norm also improve stability. Both simulation and RL ran on a GPU using the WarpDrive approach (Lan et al., 2021), see Section 4, while our PPO implementation followed Kostrikov (2018). We show pseudo-code for a single simulation and training step in Algorithm 1.

## 6 STABLE MULTI-AGENT REINFORCEMENT LEARNING THROUGH STRUCTURED CURRICULA

A key idea of our approach is the use of structured multi-agent curricula to stabilize learning, where each individual agent uses standard RL. These curricula consist of staged training, annealing of allowed actions, and annealing of penalty coefficients, see Figure 2. This method extends the approach used by Zheng et al. (2020).

Intuition for Structured Curricula. We define curricula based on these observations about our multi-agent learning problem: 1) during exploration, many actions may reduce utility, while few actions increase utility, 2) high prices or tax rates can eliminate all gains in utility, even though the consumer-worker did not change its policy, 3) for stable learning, agents should not adapt their policy too quickly when experiencing large negative (changes in) utility, and 4) in a Stackelberg game, the followers (e.g., consumers, firms) should get enough time to learn their best response to the leader's policy (e.g., the government). We now operationalize these intuitions below.

Figure 3: Training progress with structured curricula in an open RBC model. Curves show averages across 3 repetitions with different random seeds. In each plot, the blue (black) vertical line shows when the firms (government) start training, see Figure 2. Left two plots: Consumer and firm rewards during training. All runs converged to qualitatively similar outcomes. We've confirmed these solutions form an approximate equilibrium under an approximate best-response analysis, see Figure 4. Once firms start training, their reward (profits) significantly increases. When the government starts training, firms get even higher reward, as the social welfare definition includes the firms' profits. Right two plots: Average wages and prices across firms during training. Firms increase prices rapidly and lower wages once they start training. This comes at the expense of consumer reward (utility). As such, this setting represents an economy in which firms have significant economic power.

<!-- image -->

Staged Learning and Action Space Annealing. All policies are randomly initialized. We first allow consumers to train, without updating the policies of other agents. Initially, firm and government actions are completely fixed; prices and wages start at non-zero levels. We then anneal the range of firm actions without training the randomly initialized policy. This allows consumers to learn to best respond to the full scope of prices and wages, without firms strategically responding. Once firm action annealing is complete, we allow the firm to train jointly with the consumers. We then perform the same process, gradually allowing the government to increase its corporate and income tax rates, so that firms and consumers can react to a wide range of tax rates. Once the annealing process is complete, we allow the government to train to maximize welfare.

Penalty Coefficient Annealing. In addition to the action annealing, we anneal two penalty coefficients. First, we slowly increase the consumers' disutility of work over time, which avoids disincentivizing work early in the training process. Second, as each agent starts training, we assign a high value (0.5) for the entropy coefficient in their policy gradient loss, and gradually anneal it down over time to a minimum of 0.1. This ensures that when the firm or government policies start training, their 'opponent' policies are able to learn against them without being too quickly exploited.

Many Local Equilibria and Convergence. We expect that there are many possible solutions in our game that our approach may converge to. Establishing convergence is difficult in general-sum games - the best results make use of regret-minimizing algorithms for the agents and only establish convergence to coarse-correlated equilibria (Daskalakis et al., 2021). In our case, convergence is hard to guarantee and may be only local since we use deep neural networks and can't ensure global optimality of our policy parameters. We don't have theoretical bounds on the magnitude of glyph[epsilon1] for an glyph[epsilon1] -Nash equilibrium, although empirically, the degree of possible improvement seems small across RL approximate best-response analysis.

Our training curricula are designed to avoid trivial solutions where the economy shuts down, but it may also introduce bias into which non-trivial solutions are reached. However, we observe a spectrum of qualitatively different outcomes, so we see our approach as a way to explore more solutions than are possible with simplified, e.g., linearized, models.

## 7 ANALYZING LEARNED STRATEGIC BEHAVIORS IN RBC MODELS

We show that our approach is sound and finds spectra of meaningful solutions of RBC models. We analyze to what extent these solutions are RBC equilibria using an approximate best-response analysis

Consumer

141

12-

Firm

Equil.

BR Improv.

Middle-

12 -

10-

₴ 101

Mean Rewards

81

6

41

2-

2-

0-

Middle -

End

End

Government

Middle

20

— Consumer

150K

• Consumer

Firm

Firm

• Government

Government

Firm Rewards

145K -

140K 1

135K |

130K |

125K -

120K -

115K -

40

0

60

Fixed tax rate in percentage/RL tax rate

Best Response

## Approximate Best-Response Improvements

Figure 4: In each plot, the agent types (consumer, firm, and government) refer to cases when only that agent type is training. Left: best-response reward improvements during training. The stacked bar chart shows the original mean rewards (blue) and improvement after approximate best-response training (purple). For firms and governments, the mean rewards are measured in units of 10 4 and 10 3 , respectively. We compare the best-response improvement in the middle and at the end of training. The improvement from best-response is significant in the middle and much less at the end, indicating that training is closer to an equilibrium at the end. Right: outcome shifts under best-responses. We plot firm against consumer rewards on an absolute scale at training convergence for several runs. We then find best-responses by further training each agent type separately. Each arrow shows the shift in (consumer reward, firm reward) after best-response: blue for consumers best responding, red for firms, and green for government. In the figure, we display only those arrows when rewards change by more than 1%. At convergence, rewards for any agent type typically do not change significantly in this best-response analysis. This holds generally for the approximate equilibria reported in this work.

<!-- image -->

Figure 5: Mean rewards under fixed and RL taxes. For firms and governments, the mean rewards are measured in units of 10 4 and 10 3 , respectively. The mean rewards for the consumers increase with tax rates, whereas they decrease for the firms. RL taxes can improve the mean reward for both types. For instance, RL tax policies increase social welfare (green) by almost 15% over the best fixed tax policy.

<!-- image -->

and analyze the economic relationships that emerge from the learned solutions. We study variations of our RBC model with 100 consumers and 10 firms. For all simulation parameter settings, see Table 1. We repeated all experiments with 3 random seeds.

Structured Curricula during Training. Figure 3 shows a representative result of training using our structured curricula. Although all RL agents aim to improve their objective, some agents may see

End

Table 3: Reward improvement under best-response. Best-responses are measured at the end of training as a fraction of the reward improvement during training, over 10 random seeds. These results represent worst-cases: due to the stochastic nature of RL, outcomes can differ significantly across individual runs. In fact, we observed that besides a few anomalies, overall the improvements were in fact less than 0.2% for consumers, 5% for firms, and 0.1% for the government.

|                                        | Consumer   | Firm   | Government   |
|----------------------------------------|------------|--------|--------------|
| Reward improvement under best-response | < 3%       | < 10%  | < 1%         |

## Learned Solutions in Closed RBC Models

Figure 6: Outcomes at convergence for the same experiments in Figure 4, under the closed economy. Each point represents an approximate equilibrium, verified by an approximate bestresponse analysis. Points of the same color and shape correspond to the same run. In this closed economy, training often converges to solutions with low consumer reward and little production. In particular, social welfare (government reward) does not increase with higher tax rates, average labor does not change with wages, and consumption is unchanged with price. An exception is a solution with significantly higher social welfare, labor, and consumption. This suggests multiple equilibria do exist, but non-trivial equilibria are harder to learn in the closed economy.

<!-- image -->

rewards decrease as the system converges, e.g., the consumers in Figure 3. Empirically, we found that training collapses to trivial solutions much more often without curricula.

Best-responses and Local Equilibrium Analysis We abstract the RBC as a normal-form metagame between three players representing the agent types, i.e., the consumers, firms, and government. We test whether a set of learned policies is an glyph[epsilon1] -Nash equilibrium for the meta-game by evaluating whether or not they are approximate best-responses. Recall that agents of the same type share policy weights, but use agent-specific inputs to their policy, hence this still models heterogeneity between agents.

To find an approximate best-response, we train each agent type separately for a significant number of RL iterations, holding other agent types fixed, and see to what extent this improves their reward. This measurement provides a lower bound on the glyph[epsilon1] for each local equilibrium. In general, we find empirically that best-responses improve rewards much more at the start compared to the end of training, see Figure 4. The best-response method found at most small improvements in agent rewards, at the end of training , see Table 3. This suggests that the use of RL for a single agent type is a practical method that yields meaningful results.

For practical reasons, as we consider settings with a large number of agents (e.g., 100 consumers), we limit our analysis to evaluating meta-game best-responses. The meta-game is defined over agent types, and is different from the game over individual agents. In the meta-game, a meta-agent consist of the agents of a single type and a meta-action consists of the policy prescribing actions for all agents of that type. In particular, a meta-game best-response (where the best-response is over the choice of policy shared by all agents of a given type) may not be an individual agent's best-response. A meta-game best-response may feature competition-coordination dynamics between agents of the same type and may introduce both beneficial or adverse effects for some individual agents. For instance, there may be free-riders that benefit from the collective performance while not putting in effort themselves. On the other hand, aggregate demand between competing consumers for a limited

## Learned Solutions in Open RBC Models

Figure 7: Outcomes at convergence for the same experiments in Figure 4, under the open economy. Each point represents an approximate equilibrium, verified by an approximate bestresponse analysis. Points of the same color and shape correspond to the same run. We learn multiple distinct, non-trivial solutions in an open economy. Blue lines show linear regressions to the data. Consumer and firm rewards are positively correlated ( r 2 = 0 . 25 ), e.g., if consumers earn more, they can consume more, yielding higher profits. Higher prices decrease mean consumption ( r 2 = 0 . 73 ), lower wages decrease mean hours worked ( r 2 = 0 . 19 ), and there is no strong signal that higher taxes improve social welfare ( r 2 = 0 . 06 ).

<!-- image -->

supply of goods may mean consumers on average lose utility. We empirically observed the latter case: overall consumer utility sometimes decreased during the best-response analysis for the consumers.

Comparing with Baseline Government Tax Policies. To show that our approach is sound, we show that RL tax policies lead to improved social welfare, compared to several manually-defined fixed tax-rate policies. In lieu of other potential baseline tax policy methods, which are not directly compatible with our approach due to discretization, the lack of market clearing, and other features of our model, these fixed tax policies provide a useful baseline. Here, social welfare is defined as a weighted sum of firm and consumer rewards, as defined in Section 4. As observed in Figure 7, RL policies generate a social welfare ranging from 3000 to 4000, depending on the solution (as noted above, multiple solutions could be reached). Compared to that, the best social welfare achieved using fixed tax rates was 3160. Figure 4 shows the social welfare achieved under various fixed tax-rates, ranging from 20% to 80%. We note that social welfare improves by almost 15% for the best solution under RL tax policy over the best fixed tax-rate policy. This shows that the RL policy can adjust taxes across different rounds to improve average social welfare.

Analyzing Learned Solutions Figure 6 visualizes and describes the learned solutions in open and closed RBC economies (with and without an export market). In both cases, we find multiple qualitatively different solutions. In particular, learning mostly converges to mostly low-welfare solutions in the closed economy, due to very low wages resulting in little labor, production, and consumption. We stress that we describe these observations at the level of correlations. The RBC dynamics lead to circular interactions, e.g., a change in wages can cause future changes in wages due to changing consumer and hence changing firm behavior. As such, disentangling cause and effect is beyond the scope of this work.

Relationships between Aggregate Economic Variables. Figure 7 shows intuitive economic relationships between various aggregate economic quantities. For example, hours worked increase with wage, and consumption decreases with increasing price. Such relationships do not exist in the closed economy shown in Figure 6, because in most cases, the economy is stuck in a 'bad' equilibrium with very little work, production, or consumption. In such a setting, many agent actions, e.g., setting tax rates, increasing wages or prices, simply do not have any effect, e.g., increasing labor.

Analyzing Open RBC Solutions. The open RBC model admits a wide spectrum of qualitatively distinct solutions with distinct outcomes (e.g., consumer utility, hours worked, prices, and taxes), and trends that align with economic intuitions. To study the differences between outcomes, Figure 8 shows two rollouts sampled from different converged solutions, revealing qualitative similarities along with some distinct behaviors.

N-

Firm

Firm

E

Prices

234567

Hours Worked Per Firm

9|

Prices

Hours Worked Per Firm

Time step (quarters)

2400

2200

- 2000

- 1800

- 1600|

- 1400

- 1200

-1000

1000

- 800

- 600|

- 400

- 200

-0|

2400

2200

2000

- 1800

-1600

-1400

-1200

-1000

• 1000

800

600

- 400|

- 200

-0|

•30

25

20

15

-10

Figure 8: Two rollouts from an open RBC model. We show actions and states for two representative runs, in blue and green respectively. Despite differences in strategies, there are many qualitatively similar features. We observe that firms have different strategies: some set prices high and rely on exporting goods (for example, firm 3); others set prices lower and also sell to consumers (for example, firm 0). Consumers respond sensibly, only consuming when prices are low and mainly working when wages are not 0. (Note that hours worked may be very low for some firms but are not necessarily 0.) The first 5 firms have a lower level of starting capital. Within each group, the production exponent for labor ranges from 0.2 (higher tech, more reliance on capital) to 0.8 (lower tech, more reliance on labor). Firms 0 and 5 have the lowest parameter of 0.2; firms 4 and 9 have the highest parameter of 0.8. The differences in initial firm endowments as well as evolution during the episode lead to different final capital levels.

<!-- image -->

- In these examples, firms can profit by either focusing on consumers or the export market; they tend to set prices high at first and sell mainly to the export market. Some firms then lower prices halfway through the episode and begin selling to consumers.
- The consumers tend to work in intense 'bursts', and only consume when prices are lower at the end. Note that each firm receives a non-zero number of hours of labor.

Firm

Firm

Firm

0|

21

Wages

Actual Consumption of Goods

10

6

- Lower-technology firms, i.e., firms 3, 4, 8, and 9, whose production functions rely more on labor, consistently set high prices and lower wages towards the end, independent of their capital.
- Higher-technology firms lower prices over time and keep wages high. This is intuitive, as high technology firms can produce more efficiently and hence face lower labor costs per unit of production.
- The firms start with different endowments of capital and invest a fixed percentage of their profits over time, with higher tech firms being more profitable and thus able to sustain higher investment in capital.

## 8 DISCUSSION

Economic models often assume the existence of a small number of representative agents whose behavior is simple and analytically tractable. Economists have long understood that these assumptions are unrealistic, and that it could be desirable to define models with more complexity and heterogeneity. Yet once the models are no longer analytically tractable, finding solutions requires new computational tools.

In this work, we showed how to adapt multi-agent RL to enable meaningful economic analysis of general equilibrium models. By using these algorithmic and computational tools, it is possible to weaken modeling assumptions and increase the scale of economic simulations and analysis.

On the other hand, there are also limitations to this approach. Daskalakis et al. (2009) showed that computing equilibria for general-sum games is hard in terms of computational complexity, even for simple matrix games. There are no theoretical guarantees yet that our framework can find all equilibria in sequential economic games. However, our best-response analysis suggests our framework does discover meaningful local equilibria.

Thus, our hardware-accelerated MARL approach enables economic analysis with a large number of agents allows analyzing complex economic problems in a practical way, while theoretical guarantees are to be developed further. Hence, our current framework is at least an exploratory tool for finding qualitatively different solutions, e.g., by varying initial conditions, environment parameters, or conditioning sampling, and provides exciting avenues for future research.

## REFERENCES

- Daron Acemoglu, Mikhail Golosov, and Aleh Tsyvinski. 2010. Dynamic Mirrlees taxation under political economy constraints. The Review of Economic Studies 77, 3 (2010), 841-881.
- G.C. Archibald, E.S. Phelps, A.A. Alchian, and C.C. Holt. 1970. Microeconomic Foundations of Employment and Inflation Theory . Norton.
- Manoj Atolia, Santanu Chatterjee, and Stephen J Turnovsky. 2010. How misleading is linearization? Evaluating the dynamics of the neoclassical growth model. Journal of Economic Dynamics and Control 34, 9 (2010), 1550-1571.
- Yu Bai, Chi Jin, Huan Wang, and Caiming Xiong. 2021. Sample-Efficient Learning of Stackelberg Equilibria in General-Sum Games. arXiv preprint arXiv:2102.11494 (2021).
- David Balduzzi, Sebastien Racaniere, James Martens, Jakob Foerster, Karl Tuyls, and Thore Graepel. 2018. The Mechanics of N-Player Differentiable Games. arXiv preprint arXiv:1802.05642 (Feb. 2018).
- Eric Bonabeau. 2002. Agent-Based Modeling: Methods and Techniques for Simulating Human Systems. Proceedings of the National Academy of Sciences 99, suppl 3 (May 2002), 7280-7287. https://doi . org/10 . 1073/pnas . 082080899
- Lena Mareen Boneva, R. Anton Braun, and Yuichiro Waki. 2016. Some unpleasant properties of loglinearized solutions when the nominal rate is zero. Journal of Monetary Economics 84 (2016), 216-232. https://doi . org/10 . 1016/j . jmoneco . 2016 . 10 . 012

- Michael Bowling and Manuela Veloso. 2002. Multiagent learning using a variable learning rate. Artificial Intelligence 136, 2 (2002), 215-250.
- Noam Brown and Tuomas Sandholm. 2018. Superhuman AI for heads-up no-limit poker: Libratus beats top professionals. Science 359, 6374 (2018), 418-424.
- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems 33 (2020), 1877-1901.
- Mingli Chen, Andreas Joseph, Michael Kumhof, Xinlei Pan, Rui Shi, and Xuan Zhou. 2021. Deep Reinforcement Learning in a Monetary Model. arXiv preprint arXiv:2104.09368 (2021).
- Wilbur John Coleman. 1990. Solving the stochastic growth model by policy-function iteration. Journal of Business &amp; Economic Statistics 8, 1 (1990), 27-29.
- Panayiotis Danassis, Aris Filos-Ratsikas, and Boi Faltings. 2021. Achieving Diverse Objectives with AI-driven Prices in Deep Reinforcement Learning Multi-agent Markets. arXiv preprint arXiv:2106.06060 (2021).
- Constantinos Daskalakis, Maxwell Fishelson, and Noah Golowich. 2021. Near-optimal no-regret learning in general games. Advances in Neural Information Processing Systems 34 (2021).
- Constantinos Daskalakis, Paul W Goldberg, and Christos H Papadimitriou. 2009. The complexity of computing a Nash equilibrium. SIAM J. Comput. 39, 1 (2009), 195-259.
- Oliver De Groot, Ceyhun Bora Durdu, and Enrique G Mendoza. 2019. Approximately Right?: Global v. Local Methods for Open-Economy Models with Incomplete Markets . Technical Report. National Bureau of Economic Research.
- Jakob N. Foerster, Richard Y. Chen, Maruan Al-Shedivat, Shimon Whiteson, Pieter Abbeel, and Igor Mordatch. 2017. Learning With Opponent-Learning Awareness. arXiv:1709.04326 [Cs] (Sept. 2017). http://arxiv . org/abs/1709 . 04326 arXiv: 1709.04326.
- Mikhail Golosov, Maxim Troshkin, and Aleh Tsyvinski. 2011. Optimal Dynamic Taxes . Working Paper 17642. National Bureau of Economic Research. https://doi . org/10 . 3386/w17642
- Andrew G Haldane and Arthur E Turrell. 2019. Drawing on different disciplines: macroeconomic agent-based models. Journal of Evolutionary Economics 29, 1 (2019), 39-66.
- James E Hartley and James E Hartley. 2002. The representative agent in macroeconomics . Routledge.
- Burkhard Heer and Alfred Maussner. 2009. Dynamic general equilibrium modeling: computational methods and applications . Springer Science &amp; Business Media.
- Edward Hill, Marco Bardoscia, and Arthur Turrell. 2021. Solving Heterogeneous General Equilibrium Economic Models with Deep Reinforcement Learning. arXiv preprint arXiv:2103.16977 (2021).
- Tom D Holden. 2017. Existence and uniqueness of solutions to dynamic models with occasionally binding constraints. The Review of Economics and Statistics (2017), 1-45.
- Junling Hu and Michael P Wellman. 2003. Nash Q-learning for general-sum stochastic games. Journal of machine learning research 4, Nov (2003), 1039-1069.
- Kenneth L Judd. 1992. Projection methods for solving aggregate growth models. Journal of Economic theory 58, 2 (1992), 410-452.
- Kenneth L Judd. 1997. Computational economics and economic theory: Substitutes or complements? Journal of Economic Dynamics and Control 21, 6 (1997), 907-942.
- Daniel Kahneman. 2003. Maps of bounded rationality: Psychology for behavioral economics. American economic review 93, 5 (2003), 1449-1475.

- Nitin Kamra, Umang Gupta, Fei Fang, Yan Liu, and Milind Tambe. 2018. Policy learning for continuous space security games using neural networks. In Thirty-Second AAAI Conference on Artificial Intelligence .
- Greg Kaplan, Benjamin Moll, and Giovanni L Violante. 2018. Monetary policy according to HANK. American Economic Review 108, 3 (2018), 697-743.
- Alan P Kirman. 1992. Whom or what does the representative individual represent? Journal of economic perspectives 6, 2 (1992), 117-136.
- Andreas Kl¨ ockner, Nicolas Pinto, Yunsup Lee, Bryan Catanzaro, Paul Ivanov, and Ahmed Fasih. 2012. PyCUDA and PyOpenCL: A scripting-based approach to GPU run-time code generation. Parallel Comput. 38, 3 (2012), 157-174.
- Ilya Kostrikov. 2018. PyTorch Implementations of Reinforcement Learning Algorithms. https: //github . com/ikostrikov/pytorch-a2c-ppo-acktr-gail .
- Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. 2012. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems 25 (2012).
- Tian Lan, Sunil Srinivasa, Huan Wang, Caiming Xiong, Silvio Savarese, and Stephan Zheng. 2021. WarpDrive: Extremely Fast End-to-End Deep Multi-Agent Reinforcement Learning on a GPU. arXiv:2108.13976 [cs.LG]
- Joel Z. Leibo, Vinicius Zambaldi, Marc Lanctot, Janusz Marecki, and Thore Graepel. 2017. MultiAgent Reinforcement Learning in Sequential Social Dilemmas. arXiv:1702.03037 [Cs] (Feb. 2017). http://arxiv . org/abs/1702 . 03037 arXiv: 1702.03037.
- Michael L. Littman. 1994. Markov Games as a Framework for Multi-Agent Reinforcement Learning. In Machine Learning Proceedings 1994 , William W. Cohen and Haym Hirsh (Eds.). Morgan Kaufmann, San Francisco (CA), 157-163. https://doi . org/10 . 1016/B978-1-55860335-6 . 50027-1
- Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, and Igor Mordatch. 2017. Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments. arXiv:1706.02275 [Cs] (June 2017). http://arxiv . org/abs/1706 . 02275 arXiv: 1706.02275.
- Robert E Lucas and Thomas Sargent. 1981. After keynesian macroeconomics. Rational expectations and econometric practice 1 (1981), 295-319.
- Andreu Mas-Colell, Michael D. Whinston, and Jerry R. Green. 1995. Microeconomic Theory . Oxford University Press, Oxford, New York.
- Enrique G Mendoza. 1991. Real business cycles in a small open economy. The American Economic Review (1991), 797-818.
- Enrique G Mendoza and Sergio Villalvazo. 2020. FiPIt: A simple, fast global method for solving models with two endogenous states &amp; occasionally binding constraints. Review of Economic Dynamics 37 (2020), 81-102.
- Noam Nisan, Tim Roughgarden, Eva Tardos, and Vijay V. Vazirani. 2007. Algorithmic Game Theory . Cambridge University Press. https://doi . org/10 . 1017/CBO9780511800481
- OpenAI. 2018. OpenAI Five. https://blog . openai . com/openai-five/ .
- Jean Pierre Danthine and John B. Donaldson. 1993. Methodological and empirical issues in real business cycle theory. European Economic Review 37, 1 (1993), 1-35. https://doi . org/ 10 . 1016/0014-2921(93)90068-L
- Dylan Radovic, Lucas Kruitwagen, Christian Schroeder de Witt, Ben Caldecott, Shane Tomlinson, and Mark Workman. 2021. Revealing Robust Oil and Gas Company Macro-Strategies Using Deep Multi-Agent Reinforcement Learning. (Sept. 2021).
- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (2017).

- David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. 2017. Mastering the Game of Go Without Human Knowledge. Nature 550, 7676 (2017), 354.
- Ekaterina Sinitskaya and Leigh Tesfatsion. 2015. Macroeconomies as constructively rational games. Journal of Economic Dynamics and Control 61 (2015), 152-182.
- Frank Smets and Rafael Wouters. 2007. Shocks and frictions in US business cycles: A Bayesian DSGE approach. American economic review 97, 3 (2007), 586-606.
- Sriram Srinivasan, Marc Lanctot, Vinicius Zambaldi, Julien P´ erolat, Karl Tuyls, R´ emi Munos, and Michael Bowling. 2018. Actor-critic policy optimization in partially observable multiagent environments. arXiv preprint arXiv:1810.09026 (2018).
- Joseph E Stiglitz. 2018. Where modern macroeconomics went wrong. Oxford Review of Economic Policy 34, 1-2 (2018), 70-106.
- Nancy L. Stokey, Robert E. Lucas, and Edward C. Prescott. 1989. Recursive Methods in Economic Dynamics . Harvard University Press. http://www . jstor . org/stable/j . ctvjnrt76
- Lawrence H Summers et al. 1986. Some skeptical observations on real business cycle theory . Harvard Institute of Economic Research.
- Richard S Sutton and Andrew G Barto. 2018. Reinforcement Learning: An Introduction . MIT Press.
- Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, et al. 2018. Deepmind control suite. arXiv preprint arXiv:1801.00690 (2018).
- Kristal K Trejo, Julio B Clempner, and Alexander S Poznyak. 2016. Adapting strategies to dynamic environments in controllable stackelberg security games. In 2016 IEEE 55th Conference on Decision and Control (CDC) . IEEE, 5484-5489.
- Alexander Trott, Sunil Srinivasa, Douwe van der Wal, Sebastien Haneuse, and Stephan Zheng. 2021. Building a Foundation for Data-Driven, Interpretable, and Robust Policy Design using the AI Economist. arXiv preprint arXiv:2108.02904 (2021).
- Karl Tuyls, Julien Perolat, Marc Lanctot, Joel Z Leibo, and Thore Graepel. 2018. A Generalised Method for Empirical Game Theoretic Analysis. arXiv:1803.06376
- Oriol Vinyals, Igor Babuschkin, Junyoung Chung, Michael Mathieu, Max Jaderberg, Wojciech M. Czarnecki, Andrew Dudzik, Aja Huang, Petko Georgiev, Richard Powell, Timo Ewalds, Dan Horgan, Manuel Kroiss, Ivo Danihelka, John Agapiou, Junhyuk Oh, Valentin Dalibard, David Choi, Laurent Sifre, Yury Sulsky, Sasha Vezhnevets, James Molloy, Trevor Cai, David Budden, Tom Paine, Caglar Gulcehre, Ziyu Wang, Tobias Pfaff, Toby Pohlen, Yuhuai Wu, Dani Yogatama, Julia Cohen, Katrina McKinney, Oliver Smith, Tom Schaul, Timothy Lillicrap, Chris Apps, Koray Kavukcuoglu, Demis Hassabis, and David Silver. 2019a. AlphaStar: Mastering the Real-Time Strategy Game StarCraft II. https://deepmind . com/blog/alphastar-masteringreal-time-strategy-game-starcraft-ii/ .
- Oriol Vinyals, Igor Babuschkin, Wojciech M Czarnecki, Micha¨ el Mathieu, Andrew Dudzik, Junyoung Chung, David H Choi, Richard Powell, Timo Ewalds, Petko Georgiev, et al. 2019b. Grandmaster level in StarCraft II using multi-agent reinforcement learning. Nature 575, 7782 (2019), 350-354.
- H. von Stackelberg, D. Bazin, R. Hill, and L. Urch. 2010. Market Structure and Equilibrium . Springer Berlin Heidelberg.
- Yufei Wang, Zheyuan Ryan Shi, Lantao Yu, Yi Wu, Rohit Singh, Lucas Joppa, and Fei Fang. 2019. Deep reinforcement learning for green security games with real-time information. In Proceedings of the AAAI Conference on Artificial Intelligence , Vol. 33. 1401-1408.
- Michael P Wellman. 2006. Methods for empirical game-theoretic analysis.

- Ronald J. Williams. 1992. Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. Mach. Learn. 8, 3-4 (May 1992), 229-256. https://doi . org/ 10 . 1007/BF00992696
- Yaodong Yang, Rui Luo, Minne Li, Ming Zhou, Weinan Zhang, and Jun Wang. 2018. Mean field multi-agent reinforcement learning. In International Conference on Machine Learning . PMLR, 5571-5580.
- Eric Zhan, Stephan Zheng, Yisong Yue, Long Sha, and Patrick Lucey. 2018. Generative multi-agent behavioral cloning. arXiv (2018).
- Kaiqing Zhang, Zhuoran Yang, and Tamer Bas ¸ar. 2019. Policy optimization provably converges to Nash equilibria in zero-sum linear quadratic games. arXiv preprint arXiv:1906.00729 (2019).
- Stephan Zheng, Alexander Trott, Sunil Srinivasa, Nikhil Naik, Melvin Gruesbeck, David C Parkes, and Richard Socher. 2020. The ai economist: Improving equality and productivity with ai-driven tax policies. arXiv preprint arXiv:2004.13332 (2020).
- Stephan Zheng, Yisong Yue, and Jennifer Hobbs. 2016. Generating long-term trajectories using deep hierarchical networks. Advances in Neural Information Processing Systems 29 (2016).

## A ETHICS STATEMENT

Our work proposes a framework to model economies using Multi-Agent Reinforcement Learning and thus may be used to draw implications about the real world. Our findings and used simulations are purely for research purposes and should not be used to make decisions in real-world systems. Furthermore, our framework should not be used to explore methods to increase discrimination or unfairness in real-world systems.

Assumptions, limitations, and ethical implications of using ML for economics. All choices in the economic simulation model, RL algorithms, reward functions, etc, play an important but difficult-to-understand role in equilibria selection and policy design. As in all ML applications, there are assumptions and limitations in the methodology. This has ethical implications for their use in future policy design applications.

Mitigation strategies and interdisciplinary research. Economic simulation enables studying a wide range of economic incentives and their consequences, including models of stakeholder capitalism. However, the version of the simulation as used in this work is not an actual tool that should be used for policy making.

Many design choices influence the eventual policy recommendations. For example, the designer is free to set the social welfare objective that the government optimizes for. As such, it is crucial that these choices are debated and made in a socially acceptable fashion by all stakeholders, and made transparent and accessible to all.

More generally, to mitigate ethical risk, further mitigation strategies may include performing a what-if analysis over worst-outcomes, opening research results to domain experts (social scientists, ethics experts, etc), and open-sourcing the research results, amongst others. In all, the design and use of ML for policy recommendations will require robust, multilateral discussion and careful consideration of ethical risk, potential harm, and which trade-offs are being made.

We now detail some assumptions, limitations, and potential ethical risk among different dimensions of using ML for economics. We stress that there can be more (unknown) aspects that we do not address here. As such, we see this discussion as a starting point of discussion for the ML and economics community.

Economic simulation and data. While the current version of the economic simulation provides only a limited representation of the real world, we recognize that future, large-scale iterations can still contain biases and unrealistic assumptions. Furthermore, non-representative simulation environments may result in biased policy recommendations. For instance, the under-representation of communities and segments of the work-force in training data might lead to bias in simulations that build on those and lead to biased AI policies. As such, collecting more representative data is a key challenge for future research in using ML for economic policy recommendations.

Our RBC model is a stylized model of real economies. RBC models are a commonly used class of economic models (see e.g. Smets and Wouters (2007)). However, as any model, it contains assumptions and stylizations. Future simulations may miss (un)known features that pertain, e.g., to equity and equality in the economy. Therefore, using simulations that are not representative or well-calibrated, can exacerbate or create new socio-economic issues.

We list a few salient features and assumptions below, although we cannot exhaustively enumerate all features that may be relevant in future research.

- Our RBC model features consumers that differ in skill and perform different amounts of work. However, we do not model more fine-grained distributional features, such as educational attainment, wealth, inheritance, geography, or others.
- Similarly, firms produce a single good only and can invest and pay wages. Any worker can work for any firm. We do not model hiring practices, the geographic location of firms, non-monetary incentives or benefits (e.g., health insurance). To accurately model inequity in the real world, including such features may be necessary.

- On the government and societal level, we model tax policies and simple redistribution of tax revenue. We do not model targeted redistribution, tax credits, application-specific subsidies (e.g., education support). We do not model trading, inflation, debt, and other macroeconomic features that may impact social groups disparately.

Our RBC is more general than commonly used models: we do not enforce market clearing, for instance. Market clearing is an unrealistic assumption that supply always meets demand. Economic theory uses such constraints to make analysis tractable. In contrast, our learning approach is flexible and does not require such simplifying assumptions. We also assume that all agents can observe the wages offered by all the firms. However, in the real world not all agents have equal access to information - and this is a feature that can be studied by future research. As such, we view the flexibility of our learning approach as a positive, in that our framework may allow for studying more representative models.

Choice of economic incentives and rewards. Agents optimize their behavior given economic incentives, as modeled by their reward function. As such, future economic AI policies should clearly describe for which reward function they were optimized. Furthermore, more research is needed to understand how the choice of reward function influences the resulting policies, and how social and ethical values can be transparently encoded in reward functions. It is also an open question which ethical/social norms and values can or cannot be quantified, and how to encode trade-offs between conflicting values.

For example, the planner optimizes its policy to maximize 'social welfare', a standard economics concept. However, the definition of social welfare heavily influences the resulting policy and social outcomes. For example, Zheng et al. (2020) used equality times productivity as their objective and showed the resulting AI income taxes can improve equality over classic tax models. Standard economic works often use the utilitarian objective (sum of all agent rewards). An alternative is the Rawlsian objective (social welfare is the reward of the lowest-income agent). We emphasize the choice of social welfare is flexible and a choice made by the designer(s) and users of the framework.

Another key example is the discount factor used to weight rewards over time. Whether to emphasize short-term vs long-term rewards is a social choice that has ethical implications. For example, firms may emphasize short-term profits over long-term health issues, which may disparately impact different social groups.

Choice of agent model. The behavior of agents is determined by the policy model, e.g., the neural networks used in our work. Neural networks are universal function approximators, given enough width (or depth) in their layers. However, in practice, neural networks may still encode structural biases and only parameterize a particular subspace of all theoretically possible policy models. For our networks, a particular concern might be architectural constraints: our policy networks are not recurrent (so only consider the current state) and sometimes don't allow correlated actions. With enough parameters these networks are still capable of representing a wide range of policies, but these architectural constraints represent implicit priors which conceivably might not reflect human decision-making. As such, more research is needed on what the limits are of neural networks in terms of emulating human behaviors, and to what extent more and diverse datasets can help alleviate such concerns.

Choice of algorithms and learning strategies. The RBC model is an economic 'game' that has multiple equilibria. It is not well understood theoretically to which equilibria a given RL algorithm converges. Indeed, previous work has studied MARL beyond independent learners, including Nash-Q (Hu and Wellman, 2003), WoLF (Bowling and Veloso, 2002), and MADDPG (Lowe et al., 2017). This extends to our use of structured curricula, reward shaping, and other forms of multi-agent learning algorithms or strategies. These methodological choices can all impact the equilibria one finds (or doesn't find) using ML.

This is important because different equilibria can have different levels of social welfare and granular social outcomes (e.g., equality, type of work performed, unemployment level). From an ethical point of view, it is therefore possible that certain choices of algorithms, etc, may bias policy recommendations and simulation outcomes to socially or ethically undesirable situations. For instance, certain social groups in the simulation may be disparately impacted by policy recommendations. Therefore,

it is important for future research to analyze how different RL algorithms may selectively converge to certain equilibria, and how one might enumerate all possible equilibria. This is still a significant theoretical and empirical challenge.

Defining and justifying the objectives for the social planner and other methodological choices is a complex discussion, and requires a more in-depth understanding of the functioning of ML that is beyond the scope of this work. This requires multilateral, interdisciplinary discussion on, for example, what the preferred social choice is with respect to the definition of social welfare and constraints.

Choice of hyperparameters. RL algorithms may converge to different solutions depending on the chosen hyperparameters, e.g., learning rate, entropy regularization, or discount factor. For instance, the level of entropy regularization regulates the exploration-exploitation trade-off in actor-critic methods, a form of on-policy RL as used in our work. It is known that actor-critic methods may get stuck in suboptimal local maxima. This issue may be exacerbated in the multi-agent setting, where there are multiple equilibria, and it is unknown how algorithms converge towards different equilibria. As such, it is possible that certain choices of hyperparameter can encode structural biases towards certain outcomes in the simulation. These potential limitations are an area for future research.

Robustness of Deep RL. A key question is how robust learned policies are to perturbations in the simulation (parameters). This has ethical implications: policies that do well in simulation, may not do well in the real world if, e.g., income distributions differ between sim and real. As such, simulations that are not representative (enough) may lead to policy recommendations that disadvantage underrepresented social groups in the real world. More generally, it is well-known that deep learning models and RL policies can be very brittle and may not generalize well to unseen environments. As such, more robustness analysis should be done on any policy recommendation that is based on deep RL and related methods.

Explainability and Simplicity of AI policies. Even though AI policies may be effective, they may use intricate, unexplainable patterns in their input data to achieve high performance. Moreover, their behavior may vary wildly between different inputs. As a hypothetical example, an RL agent may make significantly different tax rate recommendations for people with slightly different income or education levels. Such behaviors can disproportionally affect underprivileged social groups, and have unintended short/long-term economic consequences, especially if models are not well-calibrated. It is an open question on what level/amount of data, or specific policy constraints, could mitigate such potential risk and harm. We also note that if one wanted to restrict the class of policies to ones that are sufficiently explainable, the same model-free policy optimization scheme could still be applied. Indeed, this is a big potential advantage of the RL approach.