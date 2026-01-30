OPEN

<!-- image -->

## Comparative analysis of Q-learning, SARSA, and deep Q-network for microgrid energy management

Sreyas Ramesh, Sukanth B N, Sri Jaswanth Sathyavarapu, Vishwash Sharma, Nippun Kumaar A. A.  &amp; Manju Khanna

The growing integration of renewable energy sources within microgrids necessitates innovative approaches to optimize energy management. While microgrids offer advantages in energy distribution, reliability, efficiency, and sustainability, the variable nature of renewable energy generation and fluctuating demand pose significant challenges for optimizing energy flow. This research presents a novel application of Reinforcement Learning (RL) algorithms-specifically Q-Learning, SARSA, and Deep Q-Network (DQN)-for optimal energy management in microgrids. Utilizing the PyMGrid simulation framework, this study not only develops intelligent control strategies but also integrates advanced mathematical control techniques, such as Model Predictive Control (MPC) and Kalman filters, within the Markov Decision Process (MDP) framework. The innovative aspect of this research lies in its comparative analysis of these RL algorithms, demonstrating that DQN outperforms Q-Learning and SARSA by 12% and 30%, respectively, while achieving a remarkable 92% improvement over scenarios without an RL agent. This study addresses the unique challenges of energy management in microgrids and provides practical insights into the application of RL techniques, thereby contributing to the advancement of sustainable energy solutions.

Keywords Microgrid, Q-learning, SARSA, Deep Q-network, PyMGrid, Model predictive control

In  today's  world,  sustainable  development  is  a  major  priority,  and  the  combination  of  renewable  energy sources with microgrids has revolutionized energy distribution systems. Microgrids, with their localized and self-sustaining nature, offer a solution to challenges like grid reliability, energy efficiency, and sustainability 1 . Reinforcement Learning (RL) offers a powerful solution for optimizing microgrid energy management. RL are ideal for dynamically balancing energy storage and consumption. Microgrids can operate independently or with the main grid, utilizing distributed energy resources like solar panels, wind turbines, and battery storage 2 . This research uses PyMGrid, an open-source Python microgrid simulator, to apply RL techniques for optimal energy management.

The study employs Q-learning, SARSA, and Deep Q-Network (DQN) algorithms. The study explores model predictive control (MPC) and Kalman filters. By leveraging RL, the study aims to develop intelligent control strategies to allocate energy resources efficiently, reduce costs, and enhance grid resilience.

In the subsequent sections of this paper, 'Literature survey' provides a comprehensive Literature Review, exploring  existing  methodologies  and  key  findings  in  the  realm  of  energy  management  for  microgrids. 'Research gap and motivation' outlines the Research Gap and Motivation, highlighting the existing challenges and the rationale for the study. This is followed by 'Main contribution' , which details the Main Contributions of the paper, emphasizing the novel approaches and innovations introduced. 'Proposed methodology' covers the Material and Methods, including a description of the study area and the data collection process. In 'Results and discussion' , the Results and Discussion are presented, where the findings of the study are analyzed and compared. Finally, 'Conclusion' concludes the paper, summarizing the key results, discussing their implications, and suggesting directions for future research.

Department of Computer Science and Engineering, Amrita School of Computing, Amrita Vishwa Vidyapeetham, Bengaluru, India.  email: aa\_nippunkumaar@blr.amrita.edu

## Literature survey

Arwa and Folly 3  have proposed a comprehensive review of different Reinforcement Learning (RL) algorithms used for managing power in microgrids. Comparative study had been conducted between RL algorithms and other optimization methods and a discussion was done based on challenges posed by traditional optimization methods. The review also identifies the issues with multi-agent RL methods and how important it is find solutions for seamless power distribution among interconnected microgrids. Another study proposes an RL architecture inspired by time series prediction, utilizing recurrent neural networks instead of traditional frameworks 4 . Their model outperforms traditional RL algorithms in distributed Industrial Internet of Things energy control. Erick and Folly 5 provides a deep and thorough study of the employment of RL methods for solving grid-tied mircrogrid power control problems and also provides a discussion the advantages and disadvantages of RL algorithms in solving the problems related to microgrids. Barbero et al. 6 have explored the application of RL algorithm in the operations of microgrids. The deep RL techniques called Deep Q network has been explored for learning the optimal policy. The primary objective of the work proposed is to minimize the cost of operating the microgrid, which includes a punishment for using more power. The results obtained are satisfactory when compared to other RL algorithms. A research proposes a decentralized energy management system using MARL 7 , where agents employ Q-learning to optimize strategies and reduce costs.

Xu et al. 8 ,  explore  the  optimization of multi-microgrid systems in smart grids through a novel approach using  preference-  based  multi-objective  reinforcement  learning  (MORL)  algorithms.  The  study  addresses the challenge of managing interactions between microgrids that is, the Independent System Operator (ISO), and the main power grid by formulating a multi-objective optimization problem that aims to maximize sales revenue from main grid suppliers, extend the life of energy storage, and minimize energy consumption costs for consumers. Another study focuses on enhancing power distribution resilience using Deep RL (DRL) 9 . Research compares MARL and single-agent RL methods for battery optimization in residential microgrids 10 . Alabdullah and Abido 11 poses the idea of the FDRL approach for smart finite micro-grid energy control using distributed energy resources. The proposed framework employs FL and DRL approaches to facilitate the power control of electricity load at the micro-grid edge side to relieve communication burden to the central controller. Authors consider  several  DRL  algorithms,  including  the  described  discretized  SAC,  discretized  DDPG  and  Deep  Q Network (DQN) algorithms in a single house case. The findings demonstrate that SAC performs better than other techniques used in terms of the average cumulative reward, carbon footprints, and battery replenishment strategies. The authors also highlight the effect of the number of houses on the FDRL scheme effectiveness and show that the best results are achieved when the participants aggregate heterogeneous local models to improve the generalization of the learning policies.

Harrold et al. 12 present a methodology for coordinating Renewable Energy Source (RES) units in microgrids using MADRL algorithms. Another study integrates Model-Free RL (MFRL) with microgrid control systems 13 . Similarly,  another  research  employs  DRL  to  enhance  smart  grid  efficiency 14 .  Another  study  discusses  DRL applications for power system stability control 15 .

Cao et al. 16 reviews different RL algorithms and their applications in modern power and energy systems. They  explained  different  variants  of  RL  algorithms  like  model-based,  value-based,  and  policy-based.  RL algorithms  face  a  problem  called  the  curse  of  dimensionality  and  they  gave  a  solution  combining  RL  with Deep learning which is also called Deep reinforcement learning solve this issue because we can use DL for approximating the state-action function. DRL enhances feature extraction, avoids manual feature design, and improves generalization in large state spaces. The discussion extends to Policy Gradient Algorithms, Actor-critic methods emphasizes their roles in continuous and high-dimensional action spaces. Multi-Agent DRL to address interactions  among  multiple  agents  in  scenarios  like  multiplayer  games  and  multi-robot  control  problems. There are a lot of advantages to using RL in power and energy systems considering its ability to develop nearoptimal control strategies, microgrids, energy management, electricity markets, and demand response through continuous interactions with the environment. RL algorithms like Deep Q-learning and Actor-Critic methods are utilized for specific objectives such as cost reduction and peak load shifting. The integration of renewable energy sources and the increasing flexibility of demand pose challenges to power grid stability. RL is proposed as a model-free algorithm to address uncertainties and optimize demand-side management by leveraging feedback signals for control logic. A research explores Q-Learning and SARSA for sustainable energy 17 .  Nithin et al. 18 discuss optimizing smart grids using GSM modules for load management and self-healing technology. Authors of 19 develop  an  agent-based  electricity  market  simulation  (AMES)  to  balance  market  prices  and  optimize power control, while another study focuses on integrating microgrids with Information and Communication Technologies (ICT) for improved power grid reliability 20 .

In conclusion, although substantial progress has been made in applying Reinforcement Learning (RL) to microgrid  energy  management,  significant  research  gaps  persist.  The  integration  of  RL  with  control  theory techniques, such as Model Predictive Control (MPC) and Kalman filters, remains underexplored, limiting the potential for handling uncertainties in renewable energy systems. Additionally, the sensitivity of RL models to hyperparameters is frequently overlooked, underscoring the need for systematic approaches to optimize these parameters  for  improved  robustness  in  real-world  applications.  Multi-agent  RL,  while  promising,  has  seen limited adoption, with most studies favoring single-agent frameworks that cannot fully address the complexities of real-world microgrid operations. There is also a lack of comprehensive comparative analyses of RL algorithms such as Q-Learning, SARSA, and Deep Q Networks (DQN) within a unified framework, which would provide valuable insights into the strengths and weaknesses of each method for energy management. Moreover, many studies rely predominantly on simulated data, with limited real-world validation, raising concerns about the practical applicability of these models in dynamic environments. Addressing these gaps through integration with control theory, hyperparameter optimization, robust multi-agent frameworks, comprehensive algorithm

comparisons, and validation on real-world data is essential for advancing RL-based microgrid management toward resilient, efficient, and scalable solutions.

## Research gap and motivation

Energy  optimization  in  microgrids  remains  a  significant  challenge  due  to  the  inherent  unpredictability  of renewable energy sources and fluctuating energy demands. While previous research has explored the application of Reinforcement Learning (RL) in energy management, there is a lack of comparative studies examining the performance of specific RL algorithms, such as Q-Learning, SARSA, and Deep Q-Network (DQN), in microgrid environments. This study addresses this gap by evaluating and comparing these algorithms to identify the most effective  approach for balancing energy flow, cost, and reliability in microgrid operations. The research also introduces a novel integration of RL with advanced control techniques, specifically Model Predictive Control (MPC)  and  Kalman  filters,  within  a  Markov  Decision  Process  (MDP)  framework.  This  hybrid  approach, which remains underexplored in existing literature, enhances the ability to handle the uncertainties associated with  renewable  energy  generation.  Additionally,  the  study  applies  real-world  datasets,  including  those  from NY Cambium, New York solar resources, and California ISO CO 2 emissions, providing a more realistic and applicable foundation for the proposed methods.

## Main contribution

The major limitations of this work are drawn from the extensive analysis and evaluation of three state-of-the-art RL algorithms, mainly Q-Learning, SARSA and DQN in achieving optimal energy management in microgrids. This comparative analysis not only reveals performance differences for these algorithms but also helps to discover which strategy is optimal to manage energy flow, cost, and risk in the microgrid as well as in conditions that constituent the microgrid systems. Perhaps the most important novelties of the research is the incorporation of MPC and Kalman filters into the MDP framework of RL. Unlike most of the current RL based methods, this hybrid approach resolves the variability of renewable energy generation and load which are often problematic for  RL.  The  work  also  has  the  advantage  of  utilizing  real-world  data  sets  like  NY  Cambium,  PVWatts,  and California ISO CO 2 emissions of the proposed methods under nearly real conditions. Additionally, it establishes a significant enhancement in performance where DQN exhibit remarkable cost-effective advancements over the conventional techniques and non-RL applications of the microgrid suggesting the longevity of RL in microgrid systems. Last of all, the paper identifies issues, which can shape literature further, including the question of hyperparameters' tuning in RL algorithms, and makes the proposal of the possible further research that will help to advance the application of RL-based solutions to microgrid energy management.

## Dataset description

Table 1 provides a detailed overview of the datasets used in this study to evaluate the RL agent's microgrid management capabilities. The New York Cambium Load Dataset, sourced from Cambium Data, includes hourly and seasonal load curve data in kilowatt-hours (kWh), modeling realistic electricity consumption patterns to test the RL agent's adaptability to fluctuating demand. The New York Solar Resource Dataset provides solar irradiance data specific to New York (Latitude: 40.73, Longitude: -74.02), capturing daily and seasonal variations in solar energy production, allowing the RL model to simulate the impact of renewable energy availability on microgrid performance. Lastly, the California ISO CO 2 Emissions Dataset records hourly and seasonal emissions data in tons, sourced from California ISO, and is used as a proxy for assessing the environmental impact of various energy management strategies. Together, these datasets provide comprehensive coverage of demand, renewable energy generation, and emissions, enabling a robust evaluation of the RL agent's effectiveness in maintaining an efficient and sustainable microgrid.

## Proposed methodology

The primary focus of the comparative study is to reduce the cost of electricity by integrating renewable energy in a microgrid environment using various RL algorithms. The proposed methodology diagram for microgrid energy management using RL algorithms can be referred to in Fig. 1 and each of these steps has been explained in detail in the further subsections.

## Environment

The environment used in this study to train the agents is called PyMgrid. PyMGrid is a python library that is designed for simulating and optimizing the operations of microgrids. PyMGrid consists of various features which helps in simulating various real world scenarios with microgrid and also integrating various renewable energy. PyMGrid provides Microgrid simulation which helps in simulating various components of a microgrid which includes energy sources, energy storage and energy loads, it also provides tools for forecasting energy

Table 1 .  Description of datasets used in the study.

| Dataset               | Source         | Parameters       | Time coverage    | Units   | Purpose                        |
|-----------------------|----------------|------------------|------------------|---------|--------------------------------|
| New York cambium load | Cambium data   | Load curve       | Hourly, seasonal | kWh     | Models load demand variations  |
| NYsolar resource      | Solar data     | Solar irradiance | Daily, seasonal  | kWh     | Simulates renewable generation |
| California ISO CO 2   | California ISO | Emissions        | Hourly, seasonal | Tons    | Estimates environmental impact |

Dataset - 1

Photo-

Voltaic

Energy

Dataset - 2

Electrical

Load

Dataset - 3

CO2

Emission

RL

Formulation

Action

Fig. 1 .  Overview of the proposed system.

<!-- image -->

generation  and  consumption.  PyMGrid  also  allows  the  implementation  of  various  Reinforcement  Learning algorithm to help optimize microgrid control in real time.

## RL formulation

State

State is the current situation an agent is in the given environment. The states contain important information regarding the environments observable characteristics. The state space of the microgrid is defined in Eq. (1).

<!-- formula-not-decoded -->

where S state space η is net load β is BSOC.

Net load

<!-- formula-not-decoded -->

Net load is defined in Eq. (2). If the load exceed the PV generation the net load is positive indicating an energy deficit on the other hand if PV generation exceeds the load, it indicates that the energy is a surplus. The net load can fluctuate over a time period due variation of energy demand which includes the time of the day and many other factors.

<!-- formula-not-decoded -->

The battery state of charge (BSOC) reflects the current energy level stores in the battery system deployed in the microgrid. BSOC is calculated as seen in the Eq. (3). BSOC helps in determining the amount of energy left in the microgrid for meeting the demand or providing subsidiary services.

Actions

Actions are the decisions that an agent makes to control the operations of the microgrid. The action space of the microgrid is defined in Eq. (4).

<!-- formula-not-decoded -->

where A is Action Space. ρ is Battery Charge. λ is Battery Discharge. ω is Grid Import. µ is Grid Export.

Battery  charging Charging  the  battery  system  using  the  energy  from  the  solar  cells  helps  in  storing  energy reserves for later used when demand exceeds generation. Charging power determines the amount of energy transferred to the battery per unit time.

Battery Discharging Discharging the stored energy from the battery to supply power to microgrid's loads. The main purpose of this action is to use the stored energy when renewable energy generation is insufficient during periods of high demand.

Grid  Import Importing  energy  from  the  main  electrical  grid  to  help  the  microgrid's  energy  supply.  By importing the energy from the main grid it addresses the energy deficits or peak demand periods where local generation and battery storage are insufficient.

Grid Export Exporting excess energy generated by the microgrid to the main grid, this is considered as good action as it indicates surplus energy and grid stability, which helps in sharing renewable energy with broader electricity networks.

## Rewards

The main aim of the work is to minimize the overall cost produced by the microgrid. Therefore, the reward is inversely proportional to cost. The cost is calculated using the formula in Eq. (5).

<!-- formula-not-decoded -->

where σ is Net Cost.

So by minimizing the cost agent maximizes the reward. The total cost consists of the cost of electricity and the penalty for CO 2 produced by the grid.

## Algorithms

Q-Learning

Q-Learning is used for solving problems that involve making decisions and are sequential in nature. Selecting the best action-selection policy plays an important role in helping Q learning learn and take ideal decisions which is done by continuously updating the probability of states based on action values and observed rewards and follows off policy characteristics.

## State-action-reward-state-action

State-Action-reward-state-action is another RL algorithm which is very similar to Q-Learning. SARSA learns the optimal policy by updating the Q values based on observed state Action-reward-state-action transitions and follows on policy characteristics.

## Deep Q-network

Deep Q Network (DQN) integrates deep learning techniques with a traditional reinforcement algorithm known as Q Learning 21,22 . In our study DQN employs dense layers to approximate Q-value function. The input to the network is the state of the agent and the output provided is the Q-value for every action possible following a state in value out architecture. DQN uses an unique feature called the experience replay, where all the fundamental values such as state-reward action-next action is stored in a replay memory buffer. DQN uses a separate target network with fixed parameters to compute Q-values during training.

## Training phase

Initialization Firstly the Q table is initialized which is a dictionary that contains estimated values of state action pairs. The parameters includes the information of the environment such as battery parameters and net load, the actions possible for the agent to take them to the next state. Q values are initialized to 0.

Training Q-Learning and SARSA and DQN have been trained iusing the same parameters which includes, the horizon which indicate the time steps and the number of episodes to run. The agent iterates over 100 training episodes. The agent's actions are based on the epsilon-greedy policy in which the epsilon value decreases by 0.02 and after every episode, the hyper parameters used can be observed in Table 2. The Q table values are updated based on observed reward and the difference between predicted and actual future rewards after updating the q-values.

The training phase of the DQN is slightly different, where the agent calculates the target Q value based on the experience tuple and the Bellman equation, the target Q value represents the expected cumulative future reward. The neural network is trained using the updated Q values as targets as the corresponding input states. The model learns to map states to respective target Q values. The loss function used is the MSE loss between predicted and target Q-values. The neural network consists of two hidden layers with 64 neurons. Figure 2 provides the visualization for the same.

Table 2 .  Hyper parameter used for algorithms.

| Parameter       |   Q-Learning |   SARSA |    DQN |
|-----------------|--------------|---------|--------|
| Horizon         |       100    |  100    | 100    |
| Epsilon         |         0.02 |    0.02 |   0.02 |
| Discount factor |         0.99 |    0.99 |   0.99 |

States

Battery

Charge

State

Netload

Conv (64)

Conv (64)

Fig. 2 .  Deep-Q network model.

<!-- image -->

## Model predictive control

MPC is  an  advanced  control  theory  method  that  is  used  to  optimize  the  performance  of  a  process  over  a future time step. MPC uses dynamic model of the system to predict future behaviour of the system and solves optimization problem to determine optimal control actions. MPC relies on the mathematical formulation of the system to predict the future states of the model such as the dynamic state of the battery and the net load. MPC operates at a finite time horizon Eq. (6). For each time step t it helps in predicting the systems behaviour for next H steps. At each time step MPC formulates optimization problems where the main objective is to minimize the cost function. The decision variables include battery charge and discharge rates. The optimization problem includes various constraints like battery capacity and maintaining balance between supply and demand. Only the first control action is implemented after solving the optimization problem. The time step then moves one step forward and then the process is repeated again for the next time step.

<!-- formula-not-decoded -->

where T is the Prediction Horizon. u is the Vector of Control Actions. g is the Time Period in Horizon.

```
Subject to. Soc (0) = SoC 0 Soc ( g + 1 ) = SoC ( g ) + u ( g ) -Load ( g ) + PVGeneration ( g ) SoC min /lessorequalslant SoC ( g ) /lessorequalslant SoC max , ∀ g = 0 , 1 , ? , T
```

For each time step g it helps in predicting the systems behaviour for next H steps.

## Kalman filter

Kalman filter is a recursive algorithm that provides estimates of some unknown variables given the measurements observed over time. Kalman filter has two phases: the prediction and updation phase.

Predicted state estimate:

Predicted estimate covariance:

State update:

Covariance update

Kalman gain:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where /slurabove y g | g -1 is the Predicted state estimate at time g given the state at time g -1. /slurabove y g | g is the Updated state estimate at time g. P g | g -1 is the Predicted error covariance matrix at time g. P g | g is the Updated error covariance matrix at time g. A is the State transition matrix. B is the Control input matrix (often set to zero). u g -1 is the Control input at time g -1. Q is the Process noise covariance matrix. K g is the Kalman gain at time g. H is the Observation matrix. R is  the Measurement noise covariance matrix. z g is  the Measurement at time g. I is  the Identity matrix.

Prediction  phase  the  state  transition  model.  The  prediction  phase  doesn't  take  the  new  measurements  into account.  The  current  state  is  estimated  using  the  state  transition  model  and  control  input  model.  The  state covariance matrix is updated which represents the uncertainty in the state estimation after prediction. Further noise is added to create uncertainty in the process.

Updation phase Kalman filter updates the predicted state using the new measurement. The phase adjusts the state estimate to match the observed data which takes the uncertainty in both prediction and measurement into account. The residual covariance is calculated which represents the uncertainty in the measurement prediction. The Kalman Gain is then calculated to determine the amount of prediction that should be adjusted based on the new measurement. The difference between actual measurement and predicted measurement is calculated also called the innovation. The state estimate is updates by adding the weight innovation, the Kalman gain ensures that the update is balanced. Lastly the state covariance matrix is updated which reflects the reduced uncertainty after incorporating the measurement.

This  research  has  a  quantitative  approach  where  the  RL  algorithms,  Q-learning,  SARSA,  and  DQN  are simulated under controlled CRM tasks. For consistency, the specific settings of each algorithm are stated here. Off-policy Q-learning with an error controlling reinforced learning rate of 0.1, a discount factor of 0.99 as well as an epsilon greedy policy where epsilon decreases from 1 to 0.1 in next 1000 episodes. SARSA is an on-policy algorithm that learns on the basis of action taken by an agent and has several parameters similar to Q-learning algorithm. We proposed to use the DQN model which learns with Q-learning on a approximate function defined by a neural network, suitable for high-dimensional state space. Initial values for DQN are set for learning rate ( α ) = 0.001,  discount  factor  ( γ ) = 0.99  and ε -greedy  strategy  with ε being  decayed.  To  further  improve  RL algorithm, MPC and Kalman filters are adopted.

MPC uses the deviation in energy demands from a predicted model to adjust for the upcoming control actions while Kalman filters predict the systems states from measurement which is normally tainted with noise and feeds into the RL algorithms for better decision making. It turns out that mappings and simulations cover 1000 episodes for the different conditions including high solar generation with low demand and fluctuating but high wind generation with variable loads, while the results are averaged over multiple runs to ensure statistical credibility. Primary performance factors are the cost, energy consumption, and robustness of the system, and ANOVA is used to perform detailed comparisons between the various algorithms. These features make the program highly organized and detailed in terms of parameters, thus increasing the level of transparency and replicability of the study by other interested researchers.

## Results and discussion

In this section the main focus lies on comparing the models: Q Learning, SARSA, DQN along with the integration of mathematical control theory models like MPC and Kalman filter and compared based on cost and learning curve. To conduct the comparison a microgrid has been setup which consists 1-Load, farm of solar panel, 2 batteries having a max capacity of 100 kW and1000 kW respectively and 1 grid.

Table 3 offers comprehensive results on the minimized cost of microgrid energy management on 24 h, 48 h, 1 week, and 1 month using Seven RL Models. As for the costs, the DQN algorithm outperforms others having the lowest costs without using other control strategies as MPC or the Kalman filters applying 11.1 dollars per

7

Total Cost in dollars

700

600 -

500

400

300

200 -

100

Table 3 .  Minimum cost for various models in dollars.

| Model                   |     24 h |   48 h |   1 week |   1 month |
|-------------------------|----------|--------|----------|-----------|
| Q-Learning              |  13.8    |  23.5  |  152.3   |   1384.6  |
| SARSA                   |  13.8    |  23.8  |  153     |   5919.8  |
| DQN                     |  11.1    |  22.8  |  126.8   |   1225.7  |
| MPC-Qlearning           |  24.1011 |  34.16 |  203.014 |   6124.37 |
| Kalman Filter-Qlearning |  13.786  |  23.49 |  422.074 |   6405.61 |
| MPC-SARSA               |  24.11   |  33.81 |  201.97  |   6115.08 |
| Kalman Filter-SARSA     |  18.96   |  23.49 |  108.337 |   4814.6  |
| No RL agent             | 188.74   | 481.35 | 2233.03  |   9328.93 |

Fig. 3 .  Learning curve of the algorithms.

<!-- image -->

24 h, 22.8 dollars per 48 h, 126.8 dollars per week and 1225.7 dollars per month. This shows optimization of energy cost implications by DQN.

We also notice improvements in other models when the Kalman filters are incorporated into the system. Such features integrated with Kalman filters with SARSA have for instance reduced cost to 108.34 dollars within a week unlike base SARSA cost at 153.0 dollars and Q-Learning at 152.3 dollars. This may have been made possible by the Kalman filter that can sense data noise and therefore provide better state estimations and consequent decision making in variants conditions. Likewise, the combination of Kalman filters with Q-Learning reduced costs in the scenarios of a further time horizon (48 h, 1 week, and 1 month) to confirm the impact of the filter on the steadiness of RL-based models when faced with fluctuating inputs characteristic of renewable power supply.

Although MPC alone did not give the best costs, when it is integrated with Q-Learning and SARSA it performs better relative to non-RL methods highlighting the potential of RL in microgrid energy management. Table 3 summarizes the findings obtained regarding cost optimization using DQN and the significance of incorporating Kalman filters with other RL methods such as Q-Learning and SARSA while suggesting that the right mixed of RL approaches and control methods must be deployed in order to manage the energy at the microgrid.

The learning curve analysis in Fig. 3 provides a detailed comparison of the performance of Deep Q-Network (DQN),  Q-Learning,  and  SARSA  for  microgrid  energy  management  over  100  episodes.  The  results  clearly demonstrate the superior cost optimization achieved by DQN, with its costs stabilizing within a range of 5-15 units. In contrast, Q-Learning and SARSA exhibit significantly higher costs, fluctuating between 95 and 110 units. This disparity highlights DQN's ability to learn and adapt more effectively, consistently reducing energy management costs over time. Moreover, the inclusion of error bands in the analysis further illustrates the stability and  reliability  of  DQN  compared  to  the  more  erratic  performance  of  the  other  algorithms.  These  findings emphasize DQN's potential as a robust and efficient solution for optimizing microgrid operations. This study has

- SARSA

Q-Learning

- DON

Table 4 .  Summary of monthly renewable energy generation, utilization, export, and cost savings.

| Month     |   Avg. renewable energy generated (kW) |   Avg. renewable energy utilized (kW) |   Avg. energy exported (kW) |   Avg. grid import cost (in $) |   Avg. Pv savings (in $) |   Net cost (in $) |
|-----------|----------------------------------------|---------------------------------------|-----------------------------|--------------------------------|--------------------------|-------------------|
| January   |                                 70.699 |                                70.699 |                       0     |                        1394.06 |                   70.699 |          1323.36  |
| February  |                                108.184 |                               108.184 |                       0     |                        1355.59 |                  108.184 |          1247.4   |
| March     |                                155.144 |                               152.338 |                       2.805 |                        1313.03 |                  152.338 |          1157.89  |
| April     |                                199.087 |                               182.99  |                      16.097 |                        1287.81 |                  182.99  |          1088.72  |
| May       |                                247.832 |                               227.324 |                      20.508 |                        1260.18 |                  227.324 |          1012.35  |
| June      |                                269.007 |                               252.328 |                      16.678 |                        1264.46 |                  252.328 |           995.457 |
| July      |                                252.483 |                               244.622 |                       7.861 |                        1324.4  |                  244.622 |          1071.92  |
| August    |                                217.038 |                               211.002 |                       6.035 |                        1318.06 |                  211.002 |          1101.02  |
| September |                                168.9   |                               167.223 |                       1.676 |                        1326.62 |                  167.223 |          1157.72  |
| October   |                                116.354 |                               116.109 |                       0.244 |                        1349.31 |                  116.109 |          1232.96  |
| November  |                                 66.265 |                                66.265 |                       0     |                        1398.33 |                   66.265 |          1332.06  |
| December  |                                 51.151 |                                50.498 |                       0.652 |                        1415.6  |                   50.498 |          1364.44  |

also tested the performance of the microgrid with the DQN algorithm, in a simulated environment consisting of range of PV and load values, which can be selected randomly. Table 4 tabulates the values of renewable energy generated, utilized, exported also the grid import cost, PV savings and net cost using the simulated data. On observing the table it can be noticed that during the winter months the amount of renewable energy produced and utilized is even and there isn't much of energy being exported and the costs are on the higher side, while moving to the summer months the amount of energy being generated are higher so is the exporting thereby reducing the cost during the summer months.

## Discussion

The algorithms in the study were used in a microgrid environment to address the unique challenges associated with energy management, which include dynamic energy demands, variable renewable energy supply, and the need for real-time optimization. Also these algorithms were used because, these are computationally efficient.

- Q-learning was used because it is a foundational reinforcement learning algorithm that not only provides a simple but also an effective approach for learning optimal policies in discrete action spaces. It helps evaluate how well basic RL methods can perform in managing energy distribution and decision-making in a microgrid environment where states and actions can be discretized.
- SARSA was used to explore the benefits of an on-policy approach, where the algorithm learns and updates the action- value function based on the actions taken by the current policy. This allows for a more conservative strategy, which can be beneficial in microgrids when there is a need to balance energy management without taking aggressive, risky actions that may destabilize the system.
- DQN (Deep Q-Network) was selected because microgrid environments often involve high-dimensional, continuous state spaces, such as varying power loads, energy prices, and renewable energy availability. DQN leverages the power of deep learning to handle these complexities, making it well-suited for environments where traditional tabular methods like Q-learning and SARSA would struggle.

The  reason  behind  DQN  performing  better  than  the  other  two  algorithms  is  that  DQN  uses  deep  neural networks to approximate the action-value function, which helps it to handle complex, high-dimensional state spaces unlike traditional tabular methods like Q-learning and SARSA struggle with. DQN employs experience replay, where past experiences are stored and randomly sampled during training. This reduces the correlation between consecutive samples thereby improving learning stability and efficiency.

DQN maintains a separate target network to stabilize the training process. This helps mitigate the problem of rapidly changing targets, which can destabilize learning in standard Q-learning and SARSA. The use of neural networks enables DQN to generalize better across states, leading to more effective policies for managing the dynamic and complex energy demands in microgrid systems. The combination of all these advantages can be seen in Fig. 3 where the DQN learning curve converges faster when compared to the other algorithms.

## Conclusion

This  research  explores  the  potential  of  Reinforcement  Learning  (RL)  for  optimizing  energy  management  in microgrids, presenting a comprehensive analysis of its application for achieving efficient and sustainable energy distribution. By utilizing the Microgrid (PyMGrid) framework and deploying various prominent RL algorithms, including Q-learning, SARSA, and Deep Q-Networks (DQN), the study bridges the gap between traditional control  methods  and  modern  intelligent  energy  management  strategies.  The  investigation  also  incorporates mathematical control theory models, such as Model Predictive Control (MPC) and Kalman filters, to establish a strong foundation for performance comparison. This integration provides valuable insights into the balance between exploration and exploitation, a critical aspect of RL, while allowing an in-depth analysis of the models' effectiveness in dynamic microgrid scenarios. Specifically, the study highlights that DQN, with its deep learning capabilities, outperforms Q-learning and SARSA significantly, achieving a 12% higher performance compared to

Q-learning, a 30% improvement over SARSA, and an impressive 92% performance boost compared to a baseline without any RL agent. The findings emphasize the effectiveness of using lightweight RL-based approaches in optimizing energy usage, reducing costs, and enhancing the overall efficiency of microgrid systems. We have also demonstrated how the RL algorithms infused with microgrid perform during the year in terms of cost. The research advances the understanding of intelligent microgrid energy management and paves the way for more sustainable and reliable energy solutions for the future.

Future work will build on these results by incorporating more sophisticated feature extraction techniques, such as wavelet transforms, to enhance the model's understanding of complex energy patterns. Additionally, the  inclusion  of  more  real-world  parameters  will  improve  the  training  process,  making  the  prototype  more accurate and effective. The study also envisions the integration of additional microgrid components, such as energy storage systems and variable energy sources, to test the adaptability and robustness of the management system.  Furthermore,  exploring  Multi-Agent  Reinforcement  Learning  (MARL)  frameworks  could  allow  for decentralized  energy  management  strategies,  making  the  microgrid  more  resilient  and  better  equipped  to handle real-world scenarios.

## Data availability

The datasets generated during and/or analysed during the current study are available from the corresponding author on reasonable request.

Received: 25 September 2024; Accepted: 16 December 2024

## References

1.  Muhammad &amp; Khalid Smart grids and renewable energy systems: perspectives and grid integration challenges. Energy Strateg. Rev. 51 , (2024).
2.  Khanna, M., Srinath, N. K. &amp; Mendiratta, J. K. Feature extraction of time series data for wind speed power generation. In IEEE 6th International Conference on Advanced Computing (IACC), Bhimavaram, India, 2016 (2016).
3.  Arwa,  E.  O.  &amp;  Folly,  K.  A.  Reinforcement  learning  techniques  for  optimal  power  control  in  grid-connected  microgrids:  a comprehensive review. In IEEE Access , vol. 8 (2020).
4.  Dridi, A., Afifi, H., Moungla, H. &amp; Badosa, J. A novel deep reinforcement approach for iiot microgrid energy management systems. In IEEE Transactions on Green Communications and Networking , vol. 6 (2022).
5.  Erick,  A.  O.  &amp;  Folly,  K.  A.  Reinforcement learning approaches to power management in grid-tied microgrids: a review, 2020 Clemson University Power Systems Conference (PSC), Clemson, SC, USA (2020).
6.  Domínguez-Barbero, D., García-González, J., Sanz-Bobi, M. A., Eugenio, F. &amp; Sánchez-Úbeda. Optimising a microgrid system by deep reinforcement learning techniques energies. 13 , 2020 (2020).
7.  Esmat  Samadi,  A.  &amp;  Badri  Reza  Ebrahimpour,decentralized  multi-agent  based  energy  management  of  microgrid  using reinforcement learning. Int. J. Electr. Power Energy Syst. 122 , (2020).
8.  Xu, J., Li, K. &amp; Abusara, M. Preference based multi-objective reinforcement learning for multi-microgrid system optimization problem in smart grid. Memetic Comp. (2022).
9.  Huang, Y. et al. Resilient distribution networks by microgrid formation using deep reinforcement learning. In IEEE Transactions on Smart Grid , vol. 13  (2022).
10.  Mbuwir, B. V ., Geysen, D., Spiessens, F. &amp; Deconinck, G. Reinforcement learning for control of flexibility providers in a residential microgrid. IET Smart Grid . 3 , 1 (2020).
11.  Mohammed, H., Alabdullah, M. A. &amp; Abido Microgrid energy management using deep Q-network reinforcement learning. Alex. Eng. J. 61 (11), (2022).
12.  Daniel,  J.  B.,  Harrold,  J.,  Cao,  Z.  &amp;  Fan  Renewable  energy  integration  and  microgrid  energy  trading  using  multi-agent  deep reinforcement learning. Appl. Energy 318 , (2022).
13.  She, B., Li, F., Cui, H., Zhang, J. &amp; Bo, R. July, Fusion of Microgrid Control with model-free reinforcement learning: review and vision. In IEEE Trans. Smart Grid . 14 , (2023).
14.  Li, Y. et al. Deep reinforcement learning for smart grid operations: algorithms, applications, and prospects. In Proceedings of the IEEE , vol. 111 (2023).
15.  Massaoudi, M. S., Abu-Rub, H. &amp; Ghrayeb, A. Navigating the landscape of deep reinforcement learning for power system stability control: a review. In IEEE Access , vol. 11 (2023).
16.  Cao, D. et al. Reinforcement learning and its applications in modern power and energy systems: a review. J. Mod. Power Syst. Clean. Energy 8 , (2020).
17.  Yang, T., Zhao, L., Li, W . &amp; Zomaya, A. Y. Reinforcement learning in sustainable energy and electric systems: A survey. Annu. Rev. Control . 2020 (2020).
18.  Somasundaran, N., Radhika, N. &amp; Venkataraman, V . Smart grid test bed based on GSM. Proc. Eng. (2012).
19.  Kiran, P . &amp; Vijaya Chandrakala, K. R. M. New interactive agent based reinforcement learning approach towards smart generator bidding in electricity market with micro grid integration. Appl. Soft Comput. 97 , (2020).
20.  Sowmya Reddy, V. S., Chandan, K., Nimmy, P., Smitha, T. V . &amp; Nagaraja, K. V . An efficient machine learning model for smart grid stability prediction in our prestigious conference: International Conference on Emerging Technologies in Engineering and Science (ICETES) (2023).
21.  Nippun Kumaar, A. A. &amp; Kochuvila, S. Reinforcement learning based path planning using a topological map for mobile service robot. In 2023 IEEE International Conference on Electronics, Computing and Communication Technologies (CONECCT), Bangalore, India (2023).
22.  Shivkumar, S., Amudha, J. &amp; Kumaar, N. A. A. Federated Deep Reinforcement Learning for Mobile Robot Navigation. (2024).

## Author contributions

S.R. and S.B.N. wrote the main manuscript text and S.J.S. and V .S. prepared figures and tables. N.K.A.A. and M.K. guided the work at all stages. All authors reviewed the manuscript.

## Declarations

## Competing interests

The authors declare no competing interests.

## Additional information

Correspondence and requests for materials should be addressed to N.K.A.A.

Reprints and permissions information is available at www.nature.com/reprints.

Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Open Access This article is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License, which permits any non-commercial use, sharing, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if you modified the licensed material. You do not have permission under this licence to share adapted material derived from this article or parts of it. The images or other third party material in this article are included in the article's Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit   h  t  t  p  :  /  /  c  r  e  a  t  i  v  e  c  o  m  m  o n  s .  o r  g /  l i  c e  n  s e  s /  b y  n  c  n  d  / 4  . 0  /  .

© The Author(s) 2024