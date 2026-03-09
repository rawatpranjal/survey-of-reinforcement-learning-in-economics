## Multi/-Agent Reinforcement Learning/: Independent vs/. Cooperative Agents

## Ming Tan

GTE Laboratories Incorporated /4/0 Sylvan Road Waltham/, MA /0/2/2/5/4 tan/@gte/.com

Abstract Intelligent human agents exist in a coop/erative social environment that facilitates learning/. They learn not only by trial/and/-error/, but also through cooperation by sharing instantaneous information/, episodic experience/, and learned knowledge/. The key investigations of this paper are/, /\Given the same number of reinforcement learning agents/, will cooperative agents outperform independent agents who do not communicate during learning/?/" and /\What is the price for such cooperation/?/" Using independent agents as a benchmark/, cooperative agents are studied in following ways/: /(/1/) sharing sensation/, /(/2/) sharing episodes/, and /(/3/) shar/ing learned policies/. This paper shows that /(a/) additional sensation from another agent is bene/cial if it can be used e/ciently/, /(b/) shar/ing learned policies or episodes among agents speeds up learning at the cost of communica/tion/, and /(c/) for joint tasks/, agents engaging in partnership can signi/cantly outperform independent agents although they may learn slowly in the beginning/. These tradeo/s are not just limited to multi/-agent reinforcement learning/.

/1 INTRODUCTION In human society/, learning is an essential component of intelligent behavior/. However/, each individual agent need not learn everything from scratch by its own dis/covery/. Instead/, they exchange information and knowl/edge with each other and learn from their peers or teachers/. When a task is too big for a single agent to handle/, they may cooperate in order to accomplish the task/. Examples are common in non/-human societies as well/. For example/, ants are known to communi/cate about the locations of food/, and to move objects collectively/.

In this paper/, I use reinforcement learning to study in/telligent agents /(Mahadevan /&amp; Connel /1/9/9/1/, Lin /1/9/9/1/, Tan /1/9/9/1/)/. Each reinforcement/-learning agent can in/crementally learn an e/cient decision policy over a state space by trial/-and/-error/, where the only input from an environment is a delayed scalar reward/. The task of each agent is to maximize the long/-term dis/counted reward per action/.

Although most work on reinforcement learning has focused exclusively on single agents/, we can extend reinforcement learning straightforwardly to multiple agents if they are all independent/. They together will outperform any single agent due to the fact that they have more resources and a better chance of receiving rewards/. Recently/, Whitehead /(/1/9/9/1/) has also demon/strated the potential bene/t of multiple /\complete/observing/" cooperative agents over a single agent/. However/, the more practical study is to compare the performance of n independent agents with the one of n cooperative agents and to identify their tradeo/s/. Yet/, no such study has been done previously/. It is the subject of this paper/.

How can reinforcement/-learning agents be coopera/tive/? I identify three ways of cooperation/. First/, agents can communicate instantaneous information such as sensation/, actions/, or rewards/. Second/, agents can communicate episodes that are sequences of /(sen/sation/, action/, reward/) triples experienced by agents/. Third/, agents can communicate learned decision poli/cies/. This paper presents three case studies of multi/agent reinforcement learning involving such coopera/tion and draws some related conclusions that are not limited to multi/-agent reinforcement learning/. The main thesis of this paper is that if cooperation is done intelligently/, each agent can bene/t from other agents/' instantaneous information/, episodic experience/, and learned knowledge /.

Speci/cally/, in case study /1/, I investigate the ability of an agent to utilize sensation input provided by an/other agent/. I demonstrate that sensory information from another agent is bene/cial only if it is relevant and su/cient for learning/. I show one instance where cooperative agents were not able to e/ciently learn decision policies /(compared with independent agents/) due to insu/cient sensation from other agents/.

Case study /2 focuses on sharing learned policies and episodes/. I show that in these cases cooperation speeds up learning/, but does not a/ect asymptotic perfor/mance/. I also provide upper bounds on their communi/cation costs incurred during cooperation/. While shar/ing policies is limited to homogeneous agents/, sharing episodes can be used by heterogeneous agents as long as they can interpret episodes/.

Case study /3 concerns joint tasks which require more than one agent in order to be accomplished/. I demon/strate that cooperative agents who sense their partners or communicate their sensations with each other can learn to perform the tasks at a level that independent agents cannot reach even though they start out slowly/. If a cooperative agent must sense other agents/, the size of its state space can increase exponentially in terms of the number of involved agents/.

Ideally/, intelligent agents would learn when to coop/erate and which cooperative method to use to achieve maximum gain/. This paper is a starting point for the examination of these fundamental open questions/.

/2 RELATED WORK Several multi/-agent learning systems have been de/veloped for speed and//or accuracy/. GTE/'s ILS sys/tem /(Silver et/. al /1/9/9/0/) integrates heterogeneous /(in/ductive/, search/-based/, and knowledge/-based/) learn/ing agents by a central controller through which the agents critique each other/'s proposals/. The MALE system /(Sian /1/9/9/1/) uses an interaction board /(simi/lar to a blackboard/) to coordinate di/erent learning agents/. DLS /(Shaw /&amp; Sikora /1/9/9/0/) adopts a dis/tributed problem/-solving approach to rule induction by dividing data among inductive learning agents/. Re/cently/, Chan and Stolfo /(/1/9/9/3/) advocate meta/-learning for distributed learning/. Most of these systems deal with inductive learning from examples/, rather than autonomous learning agents that involve perception and action/. One exception to this is the complexity analysis of cooperative mechanisms in reinforcement learning by Whitehead /(/1/9/9/1/)/. His main theorem is that n reinforcement/-learning agents who can observe everything about each other can decrease the required learning time at a rate that is /
/(/1 /=n /)/.

Recent work in the /eld of Distributed Arti/cial Intel/ligence /(DAI/) /(Gasser /&amp; Huhns /1/9/8/9/) has addressed the issues of organization/, coordination/, and cooper/ation among agents/, but not for multi/-agent learn/ing/. In the terms of DAI/, my case studies /1 and /2 explore reinforcement learning in collaborative reason/ing systems /(Pope et/. al /1/9/9/2/) which are concerned with coordinating intelligent behavior across multiple self/-su/cient agents/, and my case study /3 studies rein/forcement learning in distributed problem/-solving sys/tems /(Durfee /1/9/8/8/, Tan /&amp; Weihmayer /1/9/9/2/) in which a particular problem is divided among agents that coop/erate and interact to develop a solution/. Unlike DAI/, this work does not deal with issues such as commu/nication language/, agent beliefs/, resource constraint/, and negotiation/. It also mainly focus on homogeneous agents/.

/3 REINFORCEMENT LEARNING Reinforcement learning is an on/-line technique that approximates the conventional optimal control tech/nique known as dynamic programming /(Bellman /1/9/5/7/)/. The external world is modeled as a discrete/-time/, //nite state/, Markov decision process/. Each action is associated with a reward/. The task of reinforcement learning is to maximize the long/-term discounted re/ward per action/.

In this study/, each reinforcement/-learning agent uses the one/-step Q/-learning algorithm /(Watkins /1/9/8/9/)/. Its learned decision policy is determined by the state//action value function/, Q /, which estimates long/term discounted rewards for each state//action pair/. Given a current state x and available actions a i /, a Q/learning agent selects each action a with a probability given by the Boltzmann distribution/: p /( a i j x /) /= e Q /( x/;;a i /) /=T P k /2 actions e Q /( x/;;a k /) /=T /(/1/) where T is the temperature parameter that adjusts the randomness of decisions/. The agent then executes the action/, receives an immediate reward r /, moves to the next state y /.

In each time step/, the agent updates Q /( x/;; a /) by recur/sively discounting future utilities and weighting them by a positive learning rate / /:

<!-- formula-not-decoded -->

Here / /(/0 / / /&lt; /1/) is a discount parameter/, and V /( x /) is given by/:

<!-- formula-not-decoded -->

Note that Q /( x/;; a /) is updated only when taking action a from state x /. Selecting actions stochastically by /(/1/) ensures that each action will be evaluated repeatedly/.

As the agent explores the state space/, its estimate Q improves gradually/, and/, eventually/, each V /( x /) ap/proaches/: E f /1 n /=/1 / n /; /1 r t /+ n g /. Here r t is the reward received at time t due to the action chosen at time t /; /1/. Watkins and Dayan /(/1/9/9/2/) have shown that this Q/-learning algorithm converges to an optimal decision policy for a /nite Markov decision process/.

P

Figure /1/: A /1/0 by /1/0 grid world/.

<!-- image -->

/4 TASK DESCRIPTION All the tasks considered in this study involve hunter agents seeking to capture randomly/-moving prey agents in a /1/0 by /1/0 grid world/, as shown by Figure /1/. On each time step/, each agent /(hunter or prey/) has four possible actions to choose from/: moving up/, down/, left/, or right within the boundary/. Initially/, hunters also make random moves as they have equal Q val/ues/. More than one agent can occupy the same cell/. A prey is captured when it occupies the same cell as a hunter /(in case study /1 and /2/) or when two hunters either occupy the same cell as the prey or are next to the prey /(in case study /3/)/. Upon capturing a prey/, the hunter or hunters involved receive /+/1 reward/. Hunters receive /; /0 /: /1 reward for each move when they do not capture a prey/. Each hunter has a limited visual /eld inside which it can locate prey accurately/. Figure /2 shows a visual /eld of depth /2/. Each hunter/'s sensa/tion is represented by /( x /, y /) where x /( y /) is the relative distance of the closest prey to the hunter according to its x /( y /) axis/. For example/, /(/-/2/, /2/) is a perceptual state when the closest prey is in the lower left corner of the hunter/'s visual /eld /(see Figure /2/)/. If two prey are equally close to a hunter/, only one of them /(chosen randomly/) will be sensed/. If there is no prey in sight/, a unique default sensation is used/.

Each run of each experiment consisted of a sequence of trials/. In the /rst trial of each run/, all agents were given a random location/. Afterwards/, each trial began with only rewarded hunters in random locations/. Each trial ended when the /rst prey was captured/. Each run was given a su/cient number of trials until the decision policies of hunters converged /(i/.e/./, the performance of hunters stabilized/)/. I measured the average number of time steps per trial in training where actions were selected by the Boltzmann distribution/, at intervals of every /5/0 trials/. After convergence/, I also measured the average number of time steps per trial in test where actions were selected by the highest Q value/, over at least /1/0/0/0 trials/. Results were averaged over at least /5 runs/.

The Q/-learning parameters were set at / /= /0 /: /8/, / /= /0 /: /9/, and T /= /0 /: /4/. These values are reasonable for

A perceptual state represented by (-2, 2)

<!-- image -->

Figure /2/: A visual /eld of depth /2/.

these tasks/. Task parameters include the number of prey/, the number of hunters/, and the hunters/' visual//eld depth/.

Without learning/, hunters move randomly with base/line performances for four di/erent prey//hunter tasks given in Table /1/. The table shows the average num/ber of steps for random hunters to capture a prey over /2/0/0 trials/. I also tested the performances of indepen/dently learning hunters for the corresponding tasks/. Table /1 gives their average number of steps to capture a prey in training calculated after a su/cient number of trials/, where the hunters/' visual/-/eld depth was /4/. Clearly/, learning hunters signi/cantly outperform ran/dom hunters/. The real question is whether or not co/operation among learning hunters can further improve their performance/.

/5 CASE /1/: SHARING SENSATION First/, I study the e/ect of sensation from another agent/. To isolate sensing from learning/, I choose the one/-prey//one/-hunter task and add a scouting agent that cannot capture prey/. Later I extend this concept to hunters that perform both scouting and hunting/. I demonstrate that sensory information from another /(scouting/) agent is bene/cial if the information is rel/evant and su/cient for learning/.

The scout makes random moves/. At each step/, the scout send its action and sensation back to the hunter/. Assume that the initial relative location between the scout and the hunter is known/. Therefore/, the hunter can incrementally update the scout/'s relative location and also compute the location of the prey sensed by the scout/. For example/, if the relative locations of a prey to the scout /(known/) and the scout to the hunter /(sensed/) are /(/-/2/, /2/) and /(/2/, /5/) respectively/, then the relative location of the prey to the hunter is /(/0/, /7/)/. To keep the same dimension of a state representation /(i/.e/./, still use /( x /, y /)/)/, I combine sensation inputs from the hunter and the scout as follows/: use the hunter/'s sensation /rst/, if the hunter cannot sense any prey/, then use the scout/'s sensation/.

Table /2 shows the average numbers of steps to capture

1 Althonch the

N-of-prey/N-of-hunters

Hunter Visual Depth

Random hunters

2

Learning hunters

2

2

2

Scout Visual Depth

1/1

123.08

1/2

56.47

no scouting

25.32

12.21

2

3

4

1/2 (joint task)

Average Steps to Capture a Prey

Training

354.45

47.14 ($1.28)

119.17

46.33 (‡1.39)

39.78 (#1.06)

32.67 (#1.03)

2/2 (joint task)

Test

224.92

49.49 (₫1.60)

100.61

42.91 (#1.48)

32.08 (#1.22)

25.07 (#0.89)

Table /1/: Average Number of Steps to Capture a Prey/: Random vs/. Independently Learning Hunters/.

| N/-of/-prey//N/-of/-hunters   | /1///1       | /1///2     | /1///2 /(joint task/)   | /2///2 /(joint task/)   |
|-------------------------------|--------------|------------|-------------------------|-------------------------|
| Random hunters                | /1/2/3/./0/8 | /5/6/./4/7 | /3/5/4/./4/5            | /2/2/4/./9/2            |
| Learning hunters              | /2/5/./3/2   | /1/2/./2/1 | /1/1/9/./1/7            | /1/0/0/./6/1            |

Table /2/: Scouting vs/. No Scouting/.

| Hunter Visual Depth   | Scout Visual Depth   | Average Steps to Capture a Prey   | Average Steps to Capture a Prey   |
|-----------------------|----------------------|-----------------------------------|-----------------------------------|
|                       |                      | Training                          | Test                              |
| /2                    | no scouting          | /4/7/./1/4 /( / /1/./2/8/)                                   | /4/9/./4/9 /( / /1/./6/0/)                                   |
| /2                    | /2                   | /4/6/./3/3 /( / /1/./3/9/)                                   | /4/2/./9/1 /( / /1/./4/8/)                                   |
| /2                    | /3                   | /3/9/./7/8 /( / /1/./0/6/)                                   | /3/2/./0/8 /( / /1/./2/2/)                                   |
| /2                    | /4                   | /3/2/./6/7 /( / /1/./0/3/)                                   | /2/5/./0/7 /( / /0/./8/9/)                                   |

a prey in training after /2/0/0/0 trials and the ones in test after convergence with or without a scout/. Their /9/0/% con/dence intervals calculated by a t/-test are listed in the parentheses/. The hunter with a scout took fewer steps in both training and test to capture a prey than the one without/. /1 As the scout/'s visual//eld depth increases/, the di/erence in their perfor/mances becomes larger/. This observation held when the hunter/'s visual/-/eld depth was given other values /(other than /2/)/. Based on this state representation/, the maximumnumber of perceptual states in the /1/0 by /1/0 grid world is /4/4/2 /(/= /(/2 /-/1/0/+/1/) /2 /+/1/)/. After introducing a scout/, the size of the state space for the hunter was e/ectively increased from /2/6 /(/= /5 /2 /+ /1/) to /4/4/2/. This increase was traded for extra sensory information and paid o/ in the end/. In fact/, when the scout/'s visual//eld depth was /4/, no obvious slowdown was observed after only /5/0 trials/.

Once establishing the bene/t of additional sensory in/formation from a scout/, I then extended this concept to the one/-prey//two/-hunter task with each hunter act/ing as a scout for the other hunter/. Table /3 gives the similar measures for both independent and mutual/scouting agents/. Their /9/0/% con/dence intervals cal/culated by a t/-test and the resulting t/-test compar/isons within each pair are given in the parentheses/. As their visual/-/eld depth increases/, /(a/) both indepen/dent and mutual/-scouting agents take fewer and fewer steps to capture a prey/;; /(b/) mutual/-scouting agents gradually outperform independent agents/;; and /(c/) the advantage of mutual/-scouting agents over independent agents shows up sooner in test than in training/. As an

/1 Although the average steps of the hunter in training with a scout whose visual/-/eld depth was /2 /(/= /4/6/./3/3/) is less than the one of the hunter without a scout /(/= /4/7/./1/4/)/, the di/erence is not signi/cant according to the /.

example/, when the visual/-/eld depth was /4/, mutual/scouting hunters took/, on the average/, /8/./8/3 steps in test to capture a prey comparing with /1/1/./5/3 steps for independent hunters/. However/, when the visual//eld depth was limited to /2/, sharing sensory informa/tion hindered training /, because a short/-sighted scout/ing hunter could not stay with a prey long enough for the other hunter to learn to catch up with the prey/. This suggests that sensory information from another agent should be used prudently/, and extra/, insu/cient information can interfere with learning/. Scouting also incurs communication cost/. The information commu/nicated from a mutual/-scouting agent to another agent per step is bounded by the size /(in bits/) of its sensa/tion and action representation/. In this experiment/, it is /2 log /2 /(/2 V depth /+/1/)/+/2 where V depth is the visual/-/eld depth/.

/6 CASE /2/: SHARING POLICIES OR EPISODES Assume that agents do not share sensation/. If each agent is adequate to accomplish a task /(e/.g/./, each hunter can capture a prey by itself/)/, is cooperation among agents still useful/? I studied several ways of sharing learned policies and episodes in the one/prey//two/-hunter task/. Hunters can either /(/1/) use the same decision policy or /(/2/) exchange their individual policies at various frequencies/. Episodes can be ex/changed /(a/) among peer hunters or /(b/) between peer and expert hunters/. I will show that such cooperative agents can speed up learning/, measured by the aver/age number of steps in training/, even though they will eventually reach the same asymptotic performance as independent agents/. This study presents the experi/mental results when the hunters/' visual/-/eld depth is

•о.

2T accumo 4h,

Independent agents

Mutual-scouting agents

Independent agents

Mutual-scouting agents

Independent agents

Mutual-scouting agents

Average Steps to Capture a Prey

Training

Test

20.38 ($0.57)

25.20 (#0.79) (worse)

14.65 ($0.53)

14.02 (#0.75) (same)

24.04 (#1.00)

24.52 (#1.24) (same)

16.04 ($0.56)

12.98 (#0.65) (better)

Table /3/: Two Independent Agents vs/. Two Mutual/-Scouting Agents/.

Visual Depth

4

Figure /3/: Independent agents vs/. same/-policy agents/.

|                         | Visual Depth   | Average Steps to Capture a Prey   | Average Steps to Capture a Prey   |
|-------------------------|----------------|-----------------------------------|-----------------------------------|
|                         |                | Training                          | Test                              |
| Independent agents      | /2             | /2/0/./3/8 /( / /0/./5/7/)                                   | /2/4/./0/4 /( / /1/./0/0/)                                   |
| Mutual/-scouting agents | /2             | /2/5/./2/0 /( / /0/./7/9/) /(worse/)                                   | /2/4/./5/2 /( / /1/./2/4/) /(same/)                                   |
| Independent agents      | /3             | /1/4/./6/5 /( / /0/./5/3/)                                   | /1/6/./0/4 /( / /0/./5/6/)                                   |
| Mutual/-scouting agents | /3             | /1/4/./0/2 /( / /0/./7/5/) /(same/)                                   | /1/2/./9/8 /( / /0/./6/5/) /(better/)                                   |
| Independent agents      | /4             | /1/2/./2/1 /( / /0/./6/5/)                                   | /1/1/./5/3 /( / /0/./6/1/)                                   |
| Mutual/-scouting agents | /4             | /1/1/./0/5 /( / /0/./5/6/) /(better/)                                   | /8/./8/3 /( / /0/./7/8/) /(better/)                                   |

<!-- image -->

/4/. The conclusions when the visual/-/eld depth is /2 or /3 are similar to /4/.

One simple way of cooperating is that hunters use the same decision policy/. Although each hunter updates the same policy independently/, the rate of updating the policy is multiplied by the number of hunters per step/. Figure /3 shows that when two hunters used the same policy/, they converged much quicker than two independent hunters did/. The average information communicated by each same/-policy hunter per step is bounded by the number of the bits needed to describe a sensation/, an action and a reward/. /2 In this experi/ment/, it is /2 log /2 /(/2 V depth /+ /1/) /+ /3/.

/2 I assume that only one agent keeps a decision policy/. At each step/, the rest of the involved agents send their current sensation to the policy/-keeping agent/, receive cor/responding actions in return/, and then send the rewards of their actions back to the policy/-keeping agent/.

Figure /4/: Independent agents vs/. policy/-averaging agents/.

<!-- image -->

If agents perform the same task/, their decision policies during learning can di/er because they may have ex/plored the di/erent parts of a state space/. Two hunters can complement each other by exchanging their poli/cies and use what the other agent had already learned for its own bene/t/. Assume that each agent can si/multaneously send its current policy to other agents/, I adopted the following policy assimilation/: agents aver/age their policies at certain frequency/. Figure /4 shows the performance results when two hunters averaged their policies at every /1/0 steps/, /5/0 steps/, or /2/0/0 steps/. All of them converged quicker than two independent hunters/. One interesting observation is that when the visual/-/eld depth was /4/, the best frequency was ev/ery /1/0 steps /(see Figure /4/) while when the visual/-/eld depth was /2/, the best frequency was every /5/0 steps /(not shown here/)/. In general/, the information commu/nicated by each policy/-exchanging hunter per step is bounded by /( N /; /1/) / P / F where N is the number

Figure /5/: Independent agents vs/. episode/-exchanging agents/.

<!-- image -->

of participating hunters/, P is the size of a policy /(i/.e/./, number of perceptual states /-number of actions /-number of bits needed to represent a sensation/, an ac/tion and a Q value/)/, and F is the frequency of policy exchanging/. When P or F is large/, communication can be costly/. On the other hand/, unlike same/-policy agents/, a policy/-exchanging agent can be selective in assimilating another agent/'s policy/. For example/, an agent could adopt another agent/'s decision only when it did not have con/dence in certain actions/.

Instead of sharing learned knowledge such as a pol/icy/, agents can share their episodes/. An episode is a sequence of /(sensation/, action/, reward/) triples ex/perienced by an agent/. I used the following episode exchanging/: when a hunter captured a prey/, the hunter transferred its entire solution episode to the other hunter/. The other hunter then /\mentally re/played/" the episode forward to update its own pol/icy/. As a result/, two hunters doubled their learning experience/. The middle curve in Figure /5 shows the speedup in training of two hunters after exchanging their episodes/. The average information communi/cated by each episode/-exchanging hunter per step is bounded by /( N /; /1/) / E where E is the number of bits needed to represent a sensation/, an action/, and a reward /( E /= /2 log /2 /(/2 V depth /+ /1/) /+ /3 in this exper/iment/)/. In addition to the /exibility of assimilating episodes/, exchanging episodes can be used by hetero/geneous reinforcement/-learning agents as long as they can interpret episodes /(e/.g/./, hunters can have di/er/ent visual/-/eld depths/)/. To demonstrate this point/, I let two hunters learn from an expert hunter that al/ways moves towards the prey using the shortest path/. Figure /5 shows signi/cant improvement for the two

Figure /6/: Summary/.

<!-- image -->

novice hunters when the episodes they received were from an expert hunter /(see the bottom curve/)/. Note that an expert hunter could be just another hunter who has already learned hunting skills/. This result demonstrates another bene/t of learning in a coop/erative society where novices can learn quickly from experts by examples /(Lin /1/9/9/1/, Whitehead /1/9/9/1/)/.

Figure /6 summarizes the experimental results of this case study/. Generally speaking/, during the early phase of training/, cooperative learning outperforms indepen/dent learning/, and learning from an expert outper/forms both/. Their di/erences in performance are sta/tistically signi/cant according to t/-tests /. However/, among di/erent ways of cooperation /(excluding learn/ing from an expert/)/, there is no conclusive evidence that one performs better than the others/. In terms of the average information communicated/, if the num/ber of participating agents is limited to /2/, exchanging episodes is comparable to using the same policy/. Ex/changing policy is plausible if the size of a policy is small and the proper frequency of policy exchanging can be determined/.

/7 CASE /3/: ON JOINT TASKS In the previous two case studies/, each hunter can cap/ture prey by itself/. Here/, I study joint tasks where a prey can only be captured by two hunters who ei/ther occupy the same cell as the prey as or are next to the prey/. Hunters cooperate by either passively ob/serving each other or actively sharing their sensations and locations/. I demonstrate that cooperative agents can learn to perform the joint task signi/cantly better than independent agents although they start slowly/.

Figure /7/: Typical runs for the /2/-prey///2/-hunter joint task/.

<!-- image -->

Assume that the hunters/' visual/-/eld depth is /4 /(again/, the conclusions are similar when the visual/-/eld depth is /2 or /3/)/. Let us /rst consider the two/-prey//two/hunter joint task/. When two independent hunters were given this task/, each hunter tended to learn to ap/proach a prey directly/. When both hunters approached the same prey/, they succeeded and received rewards/. When they chased two di/erent prey/, they failed and were penalized/. As training continued/, their perfor/mance /uctuated noticeably around the level of tak/ing/, on the average/, /1/0/1 steps to capture a prey /(see the top curve in Figure /7/)/.

The problem with independent hunters is that they ig/nore each other/. They cannot distinguish the situation where another hunter is nearby from the one far away/. If each hunter can also sense the other hunter/, coopera/tive behavior can emerge from greedy learning hunters/. To address this problem/, I extended the sensation of a hunter to two pairs f /( x prey /;; y prey /)/( x ptn /;; y ptn /) g where /( x prey /;; y prey /) is the relative location /( / visual/-/eld depth/) between a prey and the hunter/, and /( x ptn /;; y ptn /) between a partner and the hunter/. Note that the state space is increased exponentially in terms of the number of agents/. A large state space means more state ex/ploration for a hunter/, and slower learning/. Neverthe/less/, although starting slowly/, such passively/-observing hunters began to overtake independent hunters soon after /4/0/0 trials/, and eventually reduced the average number of steps to only /4/9 /(see the middle curve in Figure /7/)/.

Two hunters can cooperate passively by observing each other in addition to prey/. Given the encouraging re/sults from case study /1/, I proceeded to let hunters also actively share their sensory information/. This

Figure /8/: Typical runs for the /1/-prey///2/-hunter joint task/.

<!-- image -->

means that the state space is further enlarged although there is no increase in the dimension of a state rep/resentation/. This enlargement made initial learning even slower than passively/-observing hunters/. Yet/, mutual/-scouting hunters soon outperformed passively/observing agents after about /1/4/0/0 trials/, and settled down at average /3/9 steps in training /(see the bottom curve in Figure /7/)/. The average number of steps per trial in test for independent/, passively/-observing and mutual/-scouting hunters are /4/9/, /4/2 and /3/4/, respec/tively/.

People may wonder what would happen if there was only one prey in the joint task/. Independent hunters might do well because both hunters can just learn to approach the prey directly/. This/, however/, is not the case/. By knowing where its partner is/, a hunter can learn better approach /(herding/) patterns/. Fig/ure /8 shows the typical runs of the three types of hunters when there was only one prey/. As you can see/, independent agents/, passively/-observing agents/, and mutual/-scouting agents settled down at average /1/1/6/, /8/4/, and /7/6 steps in training/, respectively/. Al/though it is di/cult to analyze the hunters/' speci/c approach patterns/, the fact that cooperative hunters outperformed independent hunters by at least /3/2 steps per trial suggests the existence of such patterns/.

/8 CONCLUSIONS AND FUTURE WORK This paper demonstrates that reinforcement/-learning agents can learn cooperative behavior in a simulated social environment/. Although this paper/'s results are based on simulated prey//hunter tasks/, I believe the conclusions can be applied to cooperation among au/tonomous learning agents in general/. This paper iden/ti/es three ways of agent cooperation/, i/.e/./, by com/municating instantaneous information/, episodic expe/rience/, and learned knowledge/. Speci/cally/, cooper/ative reinforcement/-learning agents can learn faster and converge sooner than independent agents via shar/ing learned policies or solution episodes/. Coopera/tive agents can also broaden their sensation via mu/tual scouting/, and can handle joint tasks via sens/ing other partners/. On the other hand/, this paper also shows that extra sensory information can interfere with learning/, sharing knowledge or episodes comes with a communication cost/, and it takes a larger state space to learn cooperative behavior for joint tasks/. These tradeo/s must be taken into consideration for autonomous and cooperative learning agents/.

This research raises several important issues of multi/agent reinforcement learning/. First/, sensation must be selective because the size of a state space can in/crease exponentially in terms of the number of involved agents/. One heuristic used here is that each hunter only pays attention to the nearest prey /(or hunter/)/. Can such selective sensation strategies be learned/? Second/, on a related issue/, one needs to use general/ization techniques to reduce a state space and improve performance for complex/, noisy tasks/. Third/, learning opportunities are hard to come by for nontrivial coop/erative behavior/. If a prey were smart enough to know how to escape/, it could take a long time for hunters to get enough learning experience/. How can learning be more focused /(e/.g/./, by learning from a teacher/)/? Fourth/, information exchanging among agents incurs communication costs/. Can agents learn to communi/cate/? This learning task gets complicated when the content of communication can be instantaneous infor/mation/, episodic experience/, and learned knowledge/. Fifth/, other cooperative methods need to be explored/. For example/, what if hunters share their action inten/tions to avoid collision/, or share their rewards to sus/tain hunger/? Finally/, can homogeneous agents learn to have job division and to specialize di/erently/? Can heterogeneous agents /(such as scouting agents vs/. blind hunting agents/) learn to cooperate/? These are direc/tions for future work/.

## Acknowledgments

I am grateful to Rich Sutton/, Steve Whitehead/, and Chris Matheus for useful discussions and careful com/ments/. I would like to thank Shri Goyal for his support of this research/.

## References

Bellman/, R/. E/. /(/1/9/5/7/)/. Dynamic Programming /. Princeton University Press/, Princeton/, NJ/.

Chan/, P/. K/. /&amp; Stolfo/, J/. S/. /(/1/9/9/3/)/. Toward parallel and distributed learning by meta/-learning/, Proceed/ings of AAAI Workshop on Knowledge Discovery in Databases/, To appear/.

Durfee/, E/. H/. /(/1/9/8/8/)/. Coordination of Dis/tributed Problem Solvers /, Kluwer Academic Publish/ers/, Boston/.

Gasser/, L/. /&amp; Huhns/, M/. /(/1/9/8/9/)/. Distributed Arti/cial Intelligence/, /2 /, /(eds/./) Pitman/, London/.

Lin/, L/. J/. /(/1/9/9/1/)/. Programmingrobots using reinforce/ment learning and teaching/. In Proceedings of AAAI//9/1/. /(pp/. /7/8/1/-/7/8/6/)/.

Mahadevan/, S/. /&amp; Connel/, J/. /(/1/9/9/1/)/. Automatic pro/gramming of behavior/-based robots using reinforce/ment learning/. In Proceedings of AAAI/-/9/1/. /(pp/. /7/6/8//7/7/3/)/.

Pope/, R/./, Conry/, S/./, /&amp; Meyer/, R/. /(/1/9/9/2/)/. Distributing the planning process in a dynamic environment/. Pro/ceedings of the /1/1th International Workshop on Dis/tributed AI/, Glen Arbor/, MI/.

Shaw/, M/. J/. /&amp; Sikora/, R/. /(/1/9/9/0/)/. A distributed problem/-solving approach to rule induction/: learning in distributed arti/cial intelligence systems/. Technical Report/, CMU/-RI/-TR/-/9/0/-/2/8/, The Robotics Institute/, Carnegie Mellon University/.

Sian/, S/. S/. /(/1/9/9/1/)/. Extending learning to multiple agents/: issues and a model for multi/-agent machine learning/. In Y/. Kodrato/ /(Ed/./)/, Machine Learning /{ EWSL /9/1 /. Springer/-Verlag/, pp/. /4/4/0/-/4/5/6/.

Silver/, B/./, Frawely/, W/./, Iba/, G/./, Vittal/, J/./, /&amp; Bradford/, K/. /(/1/9/9/0/)/. A framework for multi/-paradigmatic learn/ing/. In Proceedings of the Seventh International Con/ference on Machine Learning/, /3/4/8/-/3/5/8/. Austin/, Texas/.

Tan/, M/. /(/1/9/9/1/)/. Cost/-sensitive reinforcement learning for adaptive classi/cation and control/. In Proceedings of AAAI/-/9/1/. /(pp/. /7/7/4/-/7/8/0/)/.

Tan/, M/. /&amp; Weihmayer/, R/. /(/1/9/9/2/)/. Integrating agent/oriented programming and planning for cooperative problem solving/. Proceedings of the AAAI/-/9/2/'s Work/shop on Cooperation among Heterogeneous Intelligent Agents/, San Jose/, CA/,

Watkins/, C/. J/. C/. H/. /(/1/9/8/9/)/. Learning With Delayed Rewards/. Ph/.D/. thesis/, Cambridge University Psy/chology Department/.

Watkins/, C/. J/. C/. H/. /&amp; Dayan/, P/. /(/1/9/9/2/) Technical Note/: Q/-Learning/. Machine Learning /, /8/(/3///4/)/, Kluwer Academic Publishers/.

Whitehead/, S/. D/. /(/1/9/9/1/)/. A complexity analysis of cooperative mechanisms in reinforcement learning/. In Proceedings of AAAI/-/9/1/. /(pp/. /6/0/7/-/6/1/3/)