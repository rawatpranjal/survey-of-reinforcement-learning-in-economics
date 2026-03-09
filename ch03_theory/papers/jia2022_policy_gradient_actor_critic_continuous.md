## Policy Gradient and Actor-Critic Learning in Continuous Time and Space: Theory and Algorithms

†

Yanwei Jia ∗ Xun Yu Zhou

July 26, 2022

## Abstract

We study policy gradient (PG) for reinforcement learning in continuous time and space under the regularized exploratory formulation developed by Wang et al. (2020). We represent the gradient of the value function with respect to a given parameterized stochastic policy as the expected integration of an auxiliary running reward function that can be evaluated using samples and the current value function. This representation effectively turns PG into a policy evaluation (PE) problem, enabling us to apply the martingale approach recently developed by Jia and Zhou (2022a) for PE to solve our PG problem. Based on this analysis, we propose two types of actor-critic algorithms for RL, where we learn and update value functions and policies simultaneously and alternatingly. The first type is based directly on the aforementioned representation, which involves future trajectories and is offline. The second type, designed for online learning, employs the first-order condition of the policy gradient and turns it into martingale orthogonality conditions. These conditions are then incorporated using stochastic approximation when updating policies. Finally, we demonstrate the algorithms by simulations in two concrete examples.

Keywords: Reinforcement learning, continuous time and space, policy gradient, policy evaluation, actor-critic algorithms, martingale.

## 1 Introduction

The essence of reinforcement learning (RL) is 'trial and error': repeatedly trying a policy for actions, receiving and evaluating reward signals, and improving the policy. This manifests three key components of RL: 1) exploration with stochastic policies -to broaden search space via randomization; 2) policy

∗ Department of Industrial Engineering and Operations Research, Columbia University, New York, NY 10027, USA. Email: yj2650@columbia.edu.

† Department of Industrial Engineering and Operations Research &amp; Data Science Institute, Columbia University, New York, NY 10027, USA. Email: xz2574@columbia.edu.

evaluation - to evaluate the value function of a current policy; and 3) policy improvement - to improve the current policy. Numerous algorithms have been proposed in the RL literature, generally categorized into three types: critic-only, actor-only, and actor-critic. Here, an actor refers to a policy that governs the actions, and a critic refers to the value function that evaluates the performance of a policy. The critic-only approach learns a value function to compare the estimated outcomes of different actions and selects the best one following the current value function. The actor-only approach acts directly without learning the expected outcomes of different policies. The actor-critic approach uses an actor simultaneously to improve the policy for generating actions given the current state of the environment and a critic to judge the selected policy and guide improving the actor. See Sutton and Barto (2018) and the references therein for extensive discussions on these methods.

All these algorithms and indeed the general RL study have been hitherto predominantly limited to discrete-time Markov decision processes (MDPs). From a practical point of view, however, the study on continuous-time RL with possibly continuous state and action spaces is more important. The world is inherently continuous-time, and a discrete-time dynamic is just an approximation of the reality by taking a sequence of snapshots of the world over time. As a result, in real life, examples abound in which an agent can or actually needs to interact with a random environment at an ultra-high frequency or outright continuously, e.g., high-frequency stock trading, autonomous driving, and robot navigation. Solving these problems in the discrete-time setting has a notorious drawback: the resulting algorithms are highly sensitive to time discretization; see Tallec et al. (2019); Yildiz et al. (2021) and the references therein.

Theoretically, it remains a largely uncharted territory to study RL in continuous time and spaces. The few existing papers on RL in the continuous setting are mostly restricted to deterministic systems; see, for example Baird (1993); Doya (2000); Munos (2006); Vamvoudakis and Lewis (2010); Fr´ emaux et al. (2013); Lee and Sutton (2021); Yildiz et al. (2021); Kim et al. (2021) where there are no environmental noises. Munos and Bourgine (1997) introduce RL for diffusion-based stochastic control problems without proposing a data-driven solution. Model-based methods such as those in Basei et al. (2020); Szpruch et al. (2021) aim to estimate model coefficients by assuming their known and simple functional forms, which are still prone to model misspecification errors. The RL research for continuous-time diffusion processes with data/sample-driven solutions started only recently. Wang et al. (2020) propose an entropy-regularized stochastic relaxed control framework to study RL in continuous time and space and derive Boltzmann distributions as the generally optimal stochastic policies for exploring the environment and generating actions. In particular, when the problem is linear-quadratic (LQ), namely, when the dynamic is linear and the reward is quadratic in state and action, the optimal policy specializes to Gaussian distributions. Extensions and applications of this work include Wang and Zhou (2020); Dai et al. (2020); Guo et al. (2022); Gao et al. (2022).

While Wang et al. (2020) address the first component of RL - exploration - for the continuous setting, Jia and Zhou (2022a) investigate the second component, namely policy evaluation (PE), aiming at estab-

lishing a theoretical foundation for PE in continuous time and space. They show that PE is theoretically equivalent to maintaining the martingale condition of a specifically defined stochastic process, based on which they propose several online and offline PE algorithms. These algorithms have discrete-time counterparts, such as gradient Monte Carlo, TD( λ ), and GTD, that scatter around in the MDP RL literature. Therefore, through the 'martingale lens', Jia and Zhou (2022a) not only devise new PE algorithms for the continuous case but also interpret and unify many classical algorithms initially designed for MDPs.

The current paper is a continuation of Wang et al. (2020) and Jia and Zhou (2022a), dealing with the third component of RL - policy improvement - in the continuous setting under stochastic policies and, thereby, completing the whole procedure for typical RL tasks. Note that Wang and Zhou (2020) put forth a policy improvement theorem for the special case of a continuous-time mean-variance portfolio selection problem. Furthermore, they show that defining a new policy by maximizing the Hamiltonian of the currently learned value function is proved to achieve a better objective value than the current policy. However, this method, akin to Q-learning for MDPs, has a drawback in requiring the functional form of the Hamiltonian, which in turn involves the knowledge of the environment. 1 Moreover, even if the Hamiltonian is known, maximizing a potentially complex function in high dimensions is computationally demanding or daunting.

In this paper, we take a different approach - that of policy gradient (PG) - which optimizes the value function over a parameterized family of policies. This approach has at least two advantages. First, selecting actions does not involve maximization, and actions are sampled from a known parametric distribution. Second, approximating policies directly facilitates more efficient learning if one has prior knowledge or intuition about the classes of potentially optimal policies (e.g., Gaussian distributions), leading to fewer parameters of the parametric family to be learned.

PGas a general sub-method of RL has a long history that can be traced back to Aleksandrov et al. (1968); Glynn (1990); Williams (1992); Barto et al. (1983); see also Bhatnagar et al. (2009) for more literature review and references therein. PG theorems specifically for MDPs are established in Sutton et al. (1999) and Marbach and Tsitsiklis (2001). Deterministic policy gradient algorithms for semi-MDPs (with discrete time and continuous action space) are developed in Silver et al. (2014) and later extended to incorporate deep neural networks in Lillicrap et al. (2015). Empirically, however, such algorithms tend to be unstable (Duan et al., 2016). Recent studies have focused on stochastic policies with possible entropy regularizers, also known as the softmax method; see for example Mnih et al. (2016); Schulman et al. (2017a,b); Haarnoja et al. (2018).

PG updates and improves policies along the gradient ascent direction, and is often carried out simultaneously and alternatingly with PE. The resulting algorithms for RL, therefore, are essentially actor-critic (AC) ones. Such methods have been successful in many real-world applications, notably AlphaGo (Silver et al., 2017) and dexterous hand manipulation (Haarnoja et al., 2018). But then, again, most PG and AC algo-

1 In a more recent working paper Jia and Zhou (2022b), we develop a (little) q-learning theory to learn essentially the Hamiltonian from samples only, including a general policy improvement theorem.

rithms have been developed for discrete-time MDPs, and many of them in heuristic and ad hoc manners. Existing works on PG and AC in continuous time either focus on deterministic systems (Fr´ emaux et al., 2013; Kim et al., 2021) or study specific models such as linear-quadratic ones (Wang et al., 2021). There are a few papers on applications of specifically designed continuous-time PG- and/or AC-based algorithms. Toy examples include the cart-pole swing-up problem in Doya (2000) and Half-Cheetah in Wawrzynski (2007), both involving physical laws of motion. Real-world applications include portfolio selections (Wang and Zhou, 2020), traffic control (Aragon-G´ omez and Clempner, 2020), autonomous driving (Kiran et al., 2021), and biological neural networks (Fr´ emaux et al., 2013; Zambrano et al., 2015).

In sum, it remains a significant open question to develop general continuous-time PG and AC algorithms and, more importantly, to lay an overarching theoretical underpinning for them. This paper aims to answer these questions by studying PG for a general problem in continuous time and space. Based on this, we develop model-free, data-driven AC algorithms for RL, covering both episodic and continuing, and both online and offline tasks. As its predecessors Wang et al. (2020) and Jia and Zhou (2022a), we develop theory in continuous time and discretize time only at the final algorithmic implementation stage, instead of discretizing time upfront and applying the existing MDP results. Specifically, we conduct our analysis in the stochastic relaxed control framework of Wang et al. (2020) involving distribution-valued stochastic policies. As such, it is necessary to first extend the PE theory of Jia and Zhou (2022a), including the martingale characterization and the resulting methods of the martingale loss function and the martingale orthogonality conditions, from deterministic policies to stochastic ones. This extension is technically non-trivial. Our main contributions, however, are a thorough analysis of the PG and the resulting AC algorithms. More precisely, we deduce the representation of the gradient of the current value function with respect to a parameterized (stochastic) policy. This representation turns out to have the same form as the value function in the PE step, effectively turning PG into an auxiliary PE problem. However, a subtle difficulty is that the corresponding 'auxiliary' reward depends on the Hamiltonian and hence on the functional forms of the system dynamics. We solve this difficulty by integration by parts and Itˆ o's formula, transforming the representation into the expected integration of functions that can be evaluated using samples along with the current value function approximator.

The aforementioned representation is forward-looking. Namely, it is the conditional expectation of a term involving future states. Hence it is suitable for offline learning only. For online learning, we employ the first-order condition of the policy gradient and turn it into martingale orthogonality conditions. These conditions are then incorporated using stochastic approximation when updating policies. Finally, combining the newly developed PG methods in this paper and the PE methods in Jia and Zhou (2022a), we propose several AC algorithms for episodic and continuing/ergodic tasks.

Within the continuous-time stochastic relaxed control framework, there are several studies involving updating policies. For example, Wang and Zhou (2020) consider mean-variance portfolio selection and update policies by maximizing the Hamiltonian. As mentioned earlier, this requires knowledge about the

market and hence the method is essentially model-based. Dai et al. (2020) address the time-inconsistency issue and focus on learning equilibrium policies. Guo et al. (2022) study multi-agent RL by solving an LQ mean-field game. These two papers rely on differentiating with respect to the policy hence are both modelbased methods. In contrast, the present paper provides general model-free (up to the underlying dynamics being diffusion processes) AC algorithms that can be applied in all the above problems. In particular, we apply our algorithms to the mean-variance portfolio selection problem in Wang and Zhou (2020) and show they outperform significantly.

The rest of the paper proceeds as follows. In Section 2, we review Wang et al. (2020)'s entropyregularized, exploratory formulation for RL in continuous time and space, and put forth an equivalent formulation convenient for the subsequent analysis. In Section 3, we develop a theory for PG, based on which we present general AC algorithms. Section 4 is devoted to an extension to ergodic tasks. We demonstrate our algorithms by simulation with two concrete examples in Section 5. Finally, Section 6 concludes. In Appendix, we discuss the connection of our results with their discrete-time counterparts, present some theoretical results used in the simulation studies, and supply proofs of the results stated in the main text.

## 2 Problem Formulation and Preliminaries

Throughout this paper, by convention all vectors are column vectors unless otherwise specified, and R k is the space of all k -dimensional vectors (hence k ˆ 1 matrices). Let A and B be two matrices of the same size. We denote by A ˝ B the inner product between A and B , by | A | the Eculidean/Frobenius norm of A , and write A 2 : ' AA J , where A J is A 's transpose. For a positive semidefinite matrix A , we write ? A ' UD 1 { 2 V J , where A ' UDV J is its singular value decomposition with U, V two orthogonal matrices and D a diagonal matrix, and D 1 { 2 is the diagonal matrix whose entries are the square root of those of D . We use f ' f p¨q to denote the function f , and f p x q to denote the function value of f at x . For any stochastic process X ' t X s , s ě 0 u , we denote by t F X s u s ě 0 the natural filtration generated by X . Finally, for any filtration G ' t G s u s ě 0 and any semi-martingale Y ' t Y s , s ě 0 u , we denote

<!-- formula-not-decoded -->

which is a Hilbert space with the L 2 -norm || κ || L 2 ' ´ E ş T 0 κ 2 t d x Y y t ¯ 1 2 , where x¨y is the quadratic variation of a given process.

Let d, n be given positive integers, T ą 0, and b : r 0 , T s ˆ R d ˆ A ÞÑ R d and σ : r 0 , T s ˆ R d ˆ A ÞÑ R d ˆ n be given functions, where A is the action set. The classical stochastic control problem is to control the state (or feature ) dynamics governed by a stochastic differential equation (SDE), defined on a filtered probability

space ` Ω , F , P W ; t F s W u s ě 0 ˘ along with a standard n -dimensional Brownian motion W ' t W s , s ě 0 u :

<!-- formula-not-decoded -->

where a s stands for the agent's action (control) at time s . The goal of stochastic control is, for each initial time-state pair p t, x q of (1), to find the optimal t F s W u s ě 0 -progressively measurable (continuous) sequence of actions a ' t a s , t ď s ď T u - also called the optimal strategy - that maximizes the expected total reward:

<!-- formula-not-decoded -->

where r is the (expected) running reward function, h is the (expected) lump-sum reward function applied at the end of the planning period T , and β ě 0 is a discount factor that measures the time-value of the payoff or the impatience level of the agent. Note in the above the state process X a ' t X a s , t ď s ď T u also depends on p t, x q . However, to ease notation, here (and similarly in the sequel) we use X a instead of X t,x,a ' t X t,x,a s , t ď s ď T u to denote the solution to SDE (1) with initial condition X a t ' x whenever no ambiguity may arise.

Let L a be the infinitesimal generator associated with the diffusion process governed by (1):

<!-- formula-not-decoded -->

where B ϕ B x P R d is the gradient, and B 2 ϕ B x 2 P R d ˆ d is the Hessian. We make the following assumption to ensure theoretically the well-posedness of the stochastic control problem (1)-(2).

Assumption 1. The following conditions for the state dynamics and reward functions hold true:

- (i) b, σ, r, h are all continuous functions in their respective arguments;
- (ii) b, σ are uniformly Lipschitz continuous in x , i.e., for ϕ P t b, σ u , there exists a constant C ą 0 such that

<!-- formula-not-decoded -->

- (iii) b, σ have linear growth in x , i.e., for ϕ P t b, σ u , there exists a constant C ą 0 such that

<!-- formula-not-decoded -->

- (iv) r and h have polynomial growth in p x, a q and x respectively, i.e., there exists a constant C ą 0 and µ ě 1 such that

<!-- formula-not-decoded -->

Classical model-based stochastic control theory has been well developed (e.g., Fleming and Soner, 2006 and Yong and Zhou, 1999) to solve the above problem, under the premise that the functional forms of b, σ, r, h are all given and known. In the RL setting, however, the agent does not have this knowledge of the environment. Instead, what she can do is 'trial and error' - to try a sequence of actions a ' t a s , t ď s ď T u , observe the corresponding state process X a ' t X a s , t ď s ď T u and collect both a stream of discounted running rewards t e ´ β p s ´ t q r p s, X a s , a s q , t ď s ď T u and a discounted, end-of-period lump-sum reward e ´ β p T ´ t q h p X a T q where β is a given, known discount factor. In the offline setting, the agent can repeatedly try different sequences of actions over the same time period r 0 , T s and record the corresponding state processes and payoffs. In the online setting, the agent updates the actions as she goes, based on all the up-to-date historical observations.

A critical question is how to generate these trial-and-error sequences of actions. The idea is randomization , namely, the agent employs a stochastic policy , which is a probability distribution on the action space, to produce actions according to the current time-state pair. It is important to note that this randomization itself is independent of the underlying Brownian motion W , the random source of the original control problem that stands for the environmental noise. Wang et al. (2020) formulate an RL problem in continuous time and space, incorporating distribution-valued stochastic policies with an entropy regularizer to account for the tradeoff between exploration and exploitation. Specifically, assume the probability space is rich enough to support a random variable Z that is uniformly distributed on r 0 , 1 s and independent of W . We then expand the original filtered probability space to p Ω , F , P ; t F s u s ě 0 q where F s ' F s W \_ σ p Z q and P is now the probability measure on F T . 2 Let π : p t, x q P r 0 , T s ˆ R d ÞÑ π p¨| t, x q P P p A q be a given (feedback) policy, where P p A q is a suitable collection of probability density functions (pdfs). 3 At each time s , an action a s is generated or sampled from the distribution π p¨| s, X s q .

Given a stochastic policy π , an initial time-state pair p t, x q , and an t F s u s ě 0 -progressively measurable action process a π ' t a π s , t ď s ď T u generated from π , the corresponding state process X π ' t X π s , t ď s ď T u follows

<!-- formula-not-decoded -->

defined on p Ω , F , P ; t F s u s ě 0 q . Moreover, following Wang et al. (2020), we add a regularizer to the reward

2 Note that a single uniform random variable Z can produce many independent random variables having density functions. No dynamics are needed for these random variables and they are all independent of each other and of the Brownian motion. The independence means it makes no difference if these variables are given all at once at time 0 or are revealed as time evolves. We opt for the (mathematically speaking) easier construction where these are all defined using one single uniform Z . Meanwhile, P is the product extension from P W ; the two probability measures coincide when restricted to F T W .

3 Here we assume that the action space A is continuous and randomization is restricted to those distributions that have density functions. The analysis and results of this paper can be easily extended to the cases of discrete action spaces and/or randomization with probability mass functions.

function to encourage exploration (represented by the stochastic policy), leading to

<!-- formula-not-decoded -->

where E P is the expectation with respect to (w.r.t.) both the Brownian motion and the action randomization. In the above, p : r 0 , T sˆ R d ˆ A ˆ P p A q ÞÑ R is the regularizer and γ ě 0 a weighting parameter on exploration, also known as the temperature parameter. Wang et al. (2020) take the differential entropy as the regularizer, which corresponds to

<!-- formula-not-decoded -->

Through a law of large number argument, Wang et al. (2020) show that t X π s , t ď s ď T u has the same distribution as the solution to the following SDE, denoted by t ˜ X π s , t ď s ď T u :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

Moreover, the reward function (4) is identical to

<!-- formula-not-decoded -->

Mathematically, (5) and (6) together form a so-called relaxed stochastic control problem where the effect of individually sampled actions has been averaged out (over the randomization/exploration) and, hence, one can focus on how a policy π impacts the distribution of the 'averaged' state ˜ X ; see Wang et al. (2020).

Here, J p t, x ; π q is called the value function of the policy π , and the task of RL is to find

<!-- formula-not-decoded -->

where Π stands for the set of admissible policies. The following gives the precise definition of admissible (feedback) policies.

Definition 1. A policy π ' π p¨|¨ , ¨q is called admissible if

- (i) π p¨| t, x q P P p A q , and π p a | t, x q : p t, x, a q P r 0 , T s ˆ R d ˆ A ÞÑ R is measurable;
- (ii) the SDE (5) admits a unique weak solution (in the sense of distribution) for any initial p t, x q P

<!-- formula-not-decoded -->

- (iii) ş A | r p t, x, a q ` γp ` t, x, a, π p¨| t, x q ˘ | π p a | t, x q d a ď C p 1 ` | x | µ q , @p t, x q where C ą 0 and µ ě 1 are constants;
- (iv) π p a | t, x q is continuous in p t, x q and uniformly Lipschitz continuous in x in the total variation distance, i.e., for each fixed a , ş A | π p a | t, x q´ π p a | t 1 , x 1 q| d a Ñ 0 as p t 1 , x 1 q Ñ p t, x q , and there is a constant C ą 0 independent of p t, a q such that

<!-- formula-not-decoded -->

The conditions required in the above definition, while not necessarily the weakest ones, are to theoretically guarantee the well-posedness of the control problem (5)-(6). This is implied by the following result.

Lemma 1. Let Assumptions 1 hold and π be a given admissible policy. Then the SDE (5) admits a unique strong solution. Moreover, for any µ ě 2 , the solution satisfies the growth condition E P W ' max t ď s ď T | ˜ X π s | µ ˇ ˇ ˇ ˜ X π t ' x  ď C p 1 `| x | µ q for some constant C ' C p µ q . Finally, the expected payoff (6) is finite.

We stress that the solution to (5), t ˜ X π s , t ď s ď T u , is the average of the sample trajectories over infinitely many randomized actions and is in itself not a sample trajectory nor observable. The stochastic relaxed control problem (5)- (6), introduced in Wang et al. (2020), just provides a framework for theoretical analysis. In contrast, the solution to (3), t X π s , t ď s ď T u , is a sample trajectory under a realization of action sequence, t a π s , t ď s ď T u , generated from the policy π , and can indeed be observed. Meanwhile, the difference between (3) and (1) is that actions in the former are randomized: a π is also driven by the randomization and hence is not F t W -adapted. By taking the expectation w.r.t. the action randomization, the expectation in (4) reduces to the expectation in (6). In other words, the problem (5)-(6) is mathematically equivalent to the problem (3)-(4); yet they serve different purposes in our study: the former provides a framework for theoretical analysis of the value function while the latter directly involves observable samples.

Unlike most RL problems that are formulated in an infinite planning horizon (known as continuing tasks ), the current paper mainly focuses on a finite horizon setting (known as episodic tasks ). Finite horizons reflect limited lifespans of real-life tasks, e.g., a trader sells a financial contract with a maturity date, a robot finishes a task before a deadline, and a game player strives to pass a checkpoint given a time limit. If we let T Ñ 8 , under suitable regularity conditions (e.g., when β is large enough) our formulation covers the discounted formulation of the continuing tasks. In addition, later we will consider an ergodic setting as an alternative formulation for continuing tasks in Section 4.

## 3 Theoretical Foundation of Actor-Critic Algorithms

An actor-critic (AC) algorithm consists of two parts: to estimate the value function of a given policy and to update (improve) the policy. In this section, we provide the theoretical analysis to guide devising such an algorithm through policy evaluation (PE) and policy gradient (PG).

## 3.1 Policy Evaluation

Jia and Zhou (2022a) take a martingale perspective to characterize PE as well as its link to solving a linear partial differential equation (PDE) numerically. However, they consider only deterministic policies (i.e. no randomization/exploration), without explicitly involving actions sampled from a stochastic policy. The extension to the case of stochastic policies is non-trivial and specific statements of the corresponding results are important for the subsequent PG and AC algorithm design; so we present and prove them here.

For a given stochastic policy π , J p¨ , ¨ ; π q can be characterized by a PDE based on the celebrated Feynman-Kac formula (cf. Karatzas and Shreve, 2014), which also holds true for the relaxed control setting.

Lemma 2. Assume there is a unique viscosity solution v P C ` r 0 , T s ˆ R d ˘ to the following PDE:

<!-- formula-not-decoded -->

with the terminal condition v p T, x q ' h p x q , x P R d , which satisfies | v p t, x q| ď C p 1 ` | x | µ q for a constant C ą 0 and µ ě 1 . Then v is the value function, that is, v p t, x q ' J p t, x ; π q for all p t, x q P r 0 , T q ˆ R d .

To avoid unduly technicalities, we assume throughout this paper that the value function J P C 1 , 2 ` r 0 , T qˆ R d ˘ X C ` r 0 , T s ˆ R d ˘ . There is a rich literature on conditions ensuring the unique existence and regularity of the viscosity solution to the type of equations like (8); but see Tang et al. (2021) for some latest results.

The following is the main theoretical result underpinning PE, extended from the setting of deterministic feedback policies in Jia and Zhou (2022a) to that of stochastic policies.

Theorem 1. A function J p¨ , ¨ ; π q is the value function associated with the policy π if and only if it satisfies terminal condition J p T, x ; π q ' h p x q , and for any initial p t, x q P r 0 , T q ˆ R d :

<!-- formula-not-decoded -->

is an p F ˜ X π , P W q -martingale on r t, T s . Moreover, it is also equivalent to the martingale orthogonality condition:

<!-- formula-not-decoded -->

for any ξ P L F X π ` r 0 , T s ; J p¨ , X π ¨ ; π q ˘ .

<!-- formula-not-decoded -->

In the above theorem, ξ is called a test function by convention, although in general it is actually a stochastic process.

In RL, one typically employs function approximation for learning functions of interest. Specifically, for PE, one uses a family of parameterized functions J θ ' J θ p¨ , ¨ ; π q on r 0 , T s ˆ R d to approximate J , where θ P Θ Ď R L θ , and the problem is reduced to finding the 'best' (in some sense) θ . We make the following assumption on these function approximators to be used. (Henceforth we may drop π from J θ p¨ , ¨ ; π q whenever no ambiguity arises.)

Assumption 2. For all θ P Θ , J θ P C 1 , 2 ` r 0 , T q ˆ R d ˘ X C ` r 0 , T s ˆ R d ˘ and satisfies the polynomial growth condition in x . Moreover, J θ p t, x q is a smooth function in θ with B J θ B θ , B 2 J θ B θ 2 P C 1 , 2 ` r 0 , T qˆ R d ˘ X C ` r 0 , T sˆ R d ˘ satisfying the polynomial growth condition in x .

Thanks to the martingale characterization in Theorem 1, the PE algorithms developed in Jia and Zhou (2022a) can be adapted to the current setting in a straightforward manner. We now summarize them.

- (i) Minimize the martingale loss function (offline):

<!-- formula-not-decoded -->

This objective corresponds to the gradient Monte-Carlo algorithm for discrete MDPs (Sutton and Barto, 2018).

- (ii) Solve the martingale orthogonality condition (online/offline):

<!-- formula-not-decoded -->

This objective corresponds to various (semi-gradient) TD algorithms and their variants for MDPs (Sutton, 1988; Bradtke and Barto, 1996), depending on the choices of the test function ξ .

- (iii) Minimize a quadratic form of the martingale orthogonality condition (online/offline):

<!-- formula-not-decoded -->

where A is a positive definite matrix of a suitable size. Typical choices are A ' I or A ' ` E P r ş T 0 ξ t ξ J t d t s ˘ ´ 1 .

This objective corresponds to the gradient TD algorithms and their variants for MDPs (Sutton et al., 2008, 2009; Maei et al., 2009).

In the above, the choice of the parametric family J θ may be guided by exploiting some special structure of the underlying problem; see Wang and Zhou (2020) for an example. More general choices include linear combinations of some basis functions or neural networks. On the other hand, common choices of the test functions are ξ t ' B J θ B θ p t, X π t q or ξ t ' ş t 0 λ s ´ t B J θ B θ p s, X π s q d s . Refer to the aforementioned references for details, and in particular to Jia and Zhou (2022a) for the continuous setting. Finally, when implementing these algorithms we need to discretize time, and the convergence when the mesh size goes to zero is established in Jia and Zhou (2022a), which can be readily extended to the current setting.

## 3.2 Policy Gradient

Given an admissible policy, suppose we have carried out the PE step and obtained an estimate of the corresponding value function. The next step is PG, namely, to estimate the gradient of the (learned) value function w.r.t. the policy. Specifically, let π φ be a parametric family of policies with the parameter φ P Φ Ă R L φ . We aim to compute the policy gradient g p t, x ; φ q : ' B B φ J p t, x ; π φ q P R L φ at the current time-state pair p t, x q . Here and throughout we always assume π φ is an admissible policy.

Based on the PDE characterization (8) of the value function, we take the derivative in φ on both sides of (8), with v p t, x q replaced by J p t, x ; π φ q , to get a new PDE satisfied by g p t, x ; φ q :

<!-- formula-not-decoded -->

where q p t, x, a, φ q ' B B φ p ` t, x, a, π φ p¨| t, x q ˘ that maps r 0 , T s ˆ R d ˆ A ˆ Φ to R L φ . Note that (10) is a system of L φ equations, and L a g denotes applying the operator L a to each component of the R L φ -valued function g p¨ , ¨ ; φ q .

Define

<!-- formula-not-decoded -->

which is again a function that maps r 0 , T s ˆ R d ˆ A ˆ Φ to R L φ . Then (10) can be written as

<!-- formula-not-decoded -->

Observe that (11) has the similar form to (8). Thus a Feynman-Kac formula (similar to Lemma 2) represents g as

<!-- formula-not-decoded -->

Therefore, computing PG boils down mathematically to a PE problem with a different reward function. Indeed, the task here is much easier because we only need to compute the function value , g p t, x ; φ q , via (12) at some p t, x q along a sample trajectory, instead of learning the entire function g p¨ , ¨ ; φ q as in PE. However, unlike a normal PE problem, the new reward function ˇ r involves the operator L a applied to J which can not be observed nor computed without the knowledge of the environment.

The remedy to overcome this difficulty rests with Itˆ o's lemma and martingality. We now provide an informal argument for explanation before presenting the formal result. Suppose at time t , an action a is generated from π φ p¨| t, X t q and applied to the system within a small time window r t, t ` ∆ t s . Apply Itˆ o's lemma to obtain

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

Since the stochastic integral w.r.t. the d W term above is a martingale (under suitable regularity conditions), such a term, even if unknown, does not contribute to the expectation and thus can be ignored. As a result, ˇ r can be incrementally estimated based on observations of samples and the learned value function.

Before stating the main result of this paper, we impose the following technical conditions on the policy approximators.

Assumption 3. π φ p a | t, x q is smooth in φ P Φ for all p t, x, a q . Moreover,

<!-- formula-not-decoded -->

for all p t, x, φ q , where C ą 0 , µ ě 1 are constants. Furthermore, ş A | B B φ log π φ p a | t, x q| 2 π φ p a | t, x q d a is continuous in p t, x q for all φ P Φ .

Theorem 2. Given an admissible parameterized policy π φ , its policy gradient g p t, x ; φ q ' B B φ J p t, x ; π φ q admits the following representation:

<!-- formula-not-decoded -->

Once again, all the terms inside the expectation above are all computable given samples (including action trajectories and the corresponding state trajectories) on r t, T s , together with an estimated value function J (obtained in the previous PE step). Note that the expectation (14) gives the gradient of the value function w.r.t. any policy, which is not 0 in general.

Observing (14) more closely, we can write g p t, x ; φ q ' g 1 p t, x ; φ q ` g 2 p t, x ; φ q where and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The integrand in the expression of g 1 is the discounted derivative of the log-likelihood (log-pdf) that determines the direction, multiplied by a scalar term. This scalar term is actually the TD error in the continuous setting (Jia and Zhou, 2022a) that also appears in the martingale orthogonality condition (9). Note that g 1 p t, x ; φ q ‰ 0 in general, because B B φ log π φ p a π φ s | s, X π φ s q depends on the realization of a π φ s , and hence is not F X π φ s -measurable and does not qualify as a test function ξ in Theorem 1. On the other hand, g 2 comes entirely from the regularizer and vanishes should the latter be absent.

There are two equivalent forms of the representation (14), which can be used to add more flexibilities in designing PG algorithms and to optimize their performance. The first one is to add a 'baseline' action-

independent function B p t, x q to the integrand in (14). Precisely, it follows from a π φ s ' π φ p¨| s, X π φ s q that

<!-- formula-not-decoded -->

Hence, an alternative representation of (14) is

<!-- formula-not-decoded -->

Including such a baseline function in the representation of PG goes back at least to Williams (1992). Sutton et al. (2000) and Zhao et al. (2011) find that adding an appropriate baseline function can reduce the variance of the learning process. In particular, a common choice of baseline function, though not theoretically optimal, is the current value function, which leads to the so-called advantage AC algorithms (Degris et al., 2012; Mnih et al., 2016). Interestingly, without including any exogenous baseline function, the PG algorithms out of (14) are exactly the continuous-time versions of the advantage AC algorithms. As such, we do not add other baseline functions for designing our algorithms below. More connections to the representation of policy gradient in discrete-time and detailed discussions of the baseline function can be found in Appendix A.

The second alternative form of (14) is to add an admissible test function to the derivative of the loglikelihood. Specifically, suppose ζ P L 2 F X π φ ` r 0 , T s ; J p¨ , X π φ ¨ ; π φ q ˘ is an R L φ -valued process. Then based on Theorem 1, the policy gradient can also be represented by

<!-- formula-not-decoded -->

As discussed before, we do not use (14) to approximate the function g p¨ , ¨ ; φ q . Rather, at any current time-state p t, x q , (14) gives the gradient of J p t, x ; φ q in φ so that we can update φ in the most promising

direction (based on the gradient ascent algorithm) to improve the value of J . However, the right hand side of (14) involves only the future trajectories from t ; so Theorem 2 works only for the offline setting.

To treat the online case, assume that φ ˚ is the optimal point of J p t, x ; π φ q for any p t, x q and that the first-order condition holds (e.g., when φ ˚ is an interior point). 4 Then g p t, x ; φ ˚ q ' 0. It thus follows from (10) that

<!-- formula-not-decoded -->

This is the same type of equation as (8) involved in the Feynman-Kac formula. In the same way as (8) leading to Theorem 1, we can prove the following conclusion.

Theorem 3. If there exists an interior optimal point φ ˚ that maximizes J p 0 , x ; π φ q for any x P R d , then

<!-- formula-not-decoded -->

for any η, ζ P L 2 F X π φ ˚ ` r 0 , T s ; J p¨ , X π φ ˚ ¨ ; π φ ˚ q ˘ .

If we take η s ' e ´ βs , then the right hand side of (18) coincides with g p 0 , x, φ ˚ q . However, though only a necessary condition, (18) contains infinitely many equations with different test functions η . More importantly, besides the flexibility of choosing different sets of test functions, (18) provides a way to derive a system of equations based on only past observations and, hence, enables online learning. For example, by taking η s ' 0 on r t, T s , (18) involves sample trajectories up to only the present time t . Thus, learning the optimal policy either offline or online boils down to solving a system of equations (with suitably chosen test functions) via stochastic approximation to find φ ˚ .

In sum, Theorems 2 and 3 foreshadow two different types of algorithms which we will develop in the next subsection.

4 A theoretically optimal policy π ˚ indeed maximizes J p t, x ; π q for any p t, x q , based on the verification theorem; see Yong and Zhou (1999).

## 3.3 Actor-Critic Algorithms

We now design actor-critic (AC) algorithms by combining the PE and the PG steps. For the former, Jia and Zhou (2022a) develop two methods, those of martingale loss function and martingale orthogonality conditions, to devise several online/offline PE algorithms for the continuous setting. As discussed in Subsection 3.1, one can adopt any of these algorithms that is suitable for the given learning context and computational resource to estimate the value function of any given policy. Here we focus on how to update the policy based on our previous theoretical analysis on PG.

First, in the offline setting where full state trajectories under any given policy can be repeatedly sampled and observed, the gradient of the value function w.r.t. the policy is given by (14), which can be estimated using future samples from any current time-state p t, x q . That is, g p t, x ; φ q is the gradient direction that would maximally improve the total reward at p t, x q .

For online learning, as explained earlier, (14) is no longer implementable. Instead of computing gradients, we turn to (18) for directly solving the optimal policy. Specifically, at any current time t , we choose η s ' 0 for s P r t, T s so that the integral in (18) only utilizes past observations up to t , and hence is computable. Therefore, in the online setting one applies stochastic approximation to solve the optimal condition (18) in order to search for the optimal policy φ ˚ .

Recall that J θ ' J θ p¨ , ¨q , where J θ p t, x q P R , is a family of scalar functions on p t, x q P r 0 , T s ˆ R d parameterized by θ P Θ Ď R L θ , and π φ ' π φ p¨|¨ , ¨q , where π φ p¨| t, x q P P p A q , is a family of pdf-valued policy functions on p t, x q P r 0 , T s ˆ R d parameterized by φ P Φ Ď R L φ . The aim of an AC algorithm is to find the optimal p θ, φ q jointly, by updating the two parameters alternatingly. Note that, although our problem is continuous in time, the final algorithmic implementation requires discretizing time. For simplicity, we use equally spaced mesh grid t k ' k ∆ t , with k ' 0 , ¨ ¨ ¨ , K ' t T ∆ t u .

We now present the following pseudo codes in Algorithms 1 and 2. Algorithm 1 is for offline-episodic learning, where full trajectories are sampled and observed repeatedly during different episodes and p θ, φ q are updated after one whole episode. Algorithm 2 is for online incremental learning, where only the past sample trajectory is available and p θ, φ q are updated in real-time incrementally.

Note that Algorithms 1 and 2 presented here are just for illustrative purpose; there is ample flexibility to devise their variants depending on the specific problems concerned. In particular, the choice of test functions dictates in which sense we approximate the value function and policy. 5 For example, if we take the test functions ξ t ' B J θ B θ p t, X t q , and η t ' e ´ βt , then we have essentially TD(0) AC algorithms. If we take ξ t ' ş t 0 λ t ´ s B J θ B θ p s, X s q d s , ζ t ' ş t ´ ∆ t 0 λ t ´ s B B φ log π φ p a π φ s | s, X s q d s , then we end up with TD( λ ) algorithms (Sutton and Barto, 2018). Moreover, in the PE part of the algorithms we can also use other methods (online

5 See Jia and Zhou (2022a) for detailed discussions on this point for the PE part. Also, to save computational and memory cost of algorithms, we usually choose test functions that can be computed incrementally. For example, in a TD( λ ) algorithm, ξ t k ' ş t k 0 λ t k ´ s B J θ B θ p s, X s q d s « λ ∆ t ξ t k ´ 1 ` B J θ B θ p t k , X t k q ∆ t , and ζ t k « λ ∆ t ζ t k ´ 1 ` B B φ log π φ p a π φ t k ´ 1 | t k ´ 1 , X t k ´ 1 q ∆ t , which can be calculated recursively.

## Algorithm 1 Offline-Episodic Actor-Critic Algorithm

Inputs : initial state x 0 , horizon T , time step ∆ t , number of episodes N , number of mesh grids K , initial learning rates α θ , α φ and a learning rate schedule function l p¨q (a function of the number of episodes), functional form of the value function J θ p¨ , ¨q , functional form of the policy π φ p¨|¨ , ¨q , functional form of the regularizer p ` t, x, a, π p¨q ˘ , functional forms of the test functions ξ p t, x ¨^ t q , ζ p t, x ¨^ t q , and temperature parameter γ .

Required program : an environment simulator p x 1 , r q ' Environment ∆ t p t, x, a q that takes current time-state pair p t, x q and action a as inputs and generates state x 1 at time t ` ∆ t and the instantaneous reward r at time t .

## Learning procedure :

Initialize θ, φ .

for episode j ' 1 to N do

<!-- formula-not-decoded -->

Initialize k ' 0. Observe the initial state x 0 and store x t k Ð x 0 .

Compute and store the test function ξ t k ' ξ p t k , x t 0 , ¨ ¨ ¨ , x t k q , ζ t k ' ζ p t k , x t 0 , ¨ ¨ ¨ , x t k q . Generate action a t k ' π φ p¨| t k , x t k q .

## end while

Apply a t k to the environment simulator p x, r q ' Environment ∆ t p t k , x t k , a t k q , and observe the output new state x and reward r . Store x t k ` 1 Ð x and r t k Ð r . Update k Ð k ` 1.

Compute

<!-- formula-not-decoded -->

Update θ (policy evaluation) by

Update φ (policy gradient) by end for

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Algorithm 2 Online-Incremental Actor-Critic Algorithm

Inputs : initial state x 0 , horizon T , time step ∆ t , number of mesh grids K , initial learning rates α θ , α φ and learning rate schedule function l p¨q (a function of the number of episodes), functional form of the value function J θ p¨ , ¨q , functional form of the policy π φ p¨|¨ , ¨q , functional form of the regularizer p ` t, x, a, π p¨q ˘ , functional forms of the test functions ξ p t, x ¨^ t q , η p t, x ¨^ t q , ζ p t, x ¨^ t q , and temperature parameter γ .

Required program : an environment simulator p x 1 , r q ' Environment ∆ t p t, x, a q that takes current time-state pair p t, x q and action a as inputs and generates state x 1 at time t ` ∆ t and the instantaneous reward r at time t .

## Learning procedure :

Initialize θ, φ .

for episode j ' 1 to 8 do while

Initialize k ' 0. Observe the initial state x 0 and store x t k Ð x 0 .

k

K

do

Compute

ă

test function

<!-- formula-not-decoded -->

'

ξ

t

k

ξ

p

Generate action a t k ' π φ p¨| t k , x t k q .

¨ ¨ ¨

q

'

p

¨ ¨ ¨

q

'

Apply a t k to the environment simulator p x, r q ' Environment ∆ t p t k , x t k , a t k q , and observe the output new state x and reward r . Store x t k ` 1 Ð x and r t k Ð r . Compute

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Update θ (policy evaluation) by

Update φ (policy gradient) by

<!-- formula-not-decoded -->

t

k

, x

0

t

,

, x

t

k

,

η

t

k

η

t

k

, x

0

t

,

, x

t

k

,

and

ζ

t

k

or offline) as summarized in Subsection 3.1.

Finally, we reiterate that the main purpose of this paper is to provide a theoretical foundation to guide designing AC algorithms, instead of comparing which algorithm performs better. As such, we only present the TD-type algorithms for illustration, acknowledging that there are multiple ways to combine PE and the newly developed PG methods to design new learning algorithms.

## 4 Extension to Ergodic Tasks

In this section we extend our results and algorithms to ergodic (long-term average) tasks, which are also commonly studied in the RL literature. The ergodic objective is one possible formulation of continuing tasks, in which a learning algorithm is based on only one single trajectory.

Consider a regularized ergodic objective function

<!-- formula-not-decoded -->

where p is the regularizer and γ ě 0 is the temperature parameter. Note that now the running reward, the regularizer and the policy do not depended on time explicitly due to the stationary nature of ergodic tasks.

One way to study an ergodic task is to connect it to a discounted, infinite horizon problem:

<!-- formula-not-decoded -->

It has been shown that, under suitable conditions, the optimal value function of the discounted infinite horizon problem converges to the optimal ergodic reward as the discount factor β Ñ 0; see, e.g., Borkar and Ghosh (1988, 1990); Bensoussan and Frehse (1992).

Here, we opt for a direct treatment of ergodic problems. According to Sutton and Barto (2018, page 249), ergodic tasks are actually better behaved than continuing tasks with discounting. For a systematic account of classical ergodic control theory in continuous time, see Arapostathis et al. (2012) and the references therein.

We first present the ergodic version of the Feynman-Kac formula.

Lemma 3. Let π ' π p¨|¨q be a given (time-invariant) policy. Suppose there is a function J p¨ ; π q P C 2 p R d q and a scalar V p π q P R satisfying

<!-- formula-not-decoded -->

Then for any t ě 0 ,

<!-- formula-not-decoded -->

We emphasize that the solution to (19) is a pair of p J, V q , where J p¨ ; π q is a function of the state and V p π q P R is a scalar. The long term average of the payoff does not depend on the initial state x nor the initial time t due to the ergodicity, and hence remains a constant as (20) implies. The function J , on the other hand, only represents the first-order approximation of long-run average and is not unique. Indeed, for any constant c , p J ` c, V q is also a solution to (19). We refer to V as the 'value'. Lastly, since the value does not depend on the initial time, we will fix the latter as 0 in the following discussions and applications of ergodic tasks.

Moreover, J p X π t ; π q ` ş t 0 r r p X π s , a π s q ` γp ` X π s , a π s , π p¨| X π s q ˘ ´ V p π qs d s is an p F X π , P q -martingale.

For a given policy π , the PE problem is now to find a function J p¨ ; π q and a value V P R , such that

<!-- formula-not-decoded -->

is a martingale. Following Jia and Zhou (2022a), we can then design online PE algorithms based on the following martingale orthogonality conditions:

<!-- formula-not-decoded -->

for any T ą 0, any initial state x , and any test function ξ P L 2 F X π ` r 0 , T s ; J p X π ¨ ; π q ˘ .

We now focus on PG. Suppose we parameterize the policy by π φ , we aim to estimate B V p π φ q B φ . Taking the derivative in φ in (19), we obtain

<!-- formula-not-decoded -->

Denote q p x, a, φ q : ' B B φ p ` x, a, π φ p¨| x q ˘ ,

<!-- formula-not-decoded -->

and g p x ; φ q : ' B B φ J p x ; π φ q . Then

<!-- formula-not-decoded -->

Therefore, analogous to the case of episodic tasks, B V p π φ q B φ is the value corresponding to the long-term average of a different running reward, according to the ergodic Feynman-Kac formula (Lemma 3); that is

<!-- formula-not-decoded -->

where the last equality is due to

<!-- formula-not-decoded -->

An ergodic task is a continuing task so we are naturally interested in online algorithms only. We can design two algorithms based on the analysis above. The first one follows directly from the representation (22), in which the policy gradient is the expectation of a long-run average and hence can be estimated online incrementally by since it will converge to its stationary distribution as t Ñ8 . 6

<!-- formula-not-decoded -->

Moreover, due to the martingale orthogonality condition (21), we can also add a test function ζ as we

6 To be more specific, the reason why an infinitesimal increment of the (inner) integral can be used as an estimate for the gradient is due to the ergodicity of the state process. The expression of the gradient (22) is the long-time average of the integrand of the inner integral, which converges to its expectation with respect to the stationary measure. On the other hand, the distribution of the integrand itself also converges to its stationary measure. Therefore, the integrand itself becomes an asymptotically unbiased estimate for the gradient as time tends to infinity. For a brief summary of the ergodicity properties, see Sandri´ c (2017). More details can be found in Part III of Meyn and Tweedie (2012).

did in (16). Consequently, the algorithm updates φ by gradient ascent:

<!-- formula-not-decoded -->

The second algorithm applies a test function η and stochastic approximation to solve the optimality condition as in Theorem 3, by updating

<!-- formula-not-decoded -->

Observe the two algorithms above differ by only the presence of the test function η . To illustrate, we describe the second one in Algorithm 3.

## 5 Applications

In this section we report simulation experiments on our algorithms in two applications. The first one is mean-variance portfolio selection in a finite time horizon with multiple episodes of simulated stock price data. The second application is ergodic linear-quadratic control with a single sample trajectory.

## 5.1 Mean-Variance Portfolio Selection

We first review the formulation of the exploratory mean-variance portfolio selection problem proposed by Wang and Zhou (2020). The investment universe consists of one risky asset (e.g. a stock index) and one risk-free asset (e.g. a saving account) whose risk-free interest rate is r . The price of the risky asset is governed by a geometric Brownian motion with mean µ and volatility σ ą 0 on a filtered probability space p Ω , F , P W ; t F t W u 0 ď t ď T q :

<!-- formula-not-decoded -->

Denote by ρ ' µ ´ r σ the Sharpe ratio of the risky asset.

An agent has a fixed investment horizon 0 ă T ă 8 and an initial endowment x 0 . A self-financing portfolio is represented by the real-valued adapted process a ' t a t , 0 ď t ď T u , where a t is the discounted dollar value invested in the risky asset at time t . Then the discounted value of this portfolio satisfies the wealth equation

<!-- formula-not-decoded -->

where e ´ rt S t is the discounted stock price. We stress that the model on the stock price (23) is mainly for theoretical analysis and for generating samples in our simulation; we do not assume that the agent knows

## Algorithm 3 Actor-Critic Algorithm for Ergodic Tasks

Inputs : initial state x 0 , time step ∆ t , initial learning rates α θ , α φ , α V and learning rate schedule function l p¨q (a function of time), functional form of the value function J θ p¨q , functional form of the policy π φ p¨|¨q , functional form of the regularizer p ` x, a, π p¨q ˘ , functional forms of test functions ξ p x ¨^ t q , η p x ¨^ t q , ζ p x ¨^ t q , and temperature parameter γ .

Required program : an environment simulator p x 1 , r q ' Environment ∆ t p x, a q that takes initial state x and action a as inputs and generates a new state x 1 (at ∆ t ) and an instantaneous reward r . Learning procedure :

Initialize θ, φ, V . Initialize k ' 0. Observe the initial state x 0 and store x t k Ð x 0 . loop

<!-- formula-not-decoded -->

Compute

Apply a to the environment simulator p x 1 , r q ' Environment ∆ t p x, a q , and observe the output new state x 1 and reward r . Store x t k ` 1 Ð x 1 .

<!-- formula-not-decoded -->

Update θ and V (policy evaluation) by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Update φ (policy gradient) by

Update x Ð x 1 and k Ð k ` 1. end loop

its parameters.

The agent has the mean-variance preference, namely, she aims to minimize the variance of the discounted value of the portfolio at T while achieving a given level of expected return:

<!-- formula-not-decoded -->

where z is the target value, and the variance and expectation are w.r.t. the probability measure P W .

This problem is not a standard stochastic control problem and cannot be solved directly by the dynamic programming (DP) principle, or any DP-based reinforcement learning algorithms such as Q-learning. This is because the variance term causes time-inconsistency which violates the assumptions of DP. Strotz (1955) discusses three types of agents when facing time-inconsistency. Here, we consider one of them - the so-called pre-committed agent who solves the problem at time 0 and sticks to it afterwards. 7 For this type of agent, to overcome the difficulty of DP not being directly applicable, Zhou and Li (2000) extend the embedding method, initially introduced by Li and Ng (2000) for the discrete-time mean-variance problem, to transform (25) into an equivalent, unconstrained, and expectation-only problem:

<!-- formula-not-decoded -->

where w is the Lagrange multiplier associated with the constraint E r x a T s ' z . This new problem is timeconsistent and therefore can be solved by DP. Once the optimal a ˚ is derived, w can be obtained by the equation E r x a ˚ T s ' z .

In a reinforcement learning framework, Wang and Zhou (2020) allow randomized actions to incorporate exploration. A stochastic policy is denoted by π ' π p¨| t, x q , namely, at any current time-wealth pair p t, x q , the total amount of discounted wealth invested in the stock is a random draw from the distribution with the density function π p¨| t, x q . Under such a policy, we denote by ˜ X π ' t ˜ X π s : t ď s ď T u the solution to the following SDE

<!-- formula-not-decoded -->

which is (5) specializing to the current case.

Moreover, an entropy regularizer is added to incentivize exploration. Mathematically, the entropyregularized mean-variance portfolio choice problem is to solve

<!-- formula-not-decoded -->

7 The other two types are the na¨ ıve one who re-optimizes at any given time and the sophisticated one who seeks subgame perfect Nash equilibria among her-selves at different times. The latter has been well studied in the continuoustime setting in recent years; see e.g. Ekeland and Lazrak (2006); Bj¨ ork et al. (2014); Basak and Chabakauri (2010); Dai et al. (2021). The RL counterpart is studied in Dai et al. (2020).

where z is the target expected terminal wealth, π s ' π p¨| s, ˜ X π s q , t ď s ď T , H is the differential entropy H p π q ' ´ ş A π p a q log π p a q d a , γ is the temperature parameter, and w is the Lagrange multiplier similar to that introduced earlier.

We follow Wang and Zhou (2020) to parameterize the value function by

<!-- formula-not-decoded -->

and parameterize the policy by

<!-- formula-not-decoded -->

where N p¨| α, δ 2 q is the pdf of the normal distribution with mean α and variance δ 2 . These function approximators are derived in Wang and Zhou (2020) by exploiting the special structure of the underlying problem; see also Appendix B1.

There is no running reward from the actions except the regularizer

<!-- formula-not-decoded -->

Note that the regularizer turns out to be independent of the state x . Finally, the discount factor is β ' 0.

From this point on, we depart from Wang and Zhou (2020) and instead apply the methods developed in this paper to solve the problem. We choose the test functions for PE as the following gradients, in accordance with the most popular TD p 0 q algorithm: 8

<!-- formula-not-decoded -->

The PE updating rule is

<!-- formula-not-decoded -->

For the PG part, the gradients of log-likelihood are

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

8 Wang and Zhou (2020) employ a mean-square TD error (MSTDE) algorithm to do PE and a policy improvement theorem to update policies. However, it is shown in Jia and Zhou (2022a) that MSTDE only minimizes the qudratic variation of the martingale, which may not lead to the true solution of PE. As discussed earlier, other PE algorithms proposed in Jia and Zhou (2022a) can also be applied.

<!-- formula-not-decoded -->

and those of the regularizer are

<!-- formula-not-decoded -->

Accordingly, the offline PG updating rule is

<!-- formula-not-decoded -->

The online counterpart of this updating rule is to remove the integral ' ş T 0 ' in the above and use only the resulting increment to update the policy at every time step.

In addition, there is the Lagrange multiplier w we need to learn: we update w based on the same stochastic approximation scheme in Wang and Zhou (2020).

We present our offline and online algorithms as Algorithms 4 and 5 respectively. Then we replicate the simulation study of Wang and Zhou (2020) with the same basic setting: x 0 ' 1, z ' 1 . 4, T ' 1, ∆ t ' 1 252 . Choose temperature parameter γ ' 0 . 1. The batch size m ' 10 for updating the Lagrange multiplier. The learning rate parameters in Wang and Zhou (2020) are set to be α w ' 0 . 05, and α θ ' α φ ' 0 . 0005 with decay rate l p j q ' j ´ 0 . 51 . In our experiment we adopt these learning rate values for the Wang and Zhou (2020) algorithm unless the algorithm does not converge, in which case we tune the initial learning rates to guarantee convergence. For our algorithm, we set α w ' 0 . 05, and α θ ' α φ ' 0 . 1 with decay rate l p j q ' j ´ 0 . 51 and tune the initial learning rate when necessary. The initialization of the parameters θ and φ is set to be all 0 for both algorithms (the initialization is not discussed in Wang and Zhou 2020). In particular, to mimic the real scenario, we choose a reasonable size of the training sample, with length of 20 years. In each iteration, we randomly sample 128 1-year trajectories to update the rest parameters, and we train the model for N ' 2 ˆ 10 4 iterations. We calculate the performance metrics - the mean, variance and Sharpe ratio of the resulting terminal wealth - of the learned policies of both methods with the training set generated from the same distribution. 9 We then repeat the experiment for 100 times and report the standard deviation of each metric.

Tables 1 and 2 present the test results of the algorithm in Wang and Zhou (2020) and the offline Algorithm 4 in this paper respectively, when stock price is generated from geometric Brownian motion under different specifications of the market parameters µ and σ . Our algorithm achieves significantly higher outof-sample average Sharpe ratios in most scenarios. Underperformance of our strategy occurs mainly when

9 Wang and Zhou (2020) report in-sample performance of the last 2000 iterations in the training set but does not present out-of-sample test results.

## Algorithm 4 Offline-Episodic Actor-Critic Mean-Variance Algorithm

Inputs : initial state x 0 , horizon T , time step ∆ t , number of episodes N , number of time grids K , initial learning rates α θ , α φ , α w and learning rate schedule function l p¨q (a function of the number of episodes), and temperature parameter γ .

Required program : a market simulator x 1 ' Market ∆ t p t, x, a q that takes current time-state pair p t, x q and action a as inputs and generates state x 1 at time t ` ∆ t .

## Learning procedure :

Initialize θ, φ, w .

for episode j ' 1 to N do while k ă K do Compute and store the test function ξ t k ' B J θ B θ p t k , x t k ; w q .

Initialize k ' 0. Observe the initial state x and store x t k Ð x .

Generate action a t k ' π φ p¨| t k , x t k q .

Update k Ð k ` 1.

Apply a t k to the market simulator x ' Market ∆ t p t k , x t k , a t k q , and observe the output new state x . Store x t k ` 1 Ð x .

## end while

T Ð . Compute

Store the terminal wealth X p j q x t K

<!-- formula-not-decoded -->

Update θ (policy evaluation) by

Update φ (policy gradient) by

'

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Update w (Lagrange multiplier) every m episodes: if j 0 mod m then end if end for

<!-- formula-not-decoded -->

## Algorithm 5 Online-Episodic Actor-Critic Mean-Variance Algorithm

Inputs : initial state x 0 , horizon T , time step ∆ t , number of episodes N , number of time grids K , initial learning rates α θ , α φ , α w and learning rate schedule function l p¨q (a function of the number of episodes), and temperature parameter γ .

Required program : a market simulator x 1 ' Market ∆ t p t, x, a q that takes current time-state pair p t, x q and action a as inputs and generates state x 1 at time t ` ∆ t .

## Learning procedure :

Initialize θ, φ, w .

for episode j ' 1 to N do while k ă K do Compute and store the test function ξ t k ' B J θ B θ p t k , x t k ; w q .

Initialize k ' 0. Observe the initial state x and store x t k Ð x .

Generate action a t k ' π φ p¨| t k , x t k q .

Compute

Apply a t k to the market simulator x ' Market ∆ t p t k , x t k , a t k q , and observe the output new state x . Store x t k ` 1 Ð x .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Update k Ð k ` 1.

Store the terminal wealth X p j q T Ð x t K .

Update θ (policy evaluation) by

Update φ (policy gradient) by

## end while

Update w (Lagrange multiplier) every m episodes: if j ' 0 mod m then

<!-- formula-not-decoded -->

end if end for

the actual return of the stock is low ( µ ' 0 , ˘ 0 . 1). In those cases, our learned policy yields larger volatility and less stable out-of-sample performance. However, although the average out-of-sample Sharpe ratios of the learned policy are lower than those of Wang and Zhou (2020), the standard deviations of ours are large. Hence, even in those few scenarios it is still statistically inconclusive to determine which method is better.

We further carry out tests to compare Algorithm 4 with the online Algorithm 5. In implementing Algorithm 5, we update parameters at each time step and conduct learning for 20 years, with the same simulated stock prices as in offline learning. The batch size is set to be m ' 1 for updating the Lagrange multiplier. We also repeat the experiment for 100 times to calculate the standard deviation of each metric. The results are presented in Table 3. Compared with Table 2, offline learning outperforms online one in terms of Sharpe ratio in most cases. Moreover, the former is always preferred when it comes to stably reaching the target return (set to be 40% annually in the experiments).

With a given training data set, it is not surprising that offline learning is typically preferred because it allows us to fully use the data set by bootstrapping multiple 1-year episodes. By contrast, online learning pretends the data to come sequentially without storing past data. For example, under our online setting, the 20-year training set only contains 20 complete episodes sequentially to adjust the final terminal wealth level, unlike in the offline setting where we bootstrap multiple 1-year episodes. Therefore, offline learning uses data thoroughly and efficiently. However, other important considerations motivate or even force us to use online learning. First and foremost, data distribution may not be stationary, so offline learning may suffer from overfitting. Second, for large-scale problems, online and incremental learning is more computationally efficient in reducing storage costs and computational time. Finally, there are also computational techniques to store a certain amount of past data to boost the efficiency of online learning, such as the experience replay with off-policy learning (Zhang and Sutton, 2017; Fedus et al., 2020).

## 5.2 Ergodic Linear-Quadratic Control

Consider the ergodic linear-quadratic (LQ) control problem where state responds to actions in a linear way

<!-- formula-not-decoded -->

and the goal is to maximize the long term average payoff

<!-- formula-not-decoded -->

with r p x, a q ' ´p M 2 x 2 ` Rxa ` N 2 a 2 ` Px ` Qa q .

In the entropy-regularized RL formulation, the policy is denoted by π p¨| x q and actions are generated

Table 1: Out-of-sample performance of algorithm proposed in Wang and Zhou (2020) for mean-variance problem when data are generated by geometric Brownian motion. The columns 'Mean' and 'Variance' report the average mean and variance, respectively, of terminal wealth over 100 independent experiments. The column 'Sharpe ratio' reports the corresponding average Sharpe ratio ( Mean ´ 1 ? Variance ). The numbers in the brackets are the standard deviations.

|    µ |   σ | Mean           | Variance        | Sharpe ratio    |
|------|-----|----------------|-----------------|-----------------|
| -0.5 | 0.1 | 1.4 ( 0.015 )  | 0 ( 0.00033 )   | 6.69 ( 0.096 )  |
| -0.3 | 0.1 | 1.4 ( 0.027 )  | 0.01 ( 0.002 )  | 3.59 ( 0.064 )  |
| -0.1 | 0.1 | 1.4 ( 0.11 )   | 0.11 ( 0.059 )  | 1.25 ( 0.02 )   |
|  0   | 0.1 | 1.04 ( 0.028 ) | 0.06 ( 0.057 )  | 0.2 ( 9.2e-05 ) |
|  0.1 | 0.1 | 1.45 ( 0.29 )  | 0.43 ( 0.55 )   | 0.81 ( 0.0057 ) |
|  0.3 | 0.1 | 1.41 ( 0.033 ) | 0.02 ( 0.0032 ) | 3.06 ( 0.046 )  |
|  0.5 | 0.1 | 1.4 ( 0.017 )  | 0 ( 0.00044 )   | 6 ( 0.087 )     |
| -0.5 | 0.2 | 1.4 ( 0.03 )   | 0.01 ( 0.0031 ) | 3.38 ( 0.15 )   |
| -0.3 | 0.2 | 1.41 ( 0.067 ) | 0.05 ( 0.022 )  | 1.81 ( 0.074 )  |
| -0.1 | 0.2 | 1.44 ( 0.3 )   | 0.74 ( 1.3 )    | 0.61 ( 0.0041 ) |
|  0   | 0.2 | 1.04 ( 0.032 ) | 0.27 ( 0.27 )   | 0.1 ( 6.3e-05 ) |
|  0.1 | 0.2 | 1.25 ( 0.076 ) | 0.43 ( 0.23 )   | 0.4 ( 0.0014 )  |
|  0.3 | 0.2 | 1.42 ( 0.089 ) | 0.08 ( 0.04 )   | 1.54 ( 0.055 )  |
|  0.5 | 0.2 | 1.41 ( 0.034 ) | 0.02 ( 0.004 )  | 3.03 ( 0.13 )   |
| -0.5 | 0.3 | 1.4 ( 0.057 )  | 0.03 ( 0.016 )  | 2.25 ( 0.16 )   |
| -0.3 | 0.3 | 1.41 ( 0.14 )  | 0.13 ( 0.12 )   | 1.2 ( 0.057 )   |
| -0.1 | 0.3 | 1.32 ( 0.12 )  | 0.71 ( 0.43 )   | 0.41 ( 0.0023 ) |
|  0   | 0.3 | 1.04 ( 0.031 ) | 0.55 ( 0.58 )   | 0.07 ( 3e-05 )  |
|  0.1 | 0.3 | 1.19 ( 0.11 )  | 0.67 ( 0.49 )   | 0.27 ( 0.0011 ) |
|  0.3 | 0.3 | 1.44 ( 0.14 )  | 0.22 ( 0.21 )   | 1 ( 0.018 )     |
|  0.5 | 0.3 | 1.41 ( 0.055 ) | 0.04 ( 0.016 )  | 2.01 ( 0.13 )   |
| -0.5 | 0.4 | 1.41 ( 0.079 ) | 0.07 ( 0.041 )  | 1.67 ( 0.13 )   |
| -0.3 | 0.4 | 1.43 ( 0.15 )  | 0.28 ( 0.23 )   | 0.86 ( 0.011 )  |
| -0.1 | 0.4 | 1.28 ( 0.12 )  | 1.04 ( 0.76 )   | 0.3 ( 0.0016 )  |
|  0   | 0.4 | 1.04 ( 0.028 ) | 0.85 ( 0.89 )   | 0.05 ( 2.3e-05  |
|  0.1 | 0.4 | 1.17 ( 0.1 )   | 0.93 ( 0.82 )   | 0.2 ( 0.00069 ) |
|  0.3 | 0.4 | 1.46 ( 0.17 )  | 0.44 ( 0.43 )   | 0.74 ( 0.012 )  |
|  0.5 | 0.4 | 1.42 ( 0.082 ) | 0.09 ( 0.046 )  | 1.44 ( 0.058 )  |

Table 2: Out-of-sample performance of offline learning (Algorithm 4) for mean-variance problem when data are generated by geometric Brownian motion. The columns 'Mean' and 'Variance' report the average mean and variance, respectively, of terminal wealth over 100 independent experiments. The column 'Sharpe ratio' reports the corresponding average Sharpe ratio ( Mean ´ 1 ? Variance ). The numbers in the brackets are the standard deviations.

|    µ |   σ | Mean         | Variance            | Sharpe ratio    |
|------|-----|--------------|---------------------|-----------------|
| -0.5 | 0.1 | 1.4 ( 0.012  | ) 0 ( 0.00011 )     | 8.15 ( 0.06 )   |
| -0.3 | 0.1 | 1.4 ( 0.023  | ) 0.01 ( 0.00084 )  | 4.37 ( 0.029 )  |
| -0.1 | 0.1 | 1.41 ( 0.08  | ) 0.09 ( 0.037 )    | 1.37 ( 0.0073 ) |
|  0   | 0.1 | 1.13 ( 0.14  | ) 0.91 ( 0.49 )     | 0.12 ( 0.16 )   |
|  0.1 | 0.1 | 1.51 ( 0.27  | ) 0.47 ( 0.72 )     | 0.84 ( 0.0023 ) |
|  0.3 | 0.1 | 1.41 ( 0.028 | ) 0.01 ( 0.0015 )   | 3.71 ( 0.025 )  |
|  0.5 | 0.1 | 1.4 ( 0.014  | ) 0 ( 0.00016 )     | 7.35 ( 0.055 )  |
| -0.5 | 0.2 | 1.4 ( 0.025  | ) 0.01 ( 0.001 )    | 3.98 ( 0.047 )  |
| -0.3 | 0.2 | 1.4 ( 0.049  | ) 0.04 ( 0.0088 )   | 2.1 ( 0.017 )   |
| -0.1 | 0.2 | 1.53 ( 0.27  | ) 0.93 ( 1 )        | 0.62 ( 0.0012 ) |
|  0   | 0.2 | 1.06 ( 0.15  | ) 2.51 ( 1.6 )      | 0.04 ( 0.094 )  |
|  0.1 | 0.2 | 1.44 ( 0.37  | ) 2.02 ( 1.5 )      | 0.35 ( 0.21 )   |
|  0.3 | 0.2 | 1.42 ( 0.065 | ) 0.06 ( 0.018 )    | 1.78 ( 0.012 )  |
|  0.5 | 0.2 | 1.41 ( 0.029 | ) 0.01 ( 0.0015 )   | 3.58 ( 0.041 )  |
| -0.5 | 0.3 | 1.4 ( 0.04 ) | 0.03 ( 0.0047 )     | 2.54 ( 0.026 )  |
| -0.3 | 0.3 | 1.41 ( 0.088 | ) 0.1 ( 0.049 )     | 1.32 ( 0.007 )  |
| -0.1 | 0.3 | 1.43 ( 0.33  | ) 1.79 ( 1.9 )      | 0.37 ( 0.18 )   |
|  0   | 0.3 | 1.03 ( 0.12  | ) 3.46 ( 2.3 )      | 0.02 ( 0.064 )  |
|  0.1 | 0.3 | 1.26 ( 0.36  | ) 2.78 ( 2.3 )      | 0.18 ( 0.2 )    |
|  0.3 | 0.3 | 1.44 ( 0.13  | 0.17 ( 0.14 )       | 1.12 ( 0.012 )  |
|  0.5 | 0.3 | 1.41 ( 0.048 | ) ) 0.03 ( 0.0076 ) | 2.28 ( 0.02 )   |
| -0.5 | 0.4 | 1.41 ( 0.061 | ) 0.05 ( 0.017 )    | 1.8 ( 0.01 )    |
| -0.3 | 0.4 | 1.43 ( 0.15  | ) 0.24 ( 0.23 )     | 0.93 ( 0.014 )  |
| -0.1 | 0.4 | 1.31 ( 0.44  | ) 3.13 ( 4.8 )      | 0.25 ( 0.17 )   |
|  0   | 0.4 | 1.02 ( 0.096 | ) 3.77 ( 2.7 )      | 0.01 ( 0.049 )  |
|  0.1 | 0.4 | 1.14 ( 0.36  | ) 3.6 ( 2.8 )       | 0.1 ( 0.18 )    |
|  0.3 | 0.4 | 1.53 ( 0.34  | ) 0.74 ( 1.3 )      | 0.73 ( 0.0024 ) |
|  0.5 | 0.4 | 1.42 ( 0.076 | ) 0.07 ( 0.032 )    | 1.6 ( 0.014 )   |

Table 3: Out-of-sample performance of online learning (Algorithm 5) for mean-variance problem when data are generated by geometric Brownian motion. The columns 'Mean' and 'Variance' report the average mean and variance, respectively, of terminal wealth over 100 independent experiments. The column 'Sharpe ratio' reports the corresponding average Sharpe ratio ( Mean ´ 1 ? Variance ). The numbers in the brackets are the standard deviations.

|    µ |   σ | Mean           | Variance           | Sharpe ratio    |
|------|-----|----------------|--------------------|-----------------|
| -0.5 | 0.1 | 1.78 ( 0.0082  | ) 0.01 ( 3e-04 )   | 7.43 ( 0.04 )   |
| -0.3 | 0.1 | 1.55 ( 0.0077  | ) 0.02 ( 0.00034 ) | 3.84 ( 0.027 )  |
| -0.1 | 0.1 | 1.14 ( 0.022 ) | 0.01 ( 0.0039 )    | 1.24 ( 0.0073 ) |
|  0   | 0.1 | 1.01 ( 0.0055  | ) 0 ( 0.0016 )     | 0.12 ( 0.16 )   |
|  0.1 | 0.1 | 1.26 ( 0.052 ) | 0.09 ( 0.032 )     | 0.85 ( 0.012 )  |
|  0.3 | 0.1 | 1.83 ( 0.021   | ) 0.04 ( 0.0023 )  | 4.31 ( 0.056 )  |
|  0.5 | 0.1 | 1.92 ( 0.017   | ) 0.01 ( 0.00032 ) | 10.82 ( 0.15 )  |
| -0.5 | 0.2 | 1.77 ( 0.015   | ) 0.04 ( 0.0023 )  | 3.65 ( 0.038 )  |
| -0.3 | 0.2 | 1.54 ( 0.018 ) | 0.08 ( 0.0036 )    | 1.89 ( 0.025 )  |
| -0.1 | 0.2 | 1.14 ( 0.043 ) | 0.05 ( 0.028 )     | 0.62 ( 0.006 )  |
|  0   | 0.2 | 1.01 ( 0.01 )  | 0.01 ( 0.015 )     | 0.04 ( 0.091 )  |
|  0.1 | 0.2 | 1.12 ( 0.072 ) | 0.12 ( 0.09 )      | 0.36 ( 0.19 )   |
|  0.3 | 0.2 | 1.79 ( 0.048 ) | 0.17 ( 0.019 )     | 1.94 ( 0.052 )  |
|  0.5 | 0.2 | 1.92 ( 0.033 ) | 0.04 ( 0.0032 )    | 4.8 ( 0.11 )    |
| -0.5 | 0.3 | 1.76 ( 0.02 )  | 0.1 ( 0.0073 )     | 2.36 ( 0.035 )  |
| -0.3 | 0.3 | 1.52 ( 0.035 ) | 0.18 ( 0.018 )     | 1.23 ( 0.021 )  |
| -0.1 | 0.3 | 1.12 ( 0.061   | ) 0.11 ( 0.079 )   | 0.39 ( 0.14 )   |
|  0   | 0.3 | 1.01 ( 0.013   | ) 0.05 ( 0.051 )   | 0.02 ( 0.063 )  |
|  0.1 | 0.3 | 1.1 ( 0.12 )   | 2.27 ( 20 )        | 0.17 ( 0.21 )   |
|  0.3 | 0.3 | 1.44 ( 0.049   | ) 0.18 ( 0.032 )   | 1.05 ( 0.018 )  |
|  0.5 | 0.3 | 1.73 ( 0.018   | ) 0.12 ( 0.0072 )  | 2.1 ( 0.033 )   |
| -0.5 | 0.4 | 1.74 ( 0.028 ) | 0.19 ( 0.016 )     | 1.7 ( 0.033 )   |
| -0.3 | 0.4 | 1.48 ( 0.062 ) | 0.29 ( 0.057 )     | 0.9 ( 0.017 )   |
| -0.1 | 0.4 | 1.11 ( 0.075   | ) 0.19 ( 0.15 )    | 0.25 ( 0.17 )   |
|  0   | 0.4 | 1.01 ( 0.016 ) | 0.11 ( 0.12 )      | 0.01 ( 0.048 )  |
|  0.1 | 0.4 | 1.04 ( 0.06 )  | 0.13 ( 0.13 )      | 0.08 ( 0.18 )   |
|  0.3 | 0.4 | 1.4 ( 0.08 )   | 0.28 ( 0.087 )     | 0.77 ( 0.015 )  |
|  0.5 | 0.4 | 1.71 ( 0.027 ) | 0.22 ( 0.015 )     | 1.52 ( 0.03 )   |

from this policy. The corresponding goal is to maximize

<!-- formula-not-decoded -->

where H is the differential entropy as before. Moreover, ˜ X π satisfies

<!-- formula-not-decoded -->

Following the same line of deductions as in Wang et al. (2020), we can show that the optimal policy is a normal distribution whose mean is linear in the state and whose variance is a constant. Therefore we parameterize the policy by π φ p¨| x q ' N p¨| φ 1 x ` φ 2 , e φ 3 q . Moreover, the function J is parameterized as a quadratic function J θ p x q ' 1 2 θ 0 x 2 ` θ 1 x (we ignore the constant term since J is unique up to a constant) and the optimal value V is an extra parameter.

This problem falls into the formulation of an ergodic task; so we directly implement Algorithm 3 in our simulation and then compare the learned parameters with the theoretically optimal ones. In addition, we compare the up-to-now average reward during the online learning process to two theoretical benchmarks. The first one is the omniscient optimal level, which is the maximum long term average reward that can be achieved by a hypothetical agent who knows completely about the environment (i.e. the correct model and model parameters) and acts optimally (the optimal policy is a deterministic one) without needing to explore (and hence there is no entropy regularization). The second benchmark is the omniscient optimal level less the exploration cost, which is the maximum long term average reward that can be achieved by the aforementioned hypothetical agent who is however forced to explore under entropy regularization. 10 Clearly, since exploration (rendering a stochastic policy) is inherent in the RL setting, our algorithm can at most achieve the second benchmark. In other words, after learning for a sufficiently long time, we can learn the correct optimal policy but can only expect the up-to-now average reward to approach the optimal level less the exploration cost.

To guarantee the stationarity of the controlled state process, we set A ' ´ 1 , B ' C ' 0 and D ' 1. Moreover, we set x 0 ' 0, M ' N ' Q ' 2, R ' P ' 1, and γ ' 0 . 1. Learning rate is initialized as α θ ' α φ ' 0 . 001, and decays according to l p t q ' 1 max t 1 , log t u . All the parameters to be learned are initialized as 0 and time discretization is taken as ∆ t ' 0 . 01. We repeat the experiment for 100 times.

We implement TD p 0 q for both the PE and the PG parts of the AC algorithm, referred to as the ActorCritic Policy Gradient algorithm in Figure 1. Namely, we choose test functions ξ t ' B J θ B θ p X t q , η t ' 1 , ζ t ' 0 in Algorithm 3. Figure 1 shows the convergence of the learned policy parameters along with that of the average reward along a single state sample trajectory. Observe that the average reward first decreases at the beginning of this particular trajectory. The reason may have been that during the initial iterations the

10 See Appendix B2 for precise definitions of these two benchmarks and detailed calculations of them.

0.5

-0.5

•, Path

True Value 41

2

92 Path

True Value 42

e°3 Path

True Value e°3

maliner

0

underlying state process has not yet converged to the stationary distribution and the initial policies are still far away from the optimal one, and hence the average reward is dominated by a few 'wrong trials'. After a sufficient amount of time, however, both the policies and the average reward start to converge to the theoretically optimal values. Between the two it takes a much longer time for the average reward to approach the optimal level as we wait for the contribution from the bad performance of the beginning period to diminish. Time × 105

<!-- image -->

- (a) The learned parameters in the policy along one sample trajectory. (b) The average reward along one sample trajectory.

Figure 1: Convergence of the learned policy and the average reward under the online learning algorithm. A single state trajectory is generated with length T ' 10 6 under the online AC algorithm. The left panel illustrates the convergence of the policy parameters, where the dashed horizontal lines indicate the values of the respective parameters of the theoretically optimal policy to the entropy-regularized exploratory stochastic control problem. The right panel shows the convergence of the average reward, where the two dashed horizontal lines are respectively the omniscient optimal average reward without exploration when the model parameters are known, and the omniscient optimal average reward less the exploration cost. We repeat the experiment for 100 times to calculate the standard deviations of the predicted parameters, which are represented as the shaded areas. The width of each shaded area is twice the corresponding standard deviation, which is very small compared to the scale of the vertical axis.

## 6 Conclusion

This paper is the final installment of a 'trilogy', the first two being Wang et al. (2020) and Jia and Zhou (2022a), that endeavors to develop a systematic and unified theoretical foundation for RL in continuous time with continuous state space and possibly continuous action space. The previous two papers address exploration and PE, respectively, and this paper focuses on PG. A major finding of the current paper is that PG is intimately related to PE, and thus the martingale characterization of PE established in Jia and Zhou (2022a) can be applied to PG. Combining the theoretical results of the three papers, we propose online and offline actor-critic algorithms for general model-free RL tasks, where we learn value functions and stochastic

policies simultaneously and alternatingly.

This series of papers are characterized by conducting all the theoretical analysis within the continuous setting and discretizing time only when implementing the algorithms. The advantages of this approach, versus discretizing time right at the start and then applying existing MDP results, are articulated in Doya (2000). Moreover, more analytical tools are at our disposal in the continuous setting, including calculus, stochastic calculus, stochastic control, and differential equations. The discrete-time versions of the various algorithms devised in the three papers are indeed well known in the discrete-time RL literature; hence their convergence is well established. On the other hand, Jia and Zhou (2022a) prove that any convergent timediscretized PE algorithm also converges as the mesh size goes to zero. Because PG algorithms developed in the current paper are essentially derived from the martingality for PE, the same convergence also holds for them.

It is interesting to note that the derivation and representation of PG are not entirely analogous to that of MDPs. For example, the latter involves a state-action function (Q-function), whereas the former is essentially the expected integration of a term involving the value function.

The study on continuous-time RL is still in its infancy, and open questions abound. These include, to name but a few, regret bound of episodic RL problems in terms of the number of episodes, interpretation of Q-function and Q-learning in the continuous setting, and dependence of the performance of AC algorithms on the temperature parameter when there is an exploration regularizer.

## Acknowledgement

Zhou gratefully acknowledges financial support through a start-up grant and the Nie Center for Intelligent Asset Management at Columbia University.

## Appendix A. Connections with Policy Gradient in Discrete Time

We review the classical policy gradient approach and results for discrete-time Markov decision processes (MDPs) here and compare them with their continuous-time counterparts developed in the main text.

For simplicity, we consider a time-homogeneous MDP X ' t X t , t ' 0 , 1 , 2 , ¨ ¨ ¨ u with a state space X , an action space A , and a transition matrix P p X 1 ' x 1 | X 0 ' x, a 0 ' a q ' p p x 1 | x, a q . Both X and A are finite sets. The expected reward is r p x, a q with a discount factor β P p 0 , 1 q . The agent's total expected reward is E 'ř 8 t ' 0 β t r p X t , a t q ‰ . A (stochastic) policy is denoted by π φ p¨| x q P P p A q , which is a probability density function on A , with a suitable parameter vector φ P R L φ .

Define the value function associated with a given policy π φ by

<!-- formula-not-decoded -->

We are interested in the gradient of the value function with respect to the policy parameter φ , that is, B J p x ; π φ q B φ . The classical policy gradient theorem (e.g., Sutton et al. 1999, Theorem 1) states that

<!-- formula-not-decoded -->

where Q p x, a ; π φ q ' r p x, a q` E r ř 8 t ' 1 β t r p X π φ t , a π φ t q ˇ ˇ ˇ X π φ 0 ' x s is the Q-function, and µ π φ p x 1 q ' ř 8 t ' 0 β t P p X π φ t ' x 1 | X π φ 0 ' x q is the (discounted) occupation time.

Define /lscript p x 1 q ' ř a P A B π φ B φ p a | x 1 q Q p x 1 , a ; π φ q , which is a deterministic function of x 1 . Since ř a P A B π φ B φ p a | x 1 q ' B B φ ř a P A π φ p a | x 1 q ' 0, /lscript p x 1 q can be equivalently written as

<!-- formula-not-decoded -->

for any function B p¨q , sometimes known as a baseline (Williams, 1992).

On the other hand,

<!-- formula-not-decoded -->

Therefore, (32) is equivalent to

<!-- formula-not-decoded -->

If we choose the baseline to be the value function B p¨q ' J p¨ ; π φ q , then (33) gives the representation of policy gradient in the advantage actor-critic approach (Mnih et al., 2016).

If we are to extend the above derivation to the continuous-time setting, then an essential question is what the Q-function should be in continuous time. This question has been extensively studied in a recent paper Jia and Zhou (2022b) from which we realize that (using the notations in this paper with the discount factor e ´ βt )

<!-- formula-not-decoded -->

Therefore, (33) becomes

<!-- formula-not-decoded -->

Note that there is no policy regularizer in the current discussion; so the above coincides with the expression of the policy gradient (14) with γ ' 0.

## Appendix B. Theoretical Results Employed in Simulation Experiments

For the reader's convenience, we summarize the theoretical results employed in the two simulation studies in Section 5. Their proofs are similar to those of the analogous results in Wang and Zhou (2020) and Wang et al. (2020) respectively.

## Appendix B1. Mean-Variance Portfolio Selection

Let the true model be given as (23) and one aims to solve (25). An omniscient agent's optimal policy is a deterministic one, given by a ˚ t ' ´ µ ´ r σ 2 p x t ´ w ˚ q , where w ˚ ' z exp t p µ ´ r q 2 σ 2 T u´ x 0 exp t p µ ´ r q 2 σ 2 T u´ 1 . Given this policy, the discounted wealth process (24) becomes

<!-- formula-not-decoded -->

Hence x ˚ t ´ w is a geometric Brownian motion. We can compute E r x ˚ T s ' z , and

<!-- formula-not-decoded -->

If the agent knows the true model but is forced to take a stochastic policy subject to the entropy regularizer, then the optimal policy is

<!-- formula-not-decoded -->

Under this policy, the dynamics of ˜ X π ˚ t is

<!-- formula-not-decoded -->

With the same value of w ˚ , one can show that E P W r ˜ X π ˚ T s ' z , and

<!-- formula-not-decoded -->

## Appendix B2. Ergodic Linear-Quadratic Control

Two benchmarks, 'omniscient optimal level' and 'omniscient optimal level less exploration cost', are used in Section 5.2 for comparison. We introduce their formal definitions here.

Definition 2. The omniscient optimal level is the maximum value of (28) subject to (27) when all the model

coefficients p A,B,C,D,M,R,N,P,Q q are known to the agent. The omniscient optimal level less exploration cost is defined as where π ˚ is the optimal policy to (29) with the entropy regularizer and subject to (30) when all the model coefficients are known to the agent.

<!-- formula-not-decoded -->

We compute these two values using the Hamilton-Jacobi-Bellman (HJB) equation approach.

Let the true model be given by (27) and one aims to maximize the long-term average reward (28). Consider the associated HJB equation:

<!-- formula-not-decoded -->

Conjecturing ϕ p x q ' 1 2 k 2 x 2 ` k 1 x and plugging it to the HJB equation, we get the first-order condition a ˚ ' r k 2 p B ` CD q´ R s x ` k 1 B ´ Q N ´ k 2 D 2 , assuming N ´ k 2 D 2 ą 0. The HJB equation now becomes

<!-- formula-not-decoded -->

This leads to three algebraic equations by matching the coefficients of x 2 , x and the constant term:

<!-- formula-not-decoded -->

Note that (34) coincides with the system of equations in footnote 12 and Theorem 9 in Wang et al. (2020) when the discount factor is 0.

Solving these algebraic equation gives the omniscient optimal reward and the corresponding optimal policy a ˚ ' r k 2 p B ` CD q´ R s x ` k 1 B ´ Q N ´ k 2 D 2 .

If the agent knows the true model but still adopts stochastic policies with a entropy regularizer, then the optimal policy is given by

<!-- formula-not-decoded -->

where k 2 , k 1 are determined by (34). This optimal solution is identical to that in Wang et al. (2020, Theorem 4) when the discount factor is 0.

Under this stochastic policy, the state dynamics become

<!-- formula-not-decoded -->

To calculate the long-term average value ˜ V ' lim inf T Ñ8 1 T E P W ' ş T 0 ş R r p ˜ X π ˚ t , a q π ˚ p a | ˜ X π ˚ t q d a d t  , consider the corresponding HJB equation

<!-- formula-not-decoded -->

Starting with an ansatz ˜ ϕ p x q ' 1 2 ˜ k 2 x 2 ` ˜ k 1 x and going through the same calculations as above we obtain three equations

<!-- formula-not-decoded -->

The solutions to the above equations are ˜ k 2 ' k 2 , ˜ k 1 ' k 1 . Hence

<!-- formula-not-decoded -->

By definition, ˜ V is also the omniscient optimal level less exploration cost. The difference, γ 2 , between ˜ V and

V is hence the exploration cost due to randomization. Note that a parallel result when there is a discount factor is Theorem 10 in Wang et al. (2020).

## Appendix C. Proofs of Statements

In all the proofs we use generic notations C 1 , C 2 , ¨ ¨ ¨ to denote constants that are independent of other variables involved such as t, x, a . A same such notation may show up in different places but does not necessarily have the same value.

## Proof of Lemma 1

We start by examining ˜ b ` t, x, π p¨| t, x q ˘ . Note that

<!-- formula-not-decoded -->

Hence

<!-- formula-not-decoded -->

Moreover, note that

<!-- formula-not-decoded -->

Similarly, we can show that ˜ σ ` t, x, π p¨| t, x q ˘ is locally Lipschitz continuous and has linear growth in x . The unique existence of the strong solution to (5) then follows from the standard SDE theory.

Next, the SDE (5) yields

<!-- formula-not-decoded -->

Based on the proved growth condition on ˜ b, ˜ σ , Cauchy-Schwarz inequality, and Burkholder-Davis-Gundy

inequalities, we obtain

<!-- formula-not-decoded -->

Applying Gronwall's inequality to E ' max t ď s ď T 1 | ˜ X π s | µ ˇ ˇ ˇ ˜ X π t ' x  as a function of T 1 , we obtain the second desired result of the lemma. The final result is evident.

## Proof of Lemma 2

Set ˜ v t, x e βt v t, x . Then ˜ v T, x e βT h x , and (8) implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, consider ˜ J p t, x ; π q ' e ´ βt J p t, x ; π q . Then (6) yields

<!-- formula-not-decoded -->

So it suffices to prove the (viscosity) solution to (35), ˜ v , coincides with ˜ J p¨ , ¨ ; π q . The proof now follows from applying Beck et al. (2021, Corollary 3.3) to the SDE (5): under Assumption 1 along with Definition 1, Lemma 1 verifies the sufficient conditions in Beck et al. (2021).

## Proof of Theorem 1

Using the same discounting transformation as in the proof of Lemma 2, the first statement of Theorem 1 follows directly from Jia and Zhou (2022a, Proposition 1) along with the Markov property of the solution to the SDE (5).

For the second statement, according to Jia and Zhou (2022a, Proposition 4), we have the following

martingale orthogonality condition for ˜ X π :

<!-- formula-not-decoded -->

for all ξ P L 2 F ˜ X π ` r 0 , T s ; J p¨ , ˜ X π ¨ ; π q ˘ . Now, any ξ P L 2 F ˜ X π ` r 0 , T s ; J p¨ , ˜ X π ¨ ; π q ˘ corresponds to a measurable functional ξ : r 0 , T s ˆ C pr 0 , T s ; R d q ÞÑ R such that ξ t ' ξ p t, ˜ X π t ^¨ q . However, ˜ X π and X π have the same distribution and a π t ' π p¨| t, ˜ X π t q ; hence and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining the above two equations leads to (9).

## Proof of Theorem 2

It suffices to prove (12) equals (14).

Fix t . Define a sequence of stopping times τ n ' inf t s ě t : | X π φ s | ě n u . Applying Itˆ o's lemma to J p s, X π φ s q , we obtain:

<!-- formula-not-decoded -->

Taking expectation yields

<!-- formula-not-decoded -->

The second term above vanishes because when t ď s ď T ^ τ n , it follows from Assumptions 1 that

<!-- formula-not-decoded -->

which is bounded by a function of n due to Assumption 3. Thus,

<!-- formula-not-decoded -->

Lastly, note

<!-- formula-not-decoded -->

By Assumption 3 and Lemma 1, we get

<!-- formula-not-decoded -->

Hence by the dominance convergence theorem, we conclude that as n Ñ8 ,

<!-- formula-not-decoded -->

This proves the desired result.

## Proof of Theorem 3

The proof is similar to the proof of Theorem 1 and that in Jia and Zhou (2022a, Proposition 4) by noticing π ˚ satisfies (17).

First, it suffices to consider the case when ζ ' 0 because of Theorem 1. For η P L 2 F X π φ ˚ ` r 0 , T s , J p¨ , X π φ ˚ ¨ ; π φ ˚ q ˘ , we write η s ' η p s, X π φ ˚ s ^¨ q . Then the right hand side of (18) can be written as

<!-- formula-not-decoded -->

where the last equality follows from (17).

## Proof of Lemma 3

Apply Itˆ o's lemma to J p ˜ X π s ; π q on s P r t, T s to obtain

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

By a similar localization argument as in the proof of Theorem 2, we can show that lim sup T Ñ8 E P W ' J p ˜ X π T ; π q ˇ ˇ ˇ ˜ X π t ' x  is finite and independent of x . Taking limit T Ñ8 on both sides of the above yields (20). The above analysis also implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is an p F ˜ X π , P W q -martingale. For the same reason as in the proof of Theorem 1, we arrive at the second desired conclusion of the lemma.

## References

- Aleksandrov, V., Sysoev, V., and Shemeneva, V. (1968). Stochastic optimization. Engineering Cybernetics , 5:11-16.
- Aragon-G´ omez, R. and Clempner, J. B. (2020). Traffic-signal control reinforcement learning approach for continuous-time Markov games. Engineering Applications of Artificial Intelligence , 89:103415.
- Arapostathis, A., Borkar, V. S., and Ghosh, M. K. (2012). Ergodic control of diffusion processes , volume 143. Cambridge University Press.
- Baird, L. C. (1993). Advantage updating. Technical report, Write Lab Wright-Patterson Air Force Base, OH 45433-7301, USA.

- Barto, A. G., Sutton, R. S., and Anderson, C. W. (1983). Neuronlike adaptive elements that can solve difficult learning control problems. IEEE Transactions on Systems, Man, and Cybernetics , (5):834-846.
- Basak, S. and Chabakauri, G. (2010). Dynamic mean-variance asset allocation. The Review of Financial Studies , 23(8):2970-3016.
- Basei, M., Guo, X., Hu, A., and Zhang, Y. (2020). Logarithmic regret for episodic continuous-time linearquadratic reinforcement learning over a finite-time horizon. arXiv preprint arXiv:2006.15316 .
- Beck, C., Hutzenthaler, M., and Jentzen, A. (2021). On nonlinear Feynman-Kac formulas for viscosity solutions of semilinear parabolic partial differential equations. Stochastics and Dynamics , page 2150048.
- Bensoussan, A. and Frehse, J. (1992). On Bellman equations of ergodic control in R n . In Applied Stochastic Analysis , pages 21-29. Springer.
- Bhatnagar, S., Sutton, R. S., Ghavamzadeh, M., and Lee, M. (2009). Natural actor-critic algorithms. Automatica , 45(11):2471-2482.
- Bj¨ ork, T., Murgoci, A., and Zhou, X. Y. (2014). Mean-variance portfolio optimization with state-dependent risk aversion. Mathematical Finance , 24(1):1-24.
- Borkar, V. S. and Ghosh, M. K. (1988). Ergodic control of multidimensional diffusions I: The existence results. SIAM Journal on Control and Optimization , 26(1):112-126.
- Borkar, V. S. and Ghosh, M. K. (1990). Ergodic control of multidimensional diffusions II: Adaptive control. Applied Mathematics and Optimization , 21(1):191-220.
- Bradtke, S. J. and Barto, A. G. (1996). Linear least-squares algorithms for temporal difference learning. Machine Learning , 22(1):33-57.
- Dai, M., Dong, Y., and Jia, Y. (2020). Learning equilibrium mean-variance strategy. SSRN preprint SSRN:3770818 .
- Dai, M., Jin, H., Kou, S., and Xu, Y. (2021). A dynamic mean-variance analysis for log returns. Management Science , 67(2):1093-1108.
- Degris, T., Pilarski, P. M., and Sutton, R. S. (2012). Model-free reinforcement learning with continuous action in practice. In 2012 American Control Conference (ACC) , pages 2177-2182. IEEE.
- Doya, K. (2000). Reinforcement learning in continuous time and space. Neural Computation , 12(1):219-245.
- Duan, Y., Chen, X., Houthooft, R., Schulman, J., and Abbeel, P. (2016). Benchmarking deep reinforcement learning for continuous control. In International Conference on Machine Learning , pages 1329-1338. PMLR.

- Ekeland, I. and Lazrak, A. (2006). Being serious about non-commitment: subgame perfect equilibrium in continuous time. arXiv preprint math/0604264 .
- Fedus, W., Ramachandran, P., Agarwal, R., Bengio, Y., Larochelle, H., Rowland, M., and Dabney, W. (2020). Revisiting fundamentals of experience replay. In International Conference on Machine Learning , pages 3061-3071. PMLR.
- Fleming, W. H. and Soner, H. M. (2006). Controlled Markov Processes and Viscosity Solutions , volume 25. Springer Science &amp; Business Media.
- Fr´ emaux, N., Sprekeler, H., and Gerstner, W. (2013). Reinforcement learning using a continuous time actor-critic framework with spiking neurons. PLoS Computational Biology , 9(4):e1003024.
- Gao, X., Xu, Z. Q., and Zhou, X. Y. (2022). State-dependent temperature control for Langevin diffusions. SIAM Journal on Control and Optimization , 60(3):1250-1268.
- Glynn, P. W. (1990). Likelihood ratio gradient estimation for stochastic systems. Communications of the ACM , 33(10):75-84.
- Guo, X., Xu, R., and Zariphopoulou, T. (2022). Entropy regularization for mean field games with learning. Mathematics of Operations Research .
- Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., Kumar, V., Zhu, H., Gupta, A., Abbeel, P., et al. (2018). Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905 .
- Jia, Y. and Zhou, X. Y. (2022a). Policy evaluation and temporal-difference learning in continuous time and space: A martingale approach. Journal of Machine Learning Research , 23(154):1-55.
- Jia, Y. and Zhou, X. Y. (2022b). q-Learning in continuous time. arXiv preprint; http://arxiv.org/abs/2207.00713 .
- Karatzas, I. and Shreve, S. (2014). Brownian motion and stochastic calculus , volume 113. Springer.
- Kim, J., Shin, J., and Yang, I. (2021). Hamilton-Jacobi deep Q-Learning for deterministic continuous-time systems with Lipschitz continuous controls. Journal of Machaine Learning Research , 22:206-1.
- Kiran, B. R., Sobh, I., Talpaert, V., Mannion, P., Al Sallab, A. A., Yogamani, S., and P´ erez, P. (2021). Deep reinforcement learning for autonomous driving: A survey. IEEE Transactions on Intelligent Transportation Systems .
- Lee, J. and Sutton, R. S. (2021). Policy iterations for reinforcement learning problems in continuous time and space-Fundamental theory and methods. Automatica , 126:109421.

- Li, D. and Ng, W.-L. (2000). Optimal dynamic portfolio selection: Multiperiod mean-variance formulation. Mathematical Finance , 10(3):387-406.
- Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., and Wierstra, D. (2015). Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971 .
- Maei, H. R., Szepesvari, C., Bhatnagar, S., Precup, D., Silver, D., and Sutton, R. S. (2009). Convergent temporal-difference learning with arbitrary smooth function approximation. In NIPS , pages 1204-1212.
- Marbach, P. and Tsitsiklis, J. N. (2001). Simulation-based optimization of Markov reward processes. IEEE Transactions on Automatic Control , 46(2):191-209.
- Meyn, S. P. and Tweedie, R. L. (2012). Markov Chains and Stochastic Stability . Springer Science &amp; Business Media.
- Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., and Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning , pages 1928-1937. PMLR.
- Munos, R. (2006). Policy gradient in continuous time. Journal of Machine Learning Research , 7:771-791.
- Munos, R. and Bourgine, P. (1997). Reinforcement learning for continuous stochastic control problems. Advances in Neural Information Processing Systems , 10.
- Sandri´ c, N. (2017). A note on the Birkhoff ergodic theorem. Results in Mathematics , 72(1):715-730.
- Schulman, J., Chen, X., and Abbeel, P. (2017a). Equivalence between policy gradients and soft Q-learning. arXiv preprint arXiv:1704.06440 .
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017b). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 .
- Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., and Riedmiller, M. (2014). Deterministic policy gradient algorithms. In International Conference on Machine Learning , pages 387-395. PMLR.
- Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., et al. (2017). Mastering the game of go without human knowledge. Nature , 550(7676):354359.
- Strotz, R. H. (1955). Myopia and inconsistency in dynamic utility maximization. The Review of Economic Studies , 23(3):165-180.
- Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. Machine Learning , 3(1):9-44.

- Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An Introduction . Cambridge, MA: MIT Press.
- Sutton, R. S., Maei, H. R., Precup, D., Bhatnagar, S., Silver, D., Szepesv´ ari, C., and Wiewiora, E. (2009). Fast gradient-descent methods for temporal-difference learning with linear function approximation. In Proceedings of the 26th Annual International Conference on Machine Learning , pages 993-1000.
- Sutton, R. S., McAllester, D., Singh, S., and Mansour, Y. (1999). Policy gradient methods for reinforcement learning with function approximation. Advances in Neural Information Processing Systems , 12.
- Sutton, R. S., Singh, S., and McAllester, D. (2000). Comparing policy-gradient algorithms. IEEE Transactions on Systems, Man, and Cybernetics .
- Sutton, R. S., Szepesv´ ari, C., and Maei, H. R. (2008). A convergent o(n) temporal-difference algorithm for off-policy learning with linear function approximation. In NIPS .
- Szpruch, L., Treetanthiploet, T., and Zhang, Y. (2021). Exploration-exploitation trade-off for continuoustime episodic reinforcement learning with linear-convex models. arXiv preprint arXiv:2112.10264 .
- Tallec, C., Blier, L., and Ollivier, Y. (2019). Making deep Q-learning methods robust to time discretization. In International Conference on Machine Learning , pages 6096-6104. PMLR.
- Tang, W., Zhang, P. Y., and Zhou, X. Y. (2021). Exploratory HJB equations and their convergence. arXiv preprint arXiv:2109.10269 .
- Vamvoudakis, K. G. and Lewis, F. L. (2010). Online actor-critic algorithm to solve the continuous-time infinite horizon optimal control problem. Automatica , 46(5):878-888.
- Wang, H., Zariphopoulou, T., and Zhou, X. Y. (2020). Reinforcement learning in continuous time and space: A stochastic control approach. Journal of Machine Learning Research , 21(198):1-34.
- Wang, H. and Zhou, X. Y. (2020). Continuous-time mean-variance portfolio selection: A reinforcement learning framework. Mathematical Finance , 30(4):1273-1308.
- Wang, W., Han, J., Yang, Z., and Wang, Z. (2021). Global convergence of policy gradient for linearquadratic mean-field control/game in continuous time. In International Conference on Machine Learning , pages 10772-10782. PMLR.
- Wawrzynski, P. (2007). Learning to control a 6-degree-of-freedom walking robot. In EUROCON 2007-The International Conference on' Computer as a Tool' , pages 698-705. IEEE.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning , 8(3):229-256.

- Yildiz, C., Heinonen, M., and L¨ ahdesm¨ aki, H. (2021). Continuous-time model-based reinforcement learning. In International Conference on Machine Learning , pages 12009-12018. PMLR.
- Yong, J. and Zhou, X. Y. (1999). Stochastic Controls: Hamiltonian Systems and HJB Equations . New York, NY: Spinger.
- Zambrano, D., Roelfsema, P. R., and Bohte, S. M. (2015). Continuous-time on-policy neural reinforcement learning of working memory tasks. In 2015 International Joint Conference on Neural Networks (IJCNN) , pages 1-8. IEEE.
- Zhang, S. and Sutton, R. S. (2017). A deeper look at experience replay. arXiv preprint arXiv:1712.01275 .
- Zhao, T., Hachiya, H., Niu, G., and Sugiyama, M. (2011). Analysis and improvement of policy gradient estimation. In NIPS , pages 262-270. Citeseer.
- Zhou, X. Y. and Li, D. (2000). Continuous-time mean-variance portfolio selection: A stochastic LQ framework. Applied Mathematics and Optimization , 42(1):19-33.