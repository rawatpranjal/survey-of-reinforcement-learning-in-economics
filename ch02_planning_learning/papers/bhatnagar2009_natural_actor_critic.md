## Minimum Probability Flow Learning

Jascha Sohl-Dickstein ab ∗ Peter Battaglino ac ∗ Michael R. DeWeese acd jascha@berkeley.edu pbb@berkeley.edu deweese@berkeley.edu

a Redwood Center for Theoretical Neuroscience, b Biophysics Graduate Group, c Physics Department, d Helen Wills Neuroscience Institute, University of California, Berkeley, 94720 ∗ These authors contributed equally.

## Abstract

Fitting probabilistic models to data is often difficult, due to the general intractability of the partition function and its derivatives. Here we propose a new parameter estimation technique that does not require computing an intractable normalization factor or sampling from the equilibrium distribution of the model. This is achieved by establishing dynamics that would transform the observed data distribution into the model distribution, and then setting as the objective the minimization of the KL divergence between the data distribution and the distribution produced by running the dynamics for an infinitesimal time. Score matching, minimum velocity learning, and certain forms of contrastive divergence are shown to be special cases of this learning technique. We demonstrate parameter estimation in Ising models, deep belief networks and an independent component analysis model of natural scenes. In the Ising model case, current state of the art techniques are outperformed by at least an order of magnitude in learning time, with lower error in recovered coupling parameters.

## 1. Introduction

Estimating parameters for probabilistic models is a fundamental problem in many scientific and engineering disciplines. Unfortunately, most probabilistic learning techniques require calculating the normalization factor, or partition function, of the probabilistic model in question, or at least calculating its gradient. For the overwhelming majority of models there

Appearing in Proceedings of the 28 th International Conference on Machine Learning , Bellevue, WA, USA, 2011. Copyright 2011 by the author(s)/owner(s).

are no known analytic solutions. Thus, development of powerful new techniques for parameter estimation promises to greatly expand the variety of models that can be fit to complex data sets.

Many approaches exist for approximate learning, including mean field theory and its expansions, variational Bayes techniques and a variety of sampling or numerical integration based methods (Tanaka, 1998; Kappen &amp; Rodr´ ıguez, 1997; Jaakkola &amp; Jordan, 1997; Haykin, 2008). Of particular interest are contrastive divergence (CD), developed by Hinton, Welling and Carreira-Perpi˜ n´ an (Welling &amp; Hinton, 2002; CarreiraPerpi˜ n´ an &amp; Hinton, 2004), Hyv¨ arinen's score matching (SM) (Hyv¨ arinen, 2005), Besag's pseudolikelihood (PL) (Besag, 1975), and the minimum velocity learning framework proposed by Movellan (Movellan, 2008a;b; Movellan &amp; McClelland, 1993).

Contrastive divergence (Welling &amp; Hinton, 2002; Carreira-Perpi˜ n´ an &amp; Hinton, 2004) is a variation on steepest gradient descent of the maximum (log) likelihood (ML) objective function. Rather than integrating over the full model distribution, CD approximates the partition function term in the gradient by averaging over the distribution obtained after taking a few, or only one, Markov chain Monte Carlo (MCMC) steps away from the data distribution (Equation 17). Qualitatively, one can imagine that the data distribution is contrasted against a distribution that has evolved only a small distance towards the model distribution, whereas it would be contrasted against the true model distribution in traditional MCMC approaches. Although CD is not guaranteed to converge to the right answer, or even to a fixed point, it has proven to be an effective and fast heuristic for parameter estimation (MacKay, 2001; Yuille, 2005).

Score matching (Hyv¨ arinen, 2005) is a method that learns parameters in a probabilistic model using only derivatives of the energy function evaluated over the data distribution (see Equation (19)). This sidesteps

GLYPH&lt;c=18,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=20,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=9,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=20,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=2,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=7,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=2,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=5,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=21,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=6,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=15,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=2,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=4,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=5,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=3,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=2,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=20,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=22,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=5,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=2,font=/IBSUNV+font00000000134be52c&gt; GLYPH&lt;c=18,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=10,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=6,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=4,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=12,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=9,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=17,font=/IBSUNV+font00000000134be52c&gt;GLYPH&lt;c=8,font=/IBSUNV+font00000000134be52c&gt;

GLYPH&lt;c=19,font=/IBSUNV+font00000000134be52c&gt;

GLYPH&lt;c=3,font=/IBSUNV+font00000000134be52c&gt;

<!-- image -->

## Progression of Learning

Figure 1. An illustration of parameter estimation using minimum probability flow (MPF). In each panel, the axes represent the space of all probability distributions. The three successive panels illustrate the sequence of parameter updates that occur during learning. The dashed red curves indicate the family of model distributions p ( ∞ ) ( θ ) parametrized by θ . The black curves indicate deterministic dynamics that transform the data distribution p (0) into the model distribution p ( ∞ ) ( θ ). Under maximum likelihood learning, model parameters θ are chosen so as to minimize the Kullback-Leibler (KL) divergence between the data distribution p (0) and the model distribution p ( ∞ ) ( θ ). Under MPF, however, the KL divergence between p (0) and p ( glyph[epsilon1] ) is minimized instead, where p ( glyph[epsilon1] ) is the distribution obtained by initializing the dynamics at the data distribution p (0) and then evolving them for an infinitesimal time glyph[epsilon1] . Here we represent graphically how parameter updates that pull p ( glyph[epsilon1] ) towards p (0) also tend to pull p ( ∞ ) ( θ ) towards p (0) .

the need to explicitly sample or integrate over the model distribution. In score matching one minimizes the expected square distance of the score function with respect to spatial coordinates given by the data distribution from the similar score function given by the model distribution. A number of connections have been made between score matching and other learning techniques (Hyv¨ arinen, 2007; Sohl-Dickstein &amp; Olshausen, 2009; Movellan, 2008a; Lyu, 2009).

Pseudolikelihood (Besag, 1975) approximates the joint probability distribution of a collection of random variables with a computationally tractable product of conditional distributions, where each factor is the distribution of a single random variable conditioned on the others. This approach often leads to surprisingly good parameter estimates, despite the extreme nature of the approximation.

Minimum velocity learning is an approach recently proposed by Movellan (Movellan, 2008a) that recasts a number of the ideas behind CD, treating the minimization of the initial dynamics away from the data distribution as the goal itself rather than a surrogate for it. Rather than directly minimize the difference between the data and the model, Movellan's proposal is to introduce system dynamics that have the model as their equilibrium distribution, and minimize the initial flow of probability away from the data under those dynamics. If the model looks exactly like the data there will be no flow of probability, and if model and data are similar the flow of probability will tend to be minimal. Movellan applies this intuition to the specific case of distributions over continuous state spaces evolving via diffusion dynamics, and recovers the score matching objective function.

Two additional recent techniques deserve mention. Minimum KL contraction (Lyu, 2011) involves applying a contraction mapping to both data and model distributions, and minimizing the amount by which this contraction mapping shrinks the KL divergence between data and model distributions. Like minimum probability flow, it appears to be a generalization of a number of existing parameter estimation techniques based on 'local' information about the model distribution. Noise contrastive estimation (Gutmann &amp; Hyv¨ arinen, 2010) estimates model parameters and the partition function by training a classifier to distinguish between the data distribution and a noise distribution carefully chosen to resemble the data distribution.

Here we propose a consistent parameter estimation framework called minimum probability flow learning (MPF), applicable to any parametric model without latent variables. Minimum velocity learning, SM and certain forms of CD are all special cases of MPF, which is in many situations more powerful than any of these other algorithms. We demonstrate that learning under this framework is effective and fast in a number of cases: Ising models (Brush, 1967; Ackley et al., 1985), deep belief networks (Hinton et al., 2006), and independent component analysis (Bell AJ, 1995).

## 2. Minimum Probability Flow

Our goal is to find the parameters that cause a probabilistic model to best agree with a list D of (assumed iid) observations of the state of a system. We will do this by introducing deterministic dynamics that guarantee the transformation of the data distribution into the model distribution, and then minimizing the KL divergence between the data distribution and the distribution that results from running those dynamics for a short time glyph[epsilon1] (see Figure 1).

## 2.1. Distributions

The data distribution is represented by a vector p (0) , with p (0) i the fraction of the observations D in state i . The superscript (0) represents time t = 0 under the system dynamics (which will be described in more detail in Section 2.2). For example, in a two variable

Figure 2. Dynamics of minimum probability flow learning. Model dynamics represented by the probability flow matrix Γ ( middle ) determine how probability flows from the empirical histogram of the sample data points ( left ) to the equilibrium distribution of the model ( right ) after a sufficiently long time. In this example there are only four possible states for the system, which consists of a pair of binary variables, and the particular model parameters favor state 10 whereas the data falls on other states.

<!-- image -->

binary system, p (0) would have four entries representing the fraction of the data in states 00, 01, 10 and 11 (Figure 2).

Our goal is to find the parameters θ that cause a model distribution p ( ∞ ) ( θ ) to best match the data distribution p (0) . The superscript ( ∞ ) on the model distribution indicates that this is the equilibrium distribution reached after running the dynamics for infinite time. Without loss of generality, we assume the model distribution is of the form

<!-- formula-not-decoded -->

where E ( θ ) is referred to as the energy function, and the normalizing factor Z ( θ ) is the partition function,

<!-- formula-not-decoded -->

(this can be thought of as a Boltzmann distribution of a physical system with k B T set to 1).

## 2.2. Dynamics

Most Monte-Carlo algorithms rely on two core concepts from statistical physics, the first being conservation of probability as enforced by the master equation for the time evolution of a distribution p ( t ) (Pathria, 1972):

glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

where ˙ p ( t ) i is the time derivative of p ( t ) i . Transition rates Γ ij ( θ ), for i = j , give the rate at which probability flows from a state j into a state i . The first term of Equation (3) captures the flow of probability out of other states j into the state i , and the second captures flow out of i into other states j . The dependence on θ results from the requirement that the chosen dynamics cause p ( t ) to flow to the equilibrium distribution p ( ∞ ) ( θ ). For readability, explicit dependence on θ will be dropped except where necessary. If we choose to set the diagonal elements of Γ to obey Γ ii = -∑ j = i Γ ji , then we can write the dynamics as

(see Figure 2). The unique solution for p ( t ) is given by 1

<!-- formula-not-decoded -->

where exp ( Γ t ) is a matrix exponential.

## 2.3. Detailed Balance

The second core concept is detailed balance,

<!-- formula-not-decoded -->

which states that at equilibrium the probability flow from state i into state j equals the probability flow from j into i . When satisfied, detailed balance guarantees that the distribution p ( ∞ ) ( θ ) is a fixed point of the dynamics. Sampling in most Monte Carlo methods is performed by choosing Γ consistent with Equation 6 (and the added requirement of ergodicity), then stochastically running the dynamics of Equation 3. Note that there is no need to restrict the dynamics defined by Γ to those of any real physical process, such as diffusion.

Equation 6 can be written in terms of the model's energy function E ( θ ) by substituting in Equation 1 for p ( ∞ ) ( θ ):

<!-- formula-not-decoded -->

Γ is underconstrained by the above equation. Introducing the additional constraint that Γ be invariant to the addition of a constant to the energy function (as the model distribution p ( ∞ ) ( θ ) is), we choose the following form for the non-diagonal entries in Γ

glyph[negationslash]

<!-- formula-not-decoded -->

1 The form chosen for Γ in Equation (4), coupled with the satisfaction of detailed balance and ergodicity introduced in section 2.3, guarantees that there is a unique eigenvector p ( ∞ ) of Γ with eigenvalue zero, and that all other eigenvalues of Γ have negative real parts.

where the connectivity function

<!-- formula-not-decoded -->

glyph[negationslash]

determines which states are allowed to directly exchange probability with each other 2 . g ij can be set such that Γ is extremely sparse (see Section 2.5). Theoretically, to guarantee convergence to the model distribution, the non-zero elements of Γ must be chosen such that, given sufficient time, probability can flow between any pair of states (ergodicity).

## 2.4. Objective Function

Maximum likelihood parameter estimation involves maximizing the likelihood of some observations D under a model, or equivalently minimizing the KL divergence between the data distribution p (0) and model distribution p ( ∞ ) ,

<!-- formula-not-decoded -->

Rather than running the dynamics for infinite time, we propose to minimize the KL divergence after running the dynamics for an infinitesimal time glyph[epsilon1] ,

<!-- formula-not-decoded -->

For small glyph[epsilon1] , D KL ( p ( 0 ) || p ( glyph[epsilon1] ) ( θ ) ) can be approximated by a first order Taylor expansion,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Further algebra (see Appendix A) reduces K ( θ ) to a measure of the flow of probability, at time t = 0 under the dynamics, out of data states j ∈ D into non-data states i / ∈ D ,

<!-- formula-not-decoded -->

2 The non-zero Γ may also be sampled from a proposal distribution rather than set via a deterministic scheme, in which case g ij takes on the role of proposal distribution see Appendix D.

with gradient

<!-- formula-not-decoded -->

where |D| is the number of observed data points. Note that Equations (14) and (16) do not depend on the partition function Z ( θ ) or its derivatives.

K ( θ ) is uniquely zero when p (0) and p ( ∞ ) ( θ ) are equal. This implies consistency, in that if the data comes from the model class, in the limit of infinite data K ( θ ) will be minimized by exactly the right θ . In addition, K ( θ ) is convex for all models p ( ∞ ) ( θ ) in the exponential family - that is, models whose energy functions E ( θ ) are linear in their parameters θ (Macke &amp; Gerwinn, 2009) (see Appendix B).

## 2.5. Tractability

The dimensionality of the vector p (0) is typically huge, as is that of Γ ( e.g. , 2 d and 2 d × 2 d , respectively, for a d -bit binary system). Na¨ ıvely, this would seem to prohibit evaluation and minimization of the objective function. Fortunately, we need only visit those columns of Γ ij corresponding to data states, j ∈ D . Additionally, g ij can be populated so as to connect each state j to only a small fixed number of additional states i . The cost in both memory and time to evaluate the objective function is thus O ( |D| ), and does not depend on the number of system states, only on the (much smaller) number of observed data points.

## 2.6. Continuous State Spaces

Although we have motivated this technique using systems with a large, but finite, number of states, it generalizes to continuous state spaces. Γ ji , g ji , and p ( t ) i become continuous functions Γ ( x j , x i ), g ( x j , x i ), and p ( t ) ( x i ). Γ( x j , x i ) can be populated stochastically and extremely sparsely (see Appendix D), preserving the O ( |D| ) cost. A specific scheme (similar to CD with Hamiltonian Monte Carlo) for estimating parameters in a continuous state space via MPF is described in Appendix E.

## 2.7. Choosing the Connectivity Function g

Qualitatively, the most informative states to connect data states to are those that are most probable under the model. In discrete state spaces, nearest neighbor connectivity schemes for g ji work extremely well (eg Equation 21 below). This is because, as learning converges, the states that are near data states become the

states that are probable under the model.

In continuous state spaces, the estimated parameters are much more sensitive to the choice of g ( x j , x i ). One effective form for g ( x j , x i ) is described in Appendix E, but theory supporting different choices of g ( x j , x i ) remains an area of active exploration.

## 3. Connection to Other Learning Techniques

## 3.1. Contrastive Divergence

The contrastive divergence update rule can be written in the form

<!-- formula-not-decoded -->

where T ij is the probability of transitioning from state j to state i in a single Markov chain Monte Carlo step (or k steps for CDk ). Equation 17 has obvious similarities to the MPF learning gradient in Equation 16. Thus, steepest gradient descent under MPF resembles CD updates, but with the MCMC sampling/rejection step T ij replaced by a weighting factor g ij exp [ 1 2 ( E j ( θ ) -E i ( θ )) ] .

Note that this difference in form provides MPF with a well-defined objective function. One important consequence of the existence of an objective function is that MPF can readily utilize general purpose, off-theshelf optimization packages for gradient descent, which would have to be tailored in some way to be applied to CD. This is part of what accounts for the dramatic difference in learning time between CD and MPF in some cases (see Fig. 3).

## 3.2. Score Matching

For a continuous state space, MPF reduces to score matching if the connectivity function g ( x j , x i ) is set to connect all states within a small distance r of each other,

<!-- formula-not-decoded -->

where d ( x i , x j ) is the Euclidean distance between states x i and x j . In the limit as r goes to 0 (within an overall constant and scaling factor),

<!-- formula-not-decoded -->

where K SM ( θ ) is the SM objective function (see Appendix C). Unlike SM, MPF is applicable to any parametric model, including discrete systems, and it does not require evaluating a third order derivative, which can result in unwieldy expressions.

## 4. Experimental Results

Matlab code implementing MPF for several cases is available at https://github.com/Sohl-Dickstein/ Minimum-Probability-Flow-Learning .

All minimization was performed using minFunc (Schmidt, 2005).

## 4.1. Ising Model

The Ising model has a long and storied history in physics (Brush, 1967) and machine learning (Ackley et al., 1985) and it has recently been found to be a surprisingly useful model for networks of neurons in the retina (Schneidman et al., 2006; Shlens et al., 2006).

We estimated parameters for an Ising model (sometimes referred to as a fully visible Boltzmann machine or an Ising spin glass) of the form

<!-- formula-not-decoded -->

where the coupling matrix J only had non-zero elements corresponding to nearest-neighbor units in a two-dimensional square lattice, and bias terms along the diagonal. The training data D consisted of 20 , 000 d -element iid binary samples x ∈ { 0 , 1 } d generated via Swendsen-Wang sampling (Swendsen &amp; Wang, 1987) from a spin glass with known coupling parameters. We used a square 10 × 10 lattice, d = 10 2 . The nondiagonal nearest-neighbor elements of J were set using draws from a normal distribution with variance σ 2 = 10. The diagonal (bias) elements of J were set in such a way that each column of J summed to 0, so that the expected unit activations were 0 . 5. The transition matrix Γ had 2 d × 2 d elements, but for learning we populated it sparsely, setting

<!-- formula-not-decoded -->

Figure 3 shows the mean square error in the estimated J and the mean square error in the corresponding pairwise correlations as a function of learning time for MPF and four competing approaches: mean field theory with TAP corrections (Tanaka, 1998), CD with both one and ten sampling steps per iteration, and

.

Figure 3. A demonstration of Minimum Probability Flow (MPF) outperforming existing techniques for parameter recovery in an Ising model. (a) Time evolution of the mean square error in the coupling strengths for 5 methods for the first 60 seconds of learning. Note that mean field theory with second order corrections (MFT+TAP) actually increases the error above random parameter assignments in this case. (b) Mean square error in the coupling strengths for the first 800 seconds of learning. (c) Mean square error in coupling strengths for the entire learning period. (d) -(f) Mean square error in pairwise correlations for the first 60 seconds of learning, the first 800 seconds of learning, and the entire learning period, respectively. In every comparison above MPF finds a better fit, and for all cases but MFT+TAP does so in a shorter time (see Table 1).

<!-- image -->

pseudolikelihood. Using MPF, learning took approximately 60 seconds, compared to roughly 800 seconds for pseudolikelihood and upwards of 20 , 000 seconds for 1-step and 10-step CD. Note that given sufficient training samples, MPF would converge exactly to the right answer, as learning in the Ising model is convex (see Appendix B), and has its global minimum at the true solution. Table 1 shows the relative performance at convergence in terms of mean square error in recovered weights, mean square error in the resulting model's correlation function, and convergence time. MPF was dramatically faster to converge than any of the other models tested, with the exception of MFT+TAP, which failed to find reasonable parameters. MPF fit the model to the data substantially better than any of the other models.

Table 1. Mean square error in recovered coupling strengths ( glyph[epsilon1] J ), mean square error in pairwise correlations ( glyph[epsilon1] corr ) and learning time for MPF versus mean field theory with TAP correction (MFT+TAP), 1-step and 10-step contrastive divergence (CD-1 and CD-10), and pseudolikelihood (PL).

| Technique   |   glyph[epsilon1] J |   glyph[epsilon1] corr | Time (s)   |
|-------------|---------------------|------------------------|------------|
| MPF         |              0.0172 |                 0.0025 | ∼ 60       |
| MFT+TAP     |              7.7704 |                 0.0983 | 0.1        |
| CD-1        |              0.3196 |                 0.0127 | ∼ 20000    |
| CD-10       |              0.3341 |                 0.0123 | ∼ 20000    |
| PL          |              0.0582 |                 0.0036 | ∼ 800      |

## 4.2. Deep Belief Network

As a demonstration of learning on a more complex discrete valued model, we trained a 4 layer deep belief network (DBN) (Hinton et al., 2006) on MNIST handwritten digits. A DBN consists of stacked restricted Boltzmann machines (RBMs), such that the hidden layer of one RBM forms the visible layer of the next. Each RBM has the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Sampling-free application of MPF requires analytically marginalizing over the hidden units. RBMs were trained in sequence, starting at the bottom layer, on 10,000 samples from the MNIST postal hand written digits data set. As in the Ising case, the transition matrix Γ was populated so as to connect every state to all states that differed by only a single bit flip (Equation 21). Training was performed by both MPF and single step CD (note that CD turns into full ML learning as the number of steps is increased, and that many step CD would have produced a superior, more computationally expensive, answer).

Confabulations were generated by Gibbs sampling from the top layer RBM, then propagating each sample back down to the pixel layer by way of the conditional distribution p ( ∞ ) ( x vis | x hid ; W k ) for each of the intermediary RBMs, where k indexes the layer in the stack. 1 , 000 sampling steps were taken between each confabulation. As shown in Figure 4, MPF learned a good model of handwritten digits.

## 4.3. Independent Component Analysis

As a demonstration of parameter estimation in continuous state space probabilistic models, we trained the receptive fields J ∈ R K × K of a K dimensional in-

/

/

6

7

6

8

200 units

/

200 units

8

200 units

7

7

9

5

6

200 units

7

*3

3

28x28 pixels

7

7

/

/

-

/

/

/

9

9

6

7

7

7

7

7

9

/

9

4

9

Figure 4. A deep belief network trained using minimum probability flow learning (MPF). (a) A four layer deep belief network was trained on the MNIST postal hand written digits dataset by MPF and single step contrastive divergence (CD). (b) Confabulations after training via MPF. A reasonable probabilistic model for handwritten digits has been learned. (c) Confabulations after training via CD. The uneven distribution of digit occurrences suggests that CD-1 has learned a less representative model than MPF.

<!-- image -->

Figure 5. A continuous state space model fit using minimum probability flow learning (MPF). Learned 10 × 10 pixel independent component analysis receptive fields J trained on natural image patches via (a) MPF and (b) maximum likelihood learning (ML). The average log likelihood of the model found by MPF ( -120 . 61 nats) was nearly identical to that found by ML ( -120 . 33 nats), consistent with the visual similarity of the receptive fields.

<!-- image -->

7

/

6

-

/

5

/

7

7

7

7

7

/

7

7

8

dependent component analysis (ICA) (Bell AJ, 1995) model with a Laplace prior,

<!-- formula-not-decoded -->

on 100 , 000 10 × 10 whitened natural image patches from the van Hateren database (Hateren &amp; Schaaf, 1998). Since the log likelihood and its gradient can be calculated analytically for ICA, we solved for J via both maximum likelihood learning and MPF, and compared the resulting log likelihoods. Both training techniques were initialized with identical Gaussian noise, and trained on the same data, which accounts for the similarity of individual receptive fields found by the two algorithms. The average log likelihood of the model after parameter estimation via MPF was -120 . 61 nats, while the average log likelihood after estimation via maximum likelihood was -120 . 33 nats. The receptive fields resulting from training under both techniques are shown in Figure 5. MPF minimization was performed by alternating steps of updating the connectivity function g ( x j , x i ) using a Hamiltonian dynamics based scheme, and minimizing the objective function in Equation 15 via LBFGS for fixed g ( x j , x i ). This is described in more detail in Appendix E.

## 5. Summary

We have presented a novel, general purpose framework, called minimum probability flow learning (MPF), for parameter estimation in probabilistic models that outperforms current techniques in both learning time and accuracy. MPF works for any parametric model without hidden state variables, including those over both continuous and discrete state space systems, and it avoids explicit calculation of the partition function by employing deterministic dynamics in place of the slow sampling required by many existing approaches. Because MPF provides a simple and well-defined objective function, it can be minimized quickly using existing higher order gradient descent techniques. Furthermore, the objective function is convex for models in the exponential family, ensuring that the global minimum can be found with gradient descent in these cases. MPF was inspired by the minimum velocity approach developed by Movellan, and it reduces to that technique as well as to score matching and some forms of contrastive divergence under suitable choices for the dynamics and state space. We hope that this new approach to parameter estimation will enable probabilistic modeling for previously intractable problems.

## Acknowledgments

We would like to thank Javier Movellan, Tamara Broderick, Miroslav Dud´ ık, Gaˇ sper Tkaˇ cik, Robert E. Schapire, William Bialek for sharing work in progress and data; Ashvin Vishwanath, Jonathon Shlens, Tony Bell, Charles Cadieu, Nicole Carlson, Christopher Hillar, Kilian Koepsell, Bruno Olshausen and the rest of the Redwood Center for many useful discussions; and the James S. McDonnell Foundation (JSD, PB, JSD) and the Canadian Institute for Advanced Research - Neural Computation and Perception Program (JSD) for financial support.

Appendices Available at

http://redwood.berkeley.edu/jascha/ .

## References

- Ackley, D H, Hinton, G E, and Sejnowski, T J. A learning algorithm for Boltzmann machines. Cognitive Science , 9 (2):147-169, 1985.
- Bell AJ, Sejnowski TJ. An information-maximization approach to blind separation and blind deconvolution. Neural Computation 1995; vol. 7:1129-1159 , 1995.
- Besag, J. Statistical analysis of non-lattice data. The Statistician, 24(3), 179-195 , 1975.
- Brush, S G. History of the Lenz-Ising model. Reviews of Modern Physics , 39(4):883-893, Oct 1967.
- Carreira-Perpi˜ n´ an, M A and Hinton, G E. On contrastive divergence (CD) learning. Technical report, Dept. of Computer Science, University of Toronto , 2004.
- Gutmann, M and Hyv¨ arinen, A. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. Proc. Int. Conf. on Artificial Intelligence and Statistics (AISTATS2010) , 2010.
- Hateren, J. H. van and Schaaf, A. van der. Independent component filters of natural images compared with simple cells in primary visual cortex. Proceedings: Biological Sciences , 265(1394):359-366, Mar 1998.
- Haykin, S. Neural networks and learning machines; 3rd edition . Prentice Hall, 2008.
- Hinton, Geoffrey E, Osindero, Simon, and Teh, Yee-Whye. A fast learning algorithm for deep belief nets. Neural Computation , 18(7):1527-1554, Jul 2006. doi: 10.1162/ neco.2006.18.7.1527.
- Hyv¨ arinen, A. Estimation of non-normalized statistical models using score matching. Journal of Machine Learning Research , 6:695-709, 2005.
- Hyv¨ arinen, A. Connections between score matching, contrastive divergence, and pseudolikelihood for continuousvalued variables. IEEE Transactions on Neural Networks , Jan 2007.
- Jaakkola, T and Jordan, M. A variational approach to Bayesian logistic regression models and their extensions. Proceedings of the Sixth International Workshop on Artificial Intelligence and Statistics , Jan 1997.
- Kappen, H and Rodr´ ıguez, F. Mean field approach to learning in Boltzmann machines. Pattern Recognition Letters , Jan 1997.
- Lyu, S. Interpretation and generalization of score matching. The proceedings of the 25th conference on uncerrtainty in artificial intelligence (UAI*90) , 2009.
- Lyu, S. Personal communication. 2011.
- MacKay, D. Failures of the one-step learning algorithm. Failures of the one-step learning algorithm , Jan 2001.
- Macke, J and Gerwinn, S. Personal communication. 2009.
- Movellan, J R. A minimum velocity approach to learning. unpublished draft , Jan 2008a.
- Movellan, J R. Contrastive divergence in gaussian diffusions. Neural Computation , 20(9):2238-2252, 2008b.
- Movellan, J R and McClelland, J L. Learning continuous probability distributions with symmetric diffusion networks. Cognitive Science , 17:463-496, 1993.
- Pathria, R. Statistical Mechanics . Butterworth Heinemann, Jan 1972.
- Schmidt, M. minfunc. http://www.cs.ubc.ca/ schmidtm/Software/minFunc.html , 2005.
- Schneidman, E, 2nd, M J Berry, Segev, R, and Bialek, W. Weak pairwise correlations imply strongly correlated network states in a neural population. Nature , 440(7087):1007-12, 2006.
- Shlens, J, Field, G D, Gauthier, J L, Grivich, M I, Petrusca, D, Sher, A, Litke, A M, and Chichilnisky, E J. The structure of multi-neuron firing patterns in primate retina. J. Neurosci. , 26(32):8254-66, 2006.
- Sohl-Dickstein, J and Olshausen, B. A spatial derivation of score matching. Redwood Center Technical Report , 2009.
- Swendsen, R.H. and Wang, J.S. Nonuniversal critical dynamics in Monte Carlo simulations. Physical Review Letters , 58(2):86-88, 1987. ISSN 1079-7114.
- Tanaka, T. Mean-field theory of Boltzmann machine learning. Physical Review Letters E , Jan 1998.
- Welling, M and Hinton, G. A new learning algorithm for mean field Boltzmann machines. Lecture Notes in Computer Science , Jan 2002.
- Yuille, A. The convergence of contrastive divergences. Department of Statistics, UCLA. Department of Statistics Papers. , 2005.