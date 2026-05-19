# Phase 5 — Enrichment notes for ch12_forecasting_rl (revised)

**Revision history.** v1 (2026-05-13 morning) placed all 17 papers in
ch12 across §2/§3/§4/§5. v2 (this file) reflects the cross-chapter
audit: 12 papers stay in ch12, 5 move to ch08 (MOReL, MOPO, Komorowski,
Raghu, Gottesman). The §4 expansion now absorbs world-model material
and the Madeka operational instance that v1 placed in §5 and §3
respectively. See revised plan at
`~/.claude/plans/fore-the-forecasting-cfor-jolly-pretzel.md`.

Status: **review-before-apply draft**. Style audit against existing
chapter: no em dashes, no colons in prose, no `\textbf{}`, no
`\paragraph*{}`, `\citep` / `\citet`, footnotes for secondary detail,
3–6 sentence paragraphs, tables-first.

Execution order (per user directive — intro/conclusion last):
**§4 (centerpiece) → §5 (light) → §2 (one sentence) → §3 (no-op) → §1
intro roadmap + §5 closing wrap-up.** Compile and show PDF after each.

---

## 1. BibTeX entries — 12 active for ch12

Drop the 5 ch08-bound entries from this pass; they live at the bottom
of the file marked DEFERRED for the ch08 enrichment pass.

```bibtex
@inproceedings{FarahmandVAML2017,
  author    = {Farahmand, Amir-massoud and Barreto, Andr{\'e} M. S. and Nikovski, Daniel},
  title     = {Value-Aware Loss Function for Model-based Reinforcement Learning},
  booktitle = {Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {54},
  pages     = {1486--1494},
  year      = {2017}
}

@inproceedings{FarahmandIterVAML2018,
  author    = {Farahmand, Amir-massoud},
  title     = {Iterative Value-Aware Model Learning},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {31},
  year      = {2018}
}

@inproceedings{GrimmVALEQ2020,
  author    = {Grimm, Christopher and Barreto, Andr{\'e} and Singh, Satinder and Silver, David},
  title     = {The Value Equivalence Principle for Model-Based Reinforcement Learning},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {33},
  year      = {2020}
}

@inproceedings{GrimmPVE2021,
  author    = {Grimm, Christopher and Barreto, Andr{\'e} and Farquhar, Gregory and Silver, David and Singh, Satinder},
  title     = {Proper Value Equivalence},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {34},
  year      = {2021}
}

@inproceedings{AyoubVTR2020,
  author    = {Ayoub, Alex and Jia, Zeyu and Szepesv{\'a}ri, Csaba and Wang, Mengdi and Yang, Lin F.},
  title     = {Model-Based Reinforcement Learning with Value-Targeted Regression},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {119},
  pages     = {463--474},
  year      = {2020}
}

@misc{AsadiWasserstein2018,
  author       = {Asadi, Kavosh and Cater, Evan and Misra, Dipendra and Littman, Michael L.},
  title        = {Equivalence Between {Wasserstein} and Value-Aware Loss for Model-based Reinforcement Learning},
  howpublished = {arXiv:1806.01265},
  year         = {2018}
}

@inproceedings{VoelckerCalib2025,
  author    = {Voelcker, Claas and Pedan, Anastasiia and Ahmadian, Arash and Abachi, Romina and Gilitschenski, Igor and Farahmand, Amir-massoud},
  title     = {Calibrated Value-Aware Model Learning with Probabilistic Environment Models},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {267},
  pages     = {61745--61768},
  year      = {2025}
}

@article{Schrittwieser2020MuZero,
  author  = {Schrittwieser, Julian and Antonoglou, Ioannis and Hubert, Thomas and Simonyan, Karen and Sifre, Laurent and Schmitt, Simon and Guez, Arthur and Lockhart, Edward and Hassabis, Demis and Graepel, Thore and Lillicrap, Timothy and Silver, David},
  title   = {Mastering {Atari}, {Go}, Chess and Shogi by Planning with a Learned Model},
  journal = {Nature},
  volume  = {588},
  number  = {7839},
  pages   = {604--609},
  year    = {2020},
  doi     = {10.1038/s41586-020-03051-4}
}

@inproceedings{Antonoglou2022StochMuZero,
  author    = {Antonoglou, Ioannis and Schrittwieser, Julian and Ozair, Sherjil and Hubert, Thomas K. and Silver, David},
  title     = {Planning in Stochastic Environments with a Learned Model},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022}
}

@inproceedings{HafnerDreamer2020,
  author    = {Hafner, Danijar and Lillicrap, Timothy and Ba, Jimmy and Norouzi, Mohammad},
  title     = {Dream to Control: Learning Behaviors by Latent Imagination},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2020}
}

@inproceedings{Chua2018PETS,
  author    = {Chua, Kurtland and Calandra, Roberto and McAllister, Rowan and Levine, Sergey},
  title     = {Deep Reinforcement Learning in a Handful of Trials Using Probabilistic Dynamics Models},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {31},
  year      = {2018}
}

@misc{Madeka2022DeepInventory,
  author       = {Madeka, Dhruv and Torkkola, Kari and Eisenach, Carson and Luo, Anna and Foster, Dean P. and Kakade, Sham M.},
  title        = {Deep Inventory Management},
  howpublished = {arXiv:2210.03137; NeurIPS 2022 invited talk},
  year         = {2022}
}
```

There is no existing `HafnerDreamerV3_2025` entry in `refs.bib` (the
chapter's current DreamerV3 mention is in `RETIERING.md`, not in the
tex — so the existing `\citep{HafnerDreamerV3_2025}` we plan to use in
§4.3 is new too). Verify before adding; if absent, add:

```bibtex
@article{HafnerDreamerV3_2025,
  author  = {Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  title   = {Mastering Diverse Control Tasks through World Models},
  journal = {Nature},
  year    = {2025},
  note    = {arXiv:2301.04104}
}
```

If `HafnerDreamerV3_2025` is already in `refs.bib`, skip.

---

## 2. §4 — centerpiece rewrite

**File:** `tex/04_adp.tex` (full restructure).
**Section title rename:** `\subsection{Sequential decision-aware
learning}` (was *"Approximate dynamic programming for forecasting"*).
**Label unchanged** so existing cross-refs survive:
`\label{sec:adp}`.

Structure: 4.1 (retained material, lightly retitled) → 4.2 (new theory)
→ 4.3 (new empirics) → 4.4 (new operational instance) → Fu outro
(retained, with one transition sentence to §5).

### §4 opener paragraph (replace lines 5–15 of the current file)

```latex
% Section 4: Sequential decision-aware learning
\subsection{Sequential decision-aware learning}
\label{sec:adp}

The single-period decision-focused programme of Section~\ref{sec:dfl}
became a sequential problem the moment the retailer carried inventory
between periods. Today's order $a_t$ affects tomorrow's starting state
$s_{t+1}$, and the realised cost $L(a_t, y_t)$ aggregates into a
discounted return $\sum_{t \ge 0} \gamma^t L(a_t, y_t)$. The standard
dynamic-programming recursion gives the optimal policy via the Bellman
equation $V^\star(s) = \min_a \{ \mathbb E_y[L(a, y) + \gamma
V^\star(s')]\}$. The exact recursion is intractable whenever the state
space is high-dimensional or the next-state transition is unknown,
which is the regime that motivates approximate dynamic programming and
its modern reinforcement-learning descendants. This section follows the
same alignment principle that organises Section~\ref{sec:bridge}.
First we approximate the value function on a fixed transition model
(\S\ref{sec:adp_ndp}); then we ask how to learn the transition model
itself so that it is fit for the downstream value computation
(\S\ref{sec:vaml_ve}); then we examine the empirical architectures
that train both pieces jointly against planning-relevant targets
(\S\ref{sec:world_models}); and finally we show a production-scale
operational instance (\S\ref{sec:madeka}).
```

### §4.1 ADP/NDP — retained material, retitled (lines 17–87 of current file)

```latex
\subsubsection{Approximate dynamic programming for the value function}
\label{sec:adp_ndp}
```

(Body unchanged from current `04_adp.tex` lines 17–87 — Bertsekas
rollout, certainty equivalence, Fernandes GDP nowcasting, Algorithm 1,
panel-data validation, transfer-learning observation.)

**Transition sentence** appended at the end of §4.1 (after current line
87):

```latex
The Fernandes architecture takes the transition model as given and
fits a value-function approximator on it. The complementary question,
namely how to fit the transition model itself when it is going to be
used inside a Bellman backup, is the subject of the next subsection.
```

### §4.2 Value-aware model learning — new

```latex
\subsubsection{Value-aware model learning}
\label{sec:vaml_ve}

The forecast-versus-decision argument of Section~\ref{sec:bridge} has
a sequential analogue. When the forecaster is a transition kernel
$\widehat P(s' \mid s, a)$ to be used inside a Bellman backup,
estimating $\widehat P$ by maximum likelihood treats every component of
the next-state distribution as equally informative. The downstream
planner cares about only one functional, namely
$(\widehat P V)(s, a) = \mathbb E_{s' \sim \widehat P(\cdot \mid s, a)}
[V(s')]$ for value functions $V$ in some candidate class. A
value-aware loss aligns the model-learning objective with that
functional, in direct analogy to the single-period decision-focused
losses of Section~\ref{sec:dfl}.

\citet{FarahmandVAML2017} formalise the alignment as the value-aware
model-learning (VAML) loss
\begin{equation}
\mathcal L_{\mathrm{VAML}}(\widehat P)
= \sup_{V \in \mathcal V} \bigl\| (\widehat P V) - (P^\star V) \bigr\|_\mu,
\label{eq:vaml_loss}
\end{equation}
where $\mathcal V$ is the candidate value class, $P^\star$ is the true
kernel, and $\| \cdot \|_\mu$ is an $L^2$ norm under a state-action
distribution $\mu$. They prove a finite-sample upper bound on the
value-suboptimality of the policy planned in $\widehat P$ that
decomposes into a model-approximation term, an estimation term that
shrinks at the parametric rate, and a complexity term in
$\mathcal V$.\footnote{An alternative regularity-restricted form is
the Wasserstein-VAML equivalence of \citet{AsadiWasserstein2018}.
When $\mathcal V$ is the class of $1$-Lipschitz value functions,
$\mathcal L_{\mathrm{VAML}}$ coincides with the $1$-Wasserstein
distance between $\widehat P(\cdot \mid s, a)$ and $P^\star(\cdot
\mid s, a)$ by Kantorovich-Rubinstein duality, which gives an
optimal-transport reading of value-aware learning.}
\citet{FarahmandIterVAML2018} embeds \eqref{eq:vaml_loss} inside
approximate value iteration. At iteration $k$, the model is fit to
minimise the value-prediction error against the current iterate $V_k$
rather than against the full class $\mathcal V$, which turns the
optimisation into a tractable regression and yields an LSPI-style
performance bound on the converged policy.

The full generalisation is \citet{GrimmVALEQ2020}'s value-equivalence
principle. Two models $\widehat P_1, \widehat P_2$ are value-equivalent
with respect to a policy set $\Pi$ and a value-function set $\mathcal V$
when $(T_{\widehat P_1}^\pi V)(s) = (T_{\widehat P_2}^\pi V)(s)$ for
every $\pi \in \Pi$ and $V \in \mathcal V$, where $T_{\widehat P}^\pi$
is the Bellman operator under model $\widehat P$ and policy $\pi$. The
equivalence class shrinks monotonically as $\Pi$ and $\mathcal V$ grow,
collapsing to the singleton $\{P^\star\}$ in the limit of all policies
and all value functions. The principle subsumes Value Iteration
Networks, the Predictron, Value Prediction Networks, TreeQN, and the
MuZero family of \S\ref{sec:world_models} as instances that train a
learned model against narrow $(\Pi, \mathcal V)$ rather than against
observation likelihood.\footnote{\citet{GrimmPVE2021} sharpen the
result for the MuZero loss specifically. Their \emph{proper} value
equivalence relaxes the equivalence relation to the $k$-step Bellman
operator and bounds the value suboptimality of the planned policy in
terms of the model's loss on a value-policy pair drawn from the
planner. The bound formalises why MuZero can plan well with models
that fail at observation prediction.}

\citet{AyoubVTR2020} convert the value-aware idea into a regret
guarantee. In their value-targeted regression (VTR) algorithm, the
model at episode $k$ is fit by
\begin{equation}
\widehat P_k \in \arg\min_{P \in \mathcal P}
\sum_{t < k} \Bigl(
  (P V_t)(s_t, a_t) - V_t(s_{t+1})
\Bigr)^2,
\label{eq:vtr_loss}
\end{equation}
where $V_t$ is the most recent value-function iterate and $\mathcal P$
is a linear-mixture model class. They prove a regret bound of
$\widetilde O\bigl(d \sqrt{H^3 T}\bigr)$, where $d$ is the
model-class dimension, $H$ is the horizon, and $T$ is the number of
episodes. The result is the first regret guarantee in which model
fitting is driven by a value-prediction loss rather than by a
likelihood, and the rate matches a known $\Omega(\sqrt{d H T})$ lower
bound up to $H$ and logarithmic factors.

Value-aware losses are not without pitfalls. \citet{VoelckerCalib2025}
show that the MuZero loss and several of its variants are uncalibrated
in the surrogate-loss sense of Section~\ref{sec:bridge}. The population
minimiser of the MuZero loss can be a transition kernel whose induced
value function differs from $V^\star$ even when $V^\star$ lies in the
function class. They propose a calibration correction that couples
the model class with a stochastic latent architecture and recovers
the right population minimiser. The result transfers the calibration
question that occupied Section~\ref{sec:bridge} into the sequential
setting and ties the entire VAML-VE-VTR programme back to the proper
scoring-rule machinery.
```

### §4.3 World models for planning — new

```latex
\subsubsection{World models for planning}
\label{sec:world_models}

The value-aware losses of \S\ref{sec:vaml_ve} were defined in
isolation from the planner. In modern world-model architectures the
forecaster, the value function, and the planner are trained jointly
against the same planning-relevant targets, which is the empirical
face of the value-equivalence principle. Three architectural families
anchor the literature.

\citet{Schrittwieser2020MuZero} introduce the MuZero algorithm. A
representation network embeds the observation into a latent state
$h_0$. A dynamics network unrolls latent transitions
$h_{k+1}, r_{k+1} = g(h_k, a_k)$. A prediction network outputs a value
$v(h_k)$ and a policy prior $\pi(h_k)$ at each unrolled step. The three
networks are trained jointly against three planning-relevant targets
at every unrolled depth $k$, namely the empirical Bellman target for
$v$, the bootstrapped policy prior for $\pi$, and the realised reward
for $r$. No observation-reconstruction loss is used. The latent
dynamics network is therefore a value-equivalent model in the sense
of \citet{GrimmVALEQ2020} rather than a generative forecaster of
pixels. Embedded inside Monte Carlo Tree Search, the resulting agent
matches AlphaZero on Go, chess, and shogi without access to game
rules, and attains state of the art on $57$ Atari games without a
simulator. \citet{Antonoglou2022StochMuZero} extend the algorithm to
stochastic environments by replacing the deterministic latent
dynamics with afterstate-plus-chance-node representations, with
matching performance on $2048$ and backgammon.

A complementary line of work uses an explicit probabilistic model of
the dynamics together with model-predictive control.
\citet{Chua2018PETS} introduce probabilistic ensembles with
trajectory sampling. The dynamics model is an ensemble of probabilistic
neural networks that captures both aleatoric uncertainty (through the
per-network output variance) and epistemic uncertainty (through
ensemble disagreement). At each control step, candidate action
sequences are scored by Monte Carlo rollouts through the ensemble and
the first action of the best sequence is executed. The MPC step is a
one-step rollout in the sense of \citet{BertsekasRollout2022}; the
ensemble is the probabilistic forecast that feeds it. The combination
attains the asymptotic performance of model-free baselines such as SAC
and PPO while using between $8$ and $125$ times fewer environment
samples on standard continuous-control tasks, which is the cleanest
empirical case for probabilistic-dynamics-plus-rollout on a
benchmark suite that an econometrician would recognise.

The third family is the Dreamer line of recurrent state-space models.
\citet{HafnerDreamer2020} train a recurrent latent dynamics model from
pixels, plan in the compact latent space rather than in observation
space, and update the actor by backpropagating analytic value
gradients through imagined trajectories. DreamerV3
\citep{HafnerDreamerV3_2025} scales the recipe across more than $150$
control tasks under a single hyperparameter configuration and is the
first algorithm to collect diamonds in Minecraft from pixels and
sparse rewards without human demonstrations.\footnote{The Dreamer
architecture keeps an observation-reconstruction term in the
world-model loss, which makes it not strictly value-equivalent. The
reconstruction loss regularises the latent representation in ways
that the MuZero-style pure value-target objective does not. The
\citet{VoelckerCalib2025} calibration analysis of \S\ref{sec:vaml_ve}
applies most directly to the MuZero family and remains an open
question for Dreamer-style models.}
```

### §4.4 Operational instance: Amazon inventory — new

```latex
\subsubsection{Operational instance: deep inventory management}
\label{sec:madeka}

The architectures of \S\ref{sec:world_models} were demonstrated on
game-playing and continuous-control benchmarks. The same architectural
move, namely a differentiable forecasting model embedded inside a
sequential decision loop and trained against realised cost, has been
deployed at production scale on a problem an economist would recognise
immediately. \citet{Madeka2022DeepInventory} treat periodic-review
inventory control with stochastic vendor lead times, lost sales,
correlated demand, and price matching as a sequential decision problem
whose forecast object is the joint distribution of demand and lead
time over a finite horizon. The technical innovation is
\emph{DirectBackprop}, a differentiable historical-data simulator that
lets policy gradients flow through the realised inventory cost while
the underlying exogenous processes are sampled from history. A learned
ordering policy parametrised by a deep network beats classical
newsvendor and model-free reinforcement-learning baselines on more
than $10{,}000$ Amazon SKUs and is the direct multi-period
generalisation of the \citet{BanRudin2019} single-period
ERM.\footnote{\citet{Madeka2022DeepInventory} also prove a reduction
result that gives an economics-audience reader a familiar handle. For
the exogenous-decision-process subclass, in which the dynamics depend
on exogenous shocks but the agent's actions do not feed back into
demand or lead time, the reinforcement-learning problem reduces to
supervised learning against the realised cost. The reduction recovers
the $\sqrt{p/n}$ regret rate of \citet{BanRudin2019} as a special
case and provides a finite-sample bound for the sequential version.}
```

### Existing Fu ensembling outro — retained with one transition tweak

Keep current lines 89–98 of `04_adp.tex` (Fu RL-for-ensembling
paragraph) unchanged. Adjust the closing transition sentence (current
lines 100–102) to advertise §5's revised scope:

```latex
The next section inverts the relationship between forecasting and
RL once more. Instead of using RL machinery to train a forecaster, or
training a forecaster jointly with a planner, it treats the forecaster
as the policy itself.
```

---

## 3. §5 — light edits

**File:** `tex/05_forecasting_in_rl.tex`.

**Section rename** (line 2):

```latex
\subsection{Forecasting as policy}
\label{sec:forecasting_in_rl}
```

(Was *"Forecasting inside RL"*. Label unchanged.)

**Opener paragraph (lines 5–11)** — keep as-is. The "two flavours"
framing is still right because the two remaining ch12-owned flavours
are trajectory-generative + counterfactual-OPE. (World models moved
to §4.3.)

**Counterfactual-OPE cross-reference** — append one sentence at the end
of the existing paragraph that ends at line 113 (current text: "...the
econometric remedy and remains an open research question for the
counterfactual-RL programme.").

```latex
The econometric identification toolkit for sequential confounded
settings (backdoor adjustment, instrumental variables, proximal and
POMDP bridges, mediator-based identification) is developed in
Chapter~\ref{ch:causal_rl}; the structural-causal-model machinery of
this subsection is its structural complement rather than its
substitute.
```

The `\ref{ch:causal_rl}` requires that the ch10 master file has a
chapter-level `\label{ch:causal_rl}`. Check before applying; if absent,
use a plain section reference instead (e.g.
`Section~\ref{section:causal_rl}` if that label exists).

No other §5 edits in this pass. The final wrap-up paragraph (lines
205–217) is edited LAST per the intro/conclusion-last directive (see
§6 below).

---

## 4. §2 — one calibration forward-pointer

**File:** `tex/02_bridge.tex`, append at end of paragraph ending line
85 (the "...not the value-maximising forecast under~\eqref{eq:gp_value}"
paragraph). The new sentence becomes the paragraph's closer.

```latex
A sequential analogue of the same calibration question recurs in the
model-based reinforcement-learning literature, where a value-targeted
surrogate loss that is itself uncalibrated can return the wrong
transition model and the wrong value function even at the population
level \citep{VoelckerCalib2025}, a point we take up in
Section~\ref{sec:adp}.
```

Adds one bib reference (`VoelckerCalib2025`). No other §2 changes.

---

## 5. §3 — no-op

`tex/03_dfl.tex` is **not edited** in this pass. The v1 plan placed a
Madeka paragraph in §3 between the Cao-Xu closer and the AR(1)
simulation; the revised plan moves that paragraph to §4.4. Confirm
during execution that no Madeka prose ends up in §3.

---

## 6. §1 intro roadmap + §5 final wrap-up — LAST

These two edits run together at the end so the roadmap matches the
final §4 structure and the §5 wrap-up reflects the chapter's spine
correctly.

### §1 (`01_intro.tex`) — update the roadmap sentence

Current sentence (lines 22–28):

> "We develop the bridge between forecast loss and decision loss
> (Section~\ref{sec:bridge}), the differentiable machinery that
> operationalises the bridge for single-period decisions
> (Section~\ref{sec:dfl}), the sequential extension that brings
> approximate dynamic programming and reinforcement learning into the
> loop (Section~\ref{sec:adp}), and the converse use of forecasting
> inside RL agents that plan over trajectories
> (Section~\ref{sec:forecasting_in_rl})."

Replacement:

```latex
We develop the bridge between forecast loss and decision loss
(Section~\ref{sec:bridge}), the differentiable machinery that
operationalises the bridge for single-period decisions
(Section~\ref{sec:dfl}), the sequential extension that aligns
transition-model learning with the downstream value functional and is
the empirical home of the modern world-model literature
(Section~\ref{sec:adp}), and the alternative paradigm in which the
forecaster doubles as the policy itself or estimates a counterfactual
value of an alternative policy from logged data
(Section~\ref{sec:forecasting_in_rl}).
```

One sentence, same length, advertises the revised §4 and §5 scopes.

### §5 final wrap-up (`05_forecasting_in_rl.tex` lines 205–217)

Current paragraph ends with "...the extension to action-conditional
foundation forecasters remains a research frontier."

Replacement (one paragraph that reads as the chapter's conclusion):

```latex
The two simulations of the chapter make the same methodological point
on two different decision structures. The decision-focused AR(1)
forecaster of Section~\ref{sec:forecasting_rl_sim} beats the
prediction-focused OLS forecaster on realised newsvendor cost
precisely when the noise model is misspecified. The counterfactual
OPE estimator above beats the model-based estimator on RMSE precisely
when the outcome model is misspecified. The corrective in both cases
is a training or evaluation objective that knows about the
decision-induced loss, not the marginal forecast loss. The
value-aware programme of Section~\ref{sec:vaml_ve} carries the same
prescription to the sequential setting, and the world-model
architectures of Section~\ref{sec:world_models} demonstrate that the
prescription scales empirically. Three frontiers remain. Decision-focused
regret bounds at rate $T^{-1/4}$ exist in restricted settings under a
margin condition on the decision polytope \citep{Capitaine2025}; the
extension to action-conditional foundation forecasters is open; and
the calibration correction of \citet{VoelckerCalib2025} on
value-targeted surrogate losses is at the population level only, with
a finite-sample analogue not yet established.
```

The new closing paragraph absorbs the existing wrap-up text and adds a
sentence explicitly connecting back to §4.2 (`sec:vaml_ve`) and §4.3
(`sec:world_models`).

---

## 7. Compile checklist

After each section edit, from `docs/`:

```bash
cd docs && pdflatex -shell-escape -jobname=ch12_forecasting_rl \
  "\def\chapterfile{../ch12_forecasting_rl/tex/forecasting_rl}\input{compile_chapter}" \
  && bibtex ch12_forecasting_rl \
  && pdflatex -shell-escape -jobname=ch12_forecasting_rl \
       "\def\chapterfile{../ch12_forecasting_rl/tex/forecasting_rl}\input{compile_chapter}" \
  && pdflatex -shell-escape -jobname=ch12_forecasting_rl \
       "\def\chapterfile{../ch12_forecasting_rl/tex/forecasting_rl}\input{compile_chapter}"
```

Verify in the log:

- no `Citation '...' undefined` warnings for any of the 12 new keys
- no `Reference '...' undefined` for `sec:adp_ndp`, `sec:vaml_ve`,
  `sec:world_models`, `sec:madeka`
- the resulting PDF lives at `docs/ch12_forecasting_rl.pdf`
- expected page-count delta after §4 only: roughly +2.5 pp

Style audit per `CLAUDE.md`:

```bash
grep -E '\\textbf|—|\\paragraph\*' ch12_forecasting_rl/tex/*.tex
```

Should return nothing.

---

## 8. DEFERRED — for ch08 enrichment pass

The five papers below are not used in ch12. Their prose drafts are
preserved here for the ch08 pass.

### 8a. ch08 bibtex entries (deferred)

```bibtex
@inproceedings{KidambiMOReL2020,
  author    = {Kidambi, Rahul and Rajeswaran, Aravind and Netrapalli, Praneeth and Joachims, Thorsten},
  title     = {{MOReL}: Model-Based Offline Reinforcement Learning},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {33},
  year      = {2020}
}

@inproceedings{YuMOPO2020,
  author    = {Yu, Tianhe and Thomas, Garrett and Yu, Lantao and Ermon, Stefano and Zou, James and Levine, Sergey and Finn, Chelsea and Ma, Tengyu},
  title     = {{MOPO}: Model-Based Offline Policy Optimization},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {33},
  year      = {2020}
}

@article{Komorowski2018AIClinician,
  author  = {Komorowski, Matthieu and Celi, Leo A. and Badawi, Omar and Gordon, Anthony C. and Faisal, A. Aldo},
  title   = {The Artificial Intelligence Clinician Learns Optimal Treatment Strategies for Sepsis in Intensive Care},
  journal = {Nature Medicine},
  volume  = {24},
  number  = {11},
  pages   = {1716--1720},
  year    = {2018},
  doi     = {10.1038/s41591-018-0213-5}
}

@misc{Raghu2018SepsisMB,
  author       = {Raghu, Aniruddh and Komorowski, Matthieu and Singh, Sumeetpal},
  title        = {Model-Based Reinforcement Learning for Sepsis Treatment},
  howpublished = {NeurIPS 2018 Machine Learning for Health (ML4H) Workshop; arXiv:1811.09602},
  year         = {2018}
}

@article{Gottesman2019Critique,
  author  = {Gottesman, Omer and Johansson, Fredrik and Komorowski, Matthieu and Faisal, Aldo and Sontag, David and Doshi-Velez, Finale and Celi, Leo Anthony},
  title   = {Guidelines for Reinforcement Learning in Healthcare},
  journal = {Nature Medicine},
  volume  = {25},
  number  = {1},
  pages   = {16--18},
  year    = {2019},
  doi     = {10.1038/s41591-018-0310-5}
}
```

### 8b. ch08 prose drafts (deferred)

For the ch08 *Pessimism Principle* subsection, the MOReL/MOPO block:

> Two model-based instantiations of the pessimism principle dominate
> the offline literature. \citet{KidambiMOReL2020} learn an ensemble
> of probabilistic dynamics models and construct a pessimistic Markov
> decision process by routing transitions on which the ensemble
> disagrees into an absorbing HALT state with a large negative reward.
> Planning in the pessimistic MDP yields a policy whose value is a
> lower bound on its true value, and the resulting algorithm is
> minimax-optimal up to logarithmic factors. \citet{YuMOPO2020} take
> the complementary route: leave the dynamics ensemble untouched and
> penalise the per-step reward by a quantity proportional to the
> ensemble's disagreement, $\tilde r(s, a) = \widehat r(s, a) -
> \lambda u(s, a)$. The planned policy maximises the same value lower
> bound. The two recipes illustrate the model-based analogue of the
> propensity-versus-outcome trade-off in observational causal
> inference. Pessimism via state truncation is the
> Horvitz-Thompson posture; pessimism via reward shading is the
> augmented-IPW posture.

For the ch08 *Operational instance* (clinical application + critique):

> The canonical real-world deployment of offline RL in critical-care
> medicine is the AI Clinician of
> \citet{Komorowski2018AIClinician}, who discretise patient
> trajectories from $96{,}156$ sepsis admissions in the MIMIC-III and
> eICU databases into a finite-state Markov decision process, fit a
> tabular Q-iteration policy over fluid resuscitation and vasopressor
> dosing, and evaluate the learned policy off-policy via weighted
> importance sampling against the clinicians' realised dosing. The
> headline observation is that mortality is lowest in the subgroup
> of patients whose actual doses matched the AI's recommendation.
> \citet{Raghu2018SepsisMB} run the model-based companion analysis on
> the same data, learning a continuous-state dynamics model and
> planning in latent space.
>
> The sepsis line is also the most-cited cautionary tale in clinical
> RL. \citet{Gottesman2019Critique} document that ICU offline RL is
> brittle to behaviour-policy specification, to reward shaping, to
> action discretisation, and most of all to unobserved confounding
> (a sicker patient typically receives more aggressive treatment, so
> the policy that prescribes more treatment in sick states can match
> the clinician simply by recovering the confounder). The chapter's
> recurring takeaway sharpens to its strongest form in this context.
> A good forecast of patient state is not by itself a good treatment
> policy.

To be picked up in a separate ch08 pass.

---

## 9. Drift notes

- §4 label `sec:adp` is retained even though the subsection has been
  renamed. This is intentional, so existing cross-references from §3,
  §5, and §1 do not break.
- §5 subsection title rename does not change `sec:forecasting_in_rl`
  label.
- New labels added: `sec:adp_ndp`, `sec:vaml_ve`, `sec:world_models`,
  `sec:madeka`. All four are subsection-level inside §4.
- The §5 final wrap-up paragraph absorbs the chapter's conclusion,
  which is appropriate because the chapter has no separate `§6
  Conclusion` (see `forecasting_rl.tex` comment).
