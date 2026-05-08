# AI Alignment as Economics - Source Notes

These notes support the Chapter 9 RLHF and AI alignment chapter. They keep the paper-by-paper details out of the chapter prose.

## Social Choice and Heterogeneous Preferences

- `conitzer2024socialchoice`: Position paper arguing that AI alignment with diverse feedback is a collective-choice problem. Safe use: social choice gives tools for deciding whose feedback is collected, how it is aggregated, and what axioms the aggregation rule satisfies.
- `mishra2023socialchoice`: Policy-oriented arXiv/SSRN paper applying social-choice impossibility logic to universal democratic AI alignment. Safe use: caveat-level support for the claim that no single democratic aggregation rule is normatively neutral.
- `siththaranjan2024distributional`: Shows that preference learning with hidden context implicitly aggregates according to Borda count and can create strategic misreporting incentives. Safe use: BTL/RLHF embeds a social welfare function when labelers differ.
- `ge2024axioms`: Builds an axiomatic framework for reward learning from heterogeneous rankings and argues that BTL-style rules fail basic axioms in a linear social-choice setting. Safe use: standard reward learning has normative aggregation content, not just statistical content.
- `park2024heterogeneous`: Provides personalization and aggregation approaches for heterogeneous feedback, including mechanism-design treatment of strategic reports. Safe use: heterogeneous feedback can be handled either by personalization or by explicit aggregation rules.
- `chakraborty2024maxmin`: Proposes MaxMin-RLHF, motivated by an egalitarian social-choice criterion, and shows single-reward RLHF can ignore minority preferences. Safe use: max-min alignment is a pluralistic alternative to average-reward aggregation.
- `munos2023nash`, `wu2024sppo`, `maurarivero2025maximal`: Recast preference optimization as a game over pairwise preferences. Safe use: Nash/maximal-lottery methods are alternatives when preferences are cyclic or non-transitive. The maximal-lottery paper is a 2025 preprint, so cite it as an emerging connection, not settled evidence.

## DPO Variants

- `azar2024preferenceparadigm`: Derives a general preference-optimization family and IPO, which changes the objective/link structure to avoid some DPO overfitting behavior. Safe use: IPO is economically a different preference link/objective, not just an optimizer trick.
- `ethayarajh2024kto`: KTO applies a Kahneman-Tversky/prospect-theory utility transformation and can use binary desirable/undesirable labels. Safe use: KTO is a behavioral-economics-inspired alternative objective.
- `hong2024orpo`, `meng2024simpo`, `chowdhury2024rdpo`: ORPO and SimPO simplify implementation by removing or changing the reference-model reward formulation; rDPO addresses noisy labels. Safe use: implementation/bias/noise variants rather than new welfare foundations.

## Revealed Preference Tests of LLMs

- `chen2023rationalitygpt`: Uses budget-allocation tasks and GARP/CCEI-style revealed-preference tests across risk, time, social, and food domains. Safe use: GPT can pass structured rationality tests in some settings, but results are frame sensitive.
- `golisingh2024preferences`: Marketing Science article on whether LLMs can capture human preferences. The source is metadata verified through INFORMS, but no open PDF was saved. Safe use: cite as a peer-reviewed warning that direct substitution of LLMs for human survey respondents can be misleading.
- `seror2024moral`: Applies revealed-preference tools to moral dilemmas across many LLMs. Safe use: LLM moral preferences can look approximately stable for some models, but prompt/model/domain sensitivity remains central.
- `murawatrawat2025bayesian`: Working paper comparing human and ChatGPT decisions in Bayesian classification tasks. Safe use: caveat-level evidence that later ChatGPT versions perform much closer to Bayes rule in a narrow task.

## Mechanism Design for Feedback

- `sun2024mechanism`: Formalizes LLM fine-tuning with multiple reward models as a mechanism-design problem and extends VCG/affine-maximizer payment ideas for social-welfare-maximizing training rules. Safe use: VCG-style payments become relevant when feedback providers can strategically misreport reward models.
- `kleinebuening2025strategyproof`: Studies strategyproof RLHF and shows a tradeoff between incentive alignment and policy alignment. Safe use: 2025 preprint, use as a caveat about strategic labelers.
- `prelec2004bayesian`, `miller2005eliciting`: Older peer-prediction mechanisms for eliciting truthful subjective information. Safe use: background economics toolkit for feedback elicitation, not LLM-specific evidence.

## Structural Identification

- `cao2021identifiability`: Characterizes non-identifiability in IRL and conditions under which entropy-regularized rewards are recovered up to a constant. Safe use: full reward recovery requires normalizations or additional environments/discount factors.
- `skalse2023invariance`: Formalizes partial identifiability for reward learning data sources, including trajectory comparisons, and asks whether the ambiguity matters for downstream policy optimization. Safe use: preference learning identifies only reward features relevant to the downstream object.
- `knox2022preference`: Shows that the assumed human preference model matters for reward identifiability; regret-based preferences can identify policies in cases where partial returns do not. Safe use: BT over partial returns is a modeling assumption, not a free fact.
- `vanderlaan2025efficient`: Connects MaxEnt IRL and Gumbel-shock DDC through a softmax/log-behavior-policy structure and develops debiased inference for reward-dependent functionals. Safe use: 2025 preprint; cite as an emerging econometric bridge rather than settled textbook material.
