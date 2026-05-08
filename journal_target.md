# Strictly Survey-Only Journals for "RL for Economics": A Decision-Ready Map

## TL;DR
- **Only three strictly-survey-only journals in lighter / applied / code-friendly fields are realistic targets for this 138-page RL-for-Economics survey: ACM Computing Surveys (CSUR), Foundations and Trends in Machine Learning (FnT-ML), and — with caveats — Computer Science Review.** A fourth, Foundations and Trends in Information Retrieval, fits only the dynamic-pricing / recommender carve-outs.
- **Honest assessment: in "lighter, applied" CS/AI publishing, strictly-survey-only journals are rare.** Most applied-ML surveys land in review tracks of regular research journals (Artificial Intelligence Review, Knowledge Engineering Review, IEEE CI Magazine, AI Magazine, TMLR's Survey Certification). The user's filter ("survey-only AND applied AND open to unsolicited submissions AND not invitation-only") leaves a very small set.
- **Best single bet:** ACM Computing Surveys for the full survey (code-friendly, unsolicited, applied-CS scope, 2024 Impact Factor of 28 — top of "Computer Science, Theory & Methods" category, and explicitly welcomes ML-↔-domain bridge work). FnT-ML is the natural second target because Tibshirani's editorial scope explicitly names *economics* as a welcomed bridging field. Carve out two thematic monographs (Themes 3 and 6) and submit them in parallel.

---

## Key Findings

1. **The "strictly survey-only" filter eliminates most candidates.** Of the ~20 outlets evaluated, only ~5 have a publication mandate exclusively dedicated to surveys, reviews, or tutorials. The remainder are regular research journals that *also* publish reviews — these are excluded per the brief.

2. **Among survey-only outlets that accept unsolicited submissions, the realistic shortlist is small:**
   - **ACM Computing Surveys (CSUR)** — flagship, fully unsolicited, code-friendly.
   - **Foundations and Trends in Machine Learning (FnT-ML)** — abstract-first, explicitly welcomes ML↔statistics↔economics↔optimization bridges.
   - **Foundations and Trends in Information Retrieval (FnT-IR)** — only fits the recommendation/dynamic-pricing carve-outs.
   - **Computer Science Review (Elsevier)** — strictly survey-only, but the Guide for Authors uses invitation-leaning wording ("provide … to the Editor who invited the author"). Treat as a back-up rather than primary target.
   - **(Excluded under the brief's filters:)** AI Communications surveys are *invitation-only* per IOS Press; IEEE Communications Surveys & Tutorials is survey-only but communications-only in scope.

3. **Practitioner / mixed-content venues that welcome unsolicited surveys but are NOT strictly survey-only** — listed for honesty, with a clear flag that they don't meet the "strict" filter:
   - **AI Magazine (AAAI/Wiley)** — accepts unsolicited surveys; OA; AAAI sponsorship covers the APC.
   - **IEEE Computational Intelligence Magazine (IEEE-CIS)** — "Surveys and expository submissions are also welcome."
   - **IEEE Intelligent Systems (IEEE Computer Society)** — explicitly lists "tutorials; surveys" among encouraged article types.
   - **Transactions on Machine Learning Research (TMLR)** — operates a "Survey Certification" sub-track, not a survey-only journal, but de facto a survey venue for the ML community.

4. **Venues to explicitly *exclude*** because the user's filter rules them out:
   - **Communications of the ACM** — its own author guidance says: *"Surveys. ACM Computing Surveys, not CACM, is the place to publish a survey."* Do not target.
   - **AI Communications (IOS Press)** — "Survey and Tutorial papers are normally only accepted under invitation." Invitation-only ⇒ excluded.
   - **Annual Reviews of Economics, Computer Science, Statistics, Control, etc.** — invitation-only per the brief.
   - **JASSS, EPJ Data Science, SN Computer Science, Knowledge Engineering Review, Artificial Intelligence Review, AI Perspectives & Advances, Discover Artificial Intelligence** — regular research journals with review tracks, not survey-only.

---

## Details

### A. The shortlist (strictly survey-only, unsolicited, applied-friendly, US/UK/EU/Canada)

#### A1. ACM Computing Surveys (CSUR) — *the primary target*
- **Publisher:** Association for Computing Machinery (ACM), United States.
- **Co-Editors-in-Chief (term 1 July 2025 – 30 June 2028):** **My T. Thai** (University of Florida) and **Hanghang Tong** (University of Illinois Urbana-Champaign).
- **Standing:** 2024 Impact Factor of 28, ranked #1 in "Computer Science, Theory & Methods" (per the University of Florida Warren B. Nelms Institute announcement of Thai's appointment).
- **Unsolicited submissions:** Yes. Submissions are made via ACM Manuscript Central (https://mc.manuscriptcentral.com/csur). No invitation required.
- **Code/simulation policy:** Strongly code-friendly. The journal's own description says it welcomes "Contributions which bridge existing and emerging technologies (such as machine learning) with a variety of science and engineering domains in a novel and interesting way." This *exact* framing fits "RL ↔ economics."
- **Open access / fees:** ACM Open transition; authors at ACM-Open-participating institutions can publish OA at no charge; subscription publication remains an option.
- **Length norms:** Surveys typically 30–50 pages with 100–300+ references. A 138-page submission is long but not unprecedented; the survey would likely need to be condensed or split.
- **Editorial bar:** CSUR demands an *original taxonomy or analytical framework*, not a chronological literature catalog. The survey's organizing structure (DP → modern RL → applications → limitations) needs to be foregrounded as a contribution, not a tour.
- **Recent applied-ML / RL surveys:** "A Survey on Deep Reinforcement Learning for Data Processing and Analytics"; multiple recommender-systems surveys; ML-↔-domain bridge work is a recurring CSUR pattern.

#### A2. Foundations and Trends in Machine Learning (FnT-ML) — *the natural second target*
- **Publisher:** Now Publishers, distributed by Emerald (US-based).
- **Editor-in-Chief (2022– ):** **Ryan Tibshirani**, UC Berkeley. He concurrently chairs the UC Berkeley Statistics Department (effective 1 July 2025) and is also founding co-Editor-in-Chief of Foundations and Trends in Statistics (with Rina Foygel Barber, since 2024) — both signals that he actively oversees the FnT survey-monograph program in the ML/statistics space.
- **Scope (verbatim):** *"Foundations and Trends® welcomes monographs that touch on fundamental problems in machine learning from theoretical, methodological, and/or computational perspectives. We are particularly interested in monographs that seek to bridge such problems and perspectives with those from related fields, including (but not limited to) statistics, economics, and optimization."* (https://www.emerald.com/ftmal) — note the explicit mention of *economics*.
- **Submission policy (verbatim):** *"Foundations and Trends® in Machine Learning publishes exclusively long (± 100 pages) review and tutorial papers. … Original research papers will not be considered for publication and we will not acknowledge receipt. … In the first instance, send an abstract for initial review to the publisher. After this initial submission, a preliminary acceptance may follow. The full draft paper will be subject to a reviewing process to ensure quality standards and balance before being finally accepted."* (https://www.emerald.com/ftmal/pages/author-guidelines)
- **Length norms:** ±100 pages — the FnT-ML format is purpose-built for monograph-length surveys, which matches the 138-page draft very well.
- **Code/simulation policy:** No prohibition on code; FnT-ML monographs frequently embed pseudocode, simulation results, and reproducibility appendices.
- **Open access / fees:** Subscription model; no APC required. Final monograph also typically distributed as a Now Publishers eBook.
- **Recent applied-RL relevance:** Kairouz, McMahan, Avent, Bellet et al., "Advances and Open Problems in Federated Learning" (FnT-ML, 2021) — a clear precedent for a long, application-oriented monograph.

#### A3. Foundations and Trends in Information Retrieval (FnT-IR) — *only for the dynamic-pricing/recommender carve-out*
- **Publisher:** Now Publishers / Emerald.
- **Co-Editors-in-Chief:** **Falk Scholer** (RMIT University, Australia) and **Pablo Castells** (Universidad Autónoma de Madrid; Emerald page lists "Universidad Complutense de Madrid" — the named individual is unambiguous, the affiliation likely a publisher-side typo).
- **Submission policy:** Same FnT abstract-first model. Unsolicited abstracts welcome; original research papers rejected.
- **Scope fit:** Tight to IR / recommender systems / search. RL-for-recommendations and RL-for-dynamic-pricing both fit; broader RL-for-economics does not.
- **Length norms:** ~100-page monograph.

#### A4. Computer Science Review (Elsevier) — *strictly survey-only, but invitation-leaning*
- **Publisher:** Elsevier (Ireland).
- **Editor-in-Chief (2025–):** **Jan Kratochvíl**, Charles University, Department of Applied Mathematics, Prague (single EiC listed at https://www.sciencedirect.com/journal/computer-science-review/about/editorial-board).
- **Unsolicited submissions:** *Possible but discouraged in practice.* The Guide for Authors says: *"Authors should provide a PDF or PS copy of their manuscript to the Editor who invited the author to write the survey."* It also enforces a strict expert-author bar: *"At least one author is expected to have at least three papers on the subject of the survey published in high impact factor journals or highly ranked conferences and listed in the bibliographic references of your submission."*
- **Length norms:** ~30 printed pages / ~20,000 words is the optimal length; minimum 20 typeset pages for the author-payment honorarium.
- **Author payment / fees:** *"Submissions are free of charge and recognizing the work involved in preparing a review article, Elsevier will pay authors for their contributions to Computer Science Review. This amount will be Euro 400 per accepted article for the authors; provided the article meets minimum length requirements (at least 20 typeset pages…)."* OA option is available at USD 4,420 APC.
- **Code/simulation policy:** Not specifically advertised; expository overviews of open problems are the focus.
- **Verdict:** *Treat as a fall-back, not a primary target.* The 138-page draft is roughly 4× the journal's preferred length, and the invitation-leaning posture means a cold submission needs an exceptionally strong cover-letter pitch, ideally preceded by an email to Kratochvíl.

### B. The honest "near-miss" tier (mixed-content but unsolicited-survey-friendly)

These are NOT strictly survey-only and therefore fall outside the brief's strict filter. They are listed only because the alternative is to lie about how few survey-only outlets exist.

| Venue | Publisher | Survey-only? | Unsolicited? | Code/Applied fit |
|---|---|---|---|---|
| **AI Magazine** | AAAI / Wiley (US) | No (mixed magazine) | Yes — Free Format submission via Research Exchange (https://wiley.atyponrex.com/journal/AAAI). Co-editors: Odd Erik Gundersen and K. Brent Venable. | High — practitioner-facing, fully OA (AAAI sponsorship covers the $2,400 APC for accepted authors) |
| **IEEE Computational Intelligence Magazine** | IEEE CIS (US) | No (mixed) | Yes — *"Surveys and expository submissions are also welcome."* Survey papers up to 15 pages before page-charge surcharge. **EiC (effective 1 January 2026, current as of May 2026): Min Jiang, Xiamen University.** | High — applied / computational intelligence framing |
| **IEEE Intelligent Systems** | IEEE Computer Society (US) | No (mixed) | Yes — *"Also encouraged are application features … tutorials; surveys; and case studies."* | High — explicitly applications-oriented; sponsored by AAAI, EurAI, BCS |
| **Transactions on Machine Learning Research (TMLR)** | JMLR / OpenReview (US/Canada) | No (research journal with **Survey Certification** track) | Yes — open submissions; surveys earn a "Survey Certification" badge if exceptional. **Current EiCs (as of May 2026): Laurent Charlin (HEC Montréal/Mila, joined 1 January 2026), Gautam Kamath (Waterloo), Naila Murray (Meta), Nihar B. Shah (CMU).** | Very high — code-first ML community, OA, no APC |

I would *not* tell the author to submit *as a survey-only target* to these venues — but for Theme 6 (practical RL limitations, simulation/code-heavy) **TMLR's Survey Certification track is arguably the single best fit in all of ML publishing**, even though TMLR itself is not survey-only.

### C. Theme-to-journal mapping (lighter applied venues only)

| Theme | Best survey-only fit | Honest second-best |
|---|---|---|
| **1. RL foundations (DP → modern RL)** | **ACM Computing Surveys** — code-friendly, broad CS audience, "bridge" framing welcomed. | FnT-ML if framed as a tutorial-monograph with statistics/economics bridge. |
| **2. Causal RL (off-policy eval, confounded MDPs, observational dynamic policies)** | **FnT-ML** — Tibshirani's "ML ↔ statistics ↔ economics" mandate is a near-perfect fit; econometric off-policy evaluation reads as exactly that bridge. | ACM Computing Surveys if framed broadly enough. |
| **3. Economic structure in bandits and RLHF (rationality axioms, dynamic pricing)** | **FnT-ML** for the bandits/RLHF monograph; **FnT-IR** for the dynamic-pricing/recommender slice (Falk Scholer / Pablo Castells). | ACM CSUR for the LLM-alignment angle. |
| **4. Multi-agent RL & equilibrium selection (algorithmic collusion, equilibrium learning)** | **ACM Computing Surveys** — multi-agent RL surveys are a recurring CSUR topic. | AI Magazine (mixed venue) for a more practitioner-facing version with collusion case studies. |
| **5. Robust RL & ambiguity aversion** | *Fold into Theme 2 or Theme 1.* Lighter applied venues will not absorb a pure-theory robust-control monograph. If kept separate, IEEE Computational Intelligence Magazine (Min Jiang, EiC) is the most plausible mixed-content home. | — |
| **6. Practical limitations of RL (brittleness, sample inefficiency, hyperparameter sensitivity) — simulation/code-heavy** | **ACM Computing Surveys** if framed as a benchmark-driven survey with public code. | **TMLR with Survey Certification** is, candidly, the natural ML-community home for this theme; it is not "survey-only" but it is the venue where the ML community actually reads code-heavy practical-limitations surveys. |

### D. Honest assessment on the original premise

Are there really "many survey-only journals in lighter applied fields"? **No.**

- The "Foundations and Trends" series is the only sustained publisher mandate worldwide for survey-only monographs in CS/ML/IR/HCI/Databases — and several FnT titles (HCI, Databases, Signal Processing) are wrong-fit for RL-economics.
- ACM Computing Surveys is the only general-purpose survey-only journal in CS that accepts unsolicited submissions across all subfields, including applied AI.
- Computer Science Review is survey-only but invitation-leaning.
- Every other "review" or "magazine" venue you might consider — AI Magazine, IEEE CI Magazine, IEEE Intelligent Systems, AI Communications, Knowledge Engineering Review, Artificial Intelligence Review, EPJ Data Science, JASSS, SN Computer Science, AI Perspectives & Advances, Discover Artificial Intelligence — publishes both research and reviews and therefore fails the strict filter.

The realistic universe of strictly-survey-only, lighter-applied, unsolicited-submission, reputable journals for an RL-for-Economics piece is **3–4 outlets**, not 15–20.

---

## Recommendations

**Stage 1 — Two parallel primary submissions (within 30 days):**

1. **ACM Computing Surveys** as a full survey, but condensed from 138 pages to ~50 pages with the original taxonomy (DP → modern RL → applied carve-outs → practical limitations) made the explicit contribution. The 138-page version becomes a supplementary technical report on arXiv linked from the manuscript. Cover letter should pitch the "what economics teaches AI/ML" framing as the original organizing structure. Address to co-EiCs Thai and Tong.
2. **Foundations and Trends in Machine Learning** — submit an abstract for a monograph titled to highlight the *ML ↔ statistics ↔ economics* bridge that Tibshirani's mandate explicitly welcomes. Length (~100 pages) is a near-perfect fit for the existing draft. Two-step gate: abstract first to the publisher, then preliminary acceptance, then full submission.

**Stage 2 — Thematic carve-outs (parallel monographs / shorter pieces, 60–90 days):**

3. **Theme 3 carve-out (dynamic pricing / RL-recommender slice) → FnT-IR** abstract to Falk Scholer / Pablo Castells.
4. **Theme 6 carve-out (practical RL limitations, code-heavy) → TMLR with target Survey Certification.** This is not a survey-only journal but it *is* where the ML community actually reads this kind of paper, and the certification gives it the survey-paper visibility.

**Stage 3 — Fall-backs if Stage 1 is rejected:**

5. **Computer Science Review** for a 30-page version — but only after a pre-submission email to EiC Jan Kratochvíl explaining fit, given the invitation-leaning Guide for Authors language.
6. **AI Magazine** for a shorter (~10–15 page) practitioner-facing version with the simulation code as the central artefact. Not survey-only per the brief's strict definition, but openly accepts unsolicited surveys, OA, no APC for the author.

**Decision thresholds that would change these recommendations:**

- *If FnT-ML's editor responds to the abstract within 60 days with a preliminary accept,* drop CSUR and focus on the FnT-ML monograph (it's a better length match).
- *If CSUR returns a desk-reject citing scope or "literature catalog" framing,* rewrite around an explicit taxonomy/contribution and resubmit after the 12-month resubmission embargo, OR pivot to FnT-ML as primary.
- *If both Stage 1 venues reject,* the right move is to split the survey into the six thematic monographs and submit them to a mix of FnT venues + TMLR + AI Magazine, rather than insisting on a single "survey-only journal" home.

---

## Caveats

- **Editorial-board details have shelf life.** EiC tenures verified for ACM CSUR (Thai/Tong, July 2025–June 2028), FnT-ML (Ryan Tibshirani, 2022–), FnT-Stat (Tibshirani / Rina Foygel Barber, since 2024), FnT-IR (Falk Scholer / Pablo Castells), FnT-HCI (Florian 'Floyd' Mueller / Youn-kyung Lim), Computer Science Review (Jan Kratochvíl), AI Magazine (Odd Erik Gundersen / K. Brent Venable), IEEE CIM (Min Jiang, EiC effective 1 January 2026), TMLR (Charlin/Kamath/Murray/Shah as of January 2026, with Hugo Larochelle having completed his founding-EiC term). Re-verify before submitting.
- **The Pablo Castells affiliation listed on Emerald's FnT-IR page reads "Universidad Complutense de Madrid"; Castells is publicly known to be at Universidad Autónoma de Madrid. The name is unambiguous; the affiliation appears to be a publisher-side typo.**
- **Computer Science Review's wording is ambiguous** — *"the Editor who invited the author"* suggests an invitation model, but the journal does run an Editorial Manager portal and does accept some unsolicited submissions in practice. Authors should email the EiC before investing time.
- **Length mismatch:** A 138-page draft is too long for ACM CSUR (preferred 30–50 pp), too long for Computer Science Review (~30 pp), but a near-perfect length for any FnT monograph (~100 pp). This is a strong signal that the FnT route — particularly FnT-ML — is the most natural format-fit.
- **The brief asked specifically about "lighter, applied, code/simulation-friendly" venues; none of the qualifying journals advertise a code-mandatory policy.** ACM CSUR most overtly welcomes code-bridging work; TMLR (mixed venue, not survey-only) is the one ML venue where reviewers routinely run author-supplied code as part of review.
- **Excluded outlets that some readers might expect to see:** Annual Review of Economics / Statistics / Control / etc. (invitation-only — ruled out); WIREs Computational Statistics / Data Mining (commissioned-only); Synthesis Lectures (book series); SIAM Review / Statistical Science / Statistics Surveys / Probability Surveys (too math/stats-theory per the brief); Annals of Operations Research / EJOR / JEDC / Computational Economics / Machine Learning / KBS / ESWA / Frontiers in X (regular research journals with review tracks). Communications of the ACM is ruled out because its own author guidance redirects survey papers to ACM Computing Surveys. AI Communications (IOS Press) is ruled out because surveys are accepted only by invitation. Springer's Discover Artificial Intelligence and AI Perspectives & Advances are mixed research journals, not survey-only. Indian-based outlets and Wren-Research-style journals are excluded per the brief.

# Strictly Survey-Only Journals for Pranjal Rawat's "RL for Economics" Mini-Surveys

## TL;DR
- **Approximately 13 journals genuinely qualify as strictly survey/review-only and are realistic targets** for an RL-for-Economics carve-out: in economics — Journal of Economic Literature (JEL), Journal of Economic Perspectives (JEP), Journal of Economic Surveys (JES), Foundations and Trends in Microeconomics/Econometrics/Finance, World Bank Research Observer; in CS/ML/Stats — ACM Computing Surveys (CSUR), Foundations and Trends in Machine Learning, Foundations and Trends in Optimization, SIAM Review, Statistical Science, Statistics Surveys, Probability Surveys, International Statistical Review, Artificial Intelligence Review. The Annual Reviews family is invitation-only — list it but DO NOT submit cold.
- **Recommended carve-out plan: 4 mini-surveys, not 6.** (1) RL foundations → Foundations and Trends in Machine Learning + JES; (2) Causal RL / off-policy evaluation → Statistical Science + JES; (3) Economic structure in bandits / RLHF → JEP (broad) or Foundations and Trends in Microeconomics (technical); (4) Multi-agent RL & algorithmic collusion → JEP or JES. Themes 5 (robust RL) and 6 (practical limitations) should be folded into the foundations and the multi-agent pieces respectively rather than spun out separately — the field is not yet mature enough for free-standing surveys to land at top venues.
- **Avoid**: cold submissions to any Annual Review (Annual Reviews' verbatim policy: "We do not accept primary research proposals or unsolicited manuscripts, nor do we use outside reviewers in most cases."); Surveys in Operations Research and Management Science (folded into Computers & OR — no longer a stand-alone survey journal); WIREs Computational Statistics (commissioned-only); Synthesis Lectures (book-length series, not journal articles, now run by Springer); Journal of Economic Studies (Emerald — a regular research journal, NOT a survey journal).

## Key Findings

### What "strictly survey-only" actually means in 2026
Three publication models qualify:
1. **Pure-survey journals** with editorial mandates that exclude original research (CSUR, JES, Foundations and Trends series, Statistics Surveys, Probability Surveys, IEEE Communications Surveys & Tutorials, Artificial Intelligence Review, WIREs).
2. **Hybrid review journals** where >90% of published items are surveys/expository pieces and original research is incidental (JEL, JEP, SIAM Review's Survey & Review section, Statistical Science, International Statistical Review, World Bank Research Observer, Bulletin of the AMS).
3. **Invitation-only review series** (Annual Reviews family) — explicitly disallow unsolicited submissions per the publisher's verbatim policy: "We do not accept primary research proposals or unsolicited manuscripts, nor do we use outside reviewers in most cases."

### Editor verification (current as of May 2026)
- **JEL**: David H. Romer (UC Berkeley) is editor; commissioned/peer-reviewed surveys, ~5–10 page outline required first.
- **JEP**: Heidi Williams (Stanford) and Jeffrey Kling (Congressional Budget Office) co-edit; Timothy Taylor managing editor. Heavily solicited; ~10–15% of articles originate as unsolicited proposals (e.g., 7 of 44 published articles in 2022).
- **JES**: Brian Lucey (Trinity College Dublin), Sushanta Mallick (Queen Mary University of London), and Tom Stanley (TU Chemnitz) — all three appointed in early 2024 after the prior board's mass resignation. Open unsolicited submissions via Wiley Research Exchange. **Caveat**: Lucey was the subject of a Retraction Watch investigation (January 2026) over retractions in other Elsevier journals he edited; Wiley confirmed he remains EiC. Author should weigh reputational risk.
- **ACM CSUR**: Co-EICs My T. Thai (University of Florida) and Hanghang Tong (UIUC), term July 1, 2025–June 30, 2028. Unsolicited welcome; double-anonymous review; 8,000–20,000-word target.
- **Foundations and Trends in Machine Learning**: EIC Ryan Tibshirani (UC Berkeley/CMU). Hybrid model — abstract submitted first, then full ~100-page monograph upon preliminary acceptance.
- **Foundations and Trends in Econometrics**: William H. Greene (NYU Stern), founding/continuing EIC.
- **Foundations and Trends in Optimization**: Garud Iyengar (Columbia).
- **Foundations and Trends in Microeconomics**: W. Kip Viscusi (Vanderbilt).
- **Foundations and Trends in Finance**: Sheridan Titman (UT Austin McCombs); editors Josef Zechner (WU Vienna), Chester Spatt (CMU).
- **Statistical Science (IMS)**: Lutz Dümbgen (University of Bern), term 2026–2028; Moulinath Banerjee preceded him 2023–2025; Sonia Petrone preceded that 2020–2022.
- **SIAM Review**: Carola-Bibiane Schönlieb (DAMTP, Cambridge) is editor-in-chief; Survey & Review section accepts unsolicited 30–80-page surveys.
- **Statistics Surveys (IMS-cosponsored: ASA, Bernoulli, IMS, SSC)**: Coordinating Editor Wendy L. Martinez (US Bureau of Labor Statistics). Open submission via EJMS; fully open access; no APC.
- **Probability Surveys**: EIC Adam Jakubowski (Nicolaus Copernicus University), term 2024–2026. Open submission, fully OA, no APC.
- **IEEE Communications Surveys & Tutorials**: EIC Dan Kilper (Trinity College Dublin, CONNECT Centre); Dusit Niyato is past EIC. Out of scope for an RL-for-Economics piece — strictly communications-focused.
- **Artificial Intelligence Review (Springer)**: EIC Derong Liu. Fully open access since January 2024 with mandatory APC.
- **World Bank Research Observer**: Editor Peter Lanjouw (VU Amsterdam). Unsolicited accepted; reviewed twice yearly by editorial board (e.g., spring deadline March 13, 2026).
- **International Statistical Review**: Co-EICs Efstathia Bura (TU Wien) and J. Sunil Rao (University of Miami).
- **Annual Review of Economics**: Co-editors Philippe Aghion, Hélène Rey (LBS), Timothy Besley (LSE). Strictly invitation-only.
- **Annual Review of Financial Economics**: Co-editors Hui Chen (MIT) and Matthew P. Richardson (NYU Stern). Invitation-only.

### Journals to EXCLUDE from the target list
- **Annals of Operations Research, EJOR, JEDC, Computational Economics, Machine Learning, KBS, ESWA, Frontiers in X**: not survey-only (per task constraints).
- **Surveys in Operations Research and Management Science (Elsevier)**: discontinued as a stand-alone journal in 2018; "incorporated into Computers & Operations Research" with a Surveys section edited by Michael Gorman. Submitting there means submitting to a regular OR journal's survey track, not to a survey-only journal.
- **Journal of Economic Studies (Emerald)**: a regular research journal (Q2 SJR 0.519), NOT a survey journal. Do not target.
- **Knowledge Engineering Review (Cambridge)**: Cambridge ceased publication on completion of Volume 40 / 2025; from Volume 41 / 2026, the journal moved to Maximum Academic Press — caveat emptor on prestige and indexing.
- **Synthesis Lectures on AI and Machine Learning**: a book/monograph series (originally Morgan & Claypool, now Springer), not a journal; commissioned by series editors only.
- **Surveys in Differential Geometry**: out of scope (geometry, not RL).
- **Notices of the AMS**: too magazine-style, low technical density, wrong venue.
- **WIREs Computational Statistics**: commissioned-only; review articles are "invited, written, and peer-reviewed by experts." Cold submissions are not the path.
- **International Journal of Economic Perspectives (ijeponline.org)**: a different, lower-prestige journal — not to be confused with JEP (AEA).

### Annotated list of strictly survey-only journals

#### ECONOMICS / FINANCE
| Journal | Publisher | Editor (verified May 2026) | Unsolicited? | OA / APC | Tier |
|---|---|---|---|---|---|
| Journal of Economic Literature | AEA | David H. Romer | Outline-first, then full paper (effectively semi-invited) | Subscription; AEA member access | Top-tier (T1) |
| Journal of Economic Perspectives | AEA | Heidi Williams & Jeffrey Kling | Yes; ~10–15% acceptance from unsolicited proposals | Free public access (no APC) | Top-tier (T1) |
| Journal of Economic Surveys | Wiley | Brian Lucey, Sushanta Mallick, Tom Stanley | Yes — open submission | Hybrid OA option | Strong field journal (T2) |
| Foundations and Trends in Microeconomics | Now/Emerald | W. Kip Viscusi | Yes — abstract first | Subscription | T2 (broadly read in subfields) |
| Foundations and Trends in Econometrics | Now/Emerald | William H. Greene | Yes — abstract first | Subscription | T2 |
| Foundations and Trends in Finance | Now/Emerald | Sheridan Titman | Yes — abstract first | Subscription | T2 in finance |
| World Bank Research Observer | OUP | Peter Lanjouw | Yes — twice-yearly board review | Some OA options | T2 in development econ |
| Annual Review of Economics | Annual Reviews | Aghion, Rey, Besley | **NO — invitation-only** | OA via Subscribe to Open | T1 (but inaccessible) |
| Annual Review of Financial Economics | Annual Reviews | Hui Chen & Matthew Richardson | **NO — invitation-only** | OA via S2O | T1 (but inaccessible) |
| Annual Review of Resource Economics | Annual Reviews | (rotating editors) | **NO — invitation-only** | OA via S2O | T1 (but inaccessible) |

Note: Reviews of Economic Literature (Stanford UP, NEW) and Oxford Research Encyclopedia of Economics & Finance are excluded per the task — author has already engaged both.

#### CS / ML / AI
| Journal | Publisher | Editor (verified May 2026) | Unsolicited? | OA / APC | Tier |
|---|---|---|---|---|---|
| ACM Computing Surveys (CSUR) | ACM | My T. Thai & Hanghang Tong | Yes | As of Jan 1, 2026 ACM is fully OA; subsidized CSUR APC for non-ACM-Open institutions is **$950 (ACM/SIG members) or $1,450 (non-members)** | Top-tier in CS surveys (T1) |
| Foundations and Trends in Machine Learning | Now/Emerald | Ryan Tibshirani | Yes — abstract first | Subscription | T1 in ML monographs |
| Foundations and Trends in Optimization | Now/Emerald | Garud Iyengar | Yes — abstract first | Subscription | T2 |
| Artificial Intelligence Review | Springer | Derong Liu | Yes | **Fully OA since Jan 2024 — APC £2,490 / $3,390 / €2,790** | T2 (broad scope, mixed quality) |
| IEEE Communications Surveys & Tutorials | IEEE ComSoc | Dan Kilper | Yes | Hybrid OA | Top-tier in comms (out of scope for RL-Econ) |

Note: Knowledge Engineering Review (Cambridge → Maximum Academic Press 2026) is in transition; its prestige and indexing are uncertain. Do not target until the new publisher's standing is established.

#### STATISTICS / MATHEMATICS
| Journal | Publisher | Editor (verified May 2026) | Unsolicited? | OA / APC | Tier |
|---|---|---|---|---|---|
| Statistical Science | IMS | Lutz Dümbgen (2026–28) | Yes — review-style mandate | Hybrid; IMS open access option | Top-tier in stats reviews (T1) |
| SIAM Review (Survey & Review section) | SIAM | Carola-Bibiane Schönlieb | Yes — Section 1 (S&R) accepts unsolicited | Hybrid | Top-tier in applied math (T1) |
| Statistics Surveys | IMS et al. | Wendy L. Martinez | Yes | Fully OA, no APC | T2 (rigorous, smaller readership) |
| Probability Surveys | IMS / Bernoulli | Adam Jakubowski (2024–26) | Yes | Fully OA, no APC | T2 (very technical) |
| International Statistical Review | Wiley / ISI | Efstathia Bura & J. Sunil Rao | Yes | Hybrid OA | T2 |
| Bulletin of the AMS | AMS | Alejandro Adem (Chief Editor) | Yes for expository articles; Mathematical Perspectives by invitation only | Hybrid | T1 (but pure-math focus; weak fit) |
| Annual Review of Statistics and Its Application | Annual Reviews | (rotating) | **NO — invitation-only** | OA via S2O | T1 (inaccessible) |
| Annual Review of Control, Robotics, and Autonomous Systems | Annual Reviews | (rotating) | **NO — invitation-only** | OA via S2O | T1 (inaccessible) |

### Theme-to-Journal Mapping

| Theme | Best-fit survey journals (in priority order) | One-line rationale |
|---|---|---|
| **1. RL foundations: dynamic programming → modern RL** | (a) **Foundations and Trends in ML**; (b) **Journal of Economic Surveys**; (c) **SIAM Review (Survey & Review)**; (d) **Statistical Science** | FnT-ML accepts the long-monograph form ideal for tying Bellman/contraction theory to deep RL; JES tolerates technical depth for an econ readership; SIAM Review wants integrative applied-math surveys; Stat Sci wants reviews placed in larger statistical context. |
| **2. Causal RL: off-policy evaluation, confounded MDPs, panel-data dynamic policies** | (a) **Statistical Science**; (b) **Statistics Surveys**; (c) **JES**; (d) **JEL** if framing is broad | Stat Sci/Stat Surveys are the natural homes for off-policy evaluation and identification-style methodology surveys; JES if you keep an econometrics framing with backdoor adjustment / IV instruments; JEL only if written to a broad economist audience without heavy formalism. |
| **3. Economic structure in bandits & RLHF: WARP, McFadden, dynamic pricing** | (a) **JEP** (broad-audience version); (b) **FnT in Microeconomics** (technical); (c) **JES**; (d) **CSUR** (if framed for CS audience) | JEP is the right venue for showing economists how rationality axioms shape RLHF/LLM alignment for general readers; FnT-Micro for the formal monograph on demand systems imposed on bandits; CSUR if the framing is CS-first with economic structure as a tool. |
| **4. Multi-agent RL, equilibrium selection, algorithmic collusion** | (a) **JEP**; (b) **JEL**; (c) **JES**; (d) **CSUR** | Algorithmic collusion is currently an active topic at JEP-style venues — Calvano, Calzolari, Denicolò, and Pastorello, "Artificial Intelligence, Algorithmic Pricing, and Collusion," *American Economic Review* 110, no. 10 (Oct 2020): 3267–97 (DOI: 10.1257/aer.20190623), found that "the algorithms consistently learn to charge supracompetitive prices, without communicating with one another." JEL/JES allow more technical equilibrium-selection material; CSUR if framed as a multi-agent RL survey for CS readers. |
| **5. Robust RL & ambiguity aversion** | (a) **JES**; (b) **FnT in ML**; (c) **Statistical Science**; (d) **SIAM Review** | Better folded into Theme 1 (foundations) than spun out separately — the literature is thinner; if pursued, JES for the econ-decision-theory framing, FnT-ML for technical depth. |
| **6. Practical limitations of RL: brittleness, sample efficiency, lack of guarantees** | (a) **JEP**; (b) **CSUR** | Best framed as an honest critique for a broad audience in JEP, not a stand-alone technical survey — too "negative" in flavor for FnT/Stat Sci. Could also be embedded as the closing section of any of the other carve-outs. |

### Recommended carve-out plan: 4 mini-surveys (not 5–6)

The author's plan to carve out 3–5 papers is right; 6 papers is too many for a single PhD candidate to land simultaneously. **Cut Themes 5 and 6 as stand-alone papers** — robust RL belongs inside the foundations piece, and the practical-limitations critique belongs inside the multi-agent piece.

1. **Mini-survey #1: "From Bellman to Deep RL: A Survey of Reinforcement Learning Foundations for Economists"** (Theme 1, with Theme 5 robust-RL section folded in) → **Foundations and Trends in Machine Learning** (primary). Backup: **Journal of Economic Surveys**. Length 80–120 pages, technical, ML-rigorous. Direct precedent: Moerland, Broekens, Plaat, & Jonker, "Model-based Reinforcement Learning: A Survey," *Foundations and Trends® in Machine Learning* 16, no. 1 (2023): 1–118 (DOI: 10.1561/2200000086) — same length, same scope structure.
2. **Mini-survey #2: "Causal Reinforcement Learning: Off-Policy Evaluation, Instruments, and Dynamic Policies from Observational Data"** (Theme 2) → **Statistical Science** (primary). Backup: **Journal of Economic Surveys**. ~40–60 pages, methodologically rigorous.
3. **Mini-survey #3: "Imposing Economic Structure on Bandits and RLHF"** (Theme 3) → **Journal of Economic Perspectives** (primary, broad-audience version). Backup: **Foundations and Trends in Microeconomics** (technical version, if JEP rejects). ~25–35 pages for JEP, much longer for FnT.
4. **Mini-survey #4: "Multi-Agent RL, Equilibrium Selection, and Algorithmic Collusion"** (Theme 4, with Theme 6 critique folded in) → **Journal of Economic Perspectives** or **Journal of Economic Surveys**. Backup: **ACM Computing Surveys** if framed for CS readers.

**Sequencing**: Submit JES-eligible papers first (fastest review, accepts unsolicited, technical depth tolerated), then submit the longer FnT-ML monograph, then the JEP/JEL pieces last (they require outline-first proposals and have ~80%+ desk-rejection rates on first proposals).

### Predatory or invitation-only flags (do NOT waste time)
- **All Annual Reviews titles** are invitation-only by explicit policy. Listed for completeness; do not submit cold.
- **WIREs Computational Statistics** is commissioned-only. Listed but unrealistic without prior editor contact.
- **Knowledge Engineering Review (2026–)** moved from Cambridge to Maximum Academic Press; prestige and indexing are unproven. Avoid until established.
- **Surveys in Operations Research and Management Science** has been folded into Computers & Operations Research; it is no longer a stand-alone survey-only journal.
- **International Journal of Economic Perspectives** (ijeponline.org) is a different, lower-prestige journal — do not confuse with JEP (AEA).

## Recommendations

### Priority targets (submit first)
1. **Journal of Economic Surveys** — the highest-yield single target. Open submissions, accepts unsolicited, current editorial team (Lucey/Mallick/Stanley) has been actively soliciting AI/ML-econ surveys (recent 2025 issues include Lucchetti & Cajueiro, "Language as Data: A Survey of Natural Language Processing for Economics and Finance," and a forthcoming meta-analysis special issue). Submit Theme 2 (causal RL) and/or Theme 4 (multi-agent RL) here.
2. **Foundations and Trends in Machine Learning** — submit a 2-page abstract for the foundations monograph (Theme 1). FnT-ML has published landmark RL surveys (cf. Moerland et al. 2023 above). Tibshirani's editorship favors statistical-foundations content, which fits the dynamic-programming-to-RL bridge well.
3. **Journal of Economic Perspectives** — submit a 2–5-page proposal for either Theme 3 (rationality axioms in RLHF) or Theme 4 (algorithmic collusion). JEP wants policy-relevant, broad-audience pieces; algorithmic collusion is at the right level of public interest in 2026, building on Calvano et al. 2020 (AER).

### Secondary targets (after first round)
- **Statistical Science** for Theme 2 if the JES route fails — explicitly accepts review-character technical papers.
- **Foundations and Trends in Microeconomics** for a deeply technical version of Theme 3.
- **ACM Computing Surveys** for any theme where the framing tilts CS-first; with ACM's January 2026 fully-OA conversion, the subsidized APC of $950 (ACM/SIG member) or $1,450 (non-member) is modest.

### Benchmarks that would change recommendations
- If JEL/JEP send positive feedback on a proposal → prioritize finishing those papers first; they outweigh JES on impact.
- If author lacks senior co-author signaling → favor JES over JEL/JEP, since proposal acceptance there is heavily reputation-weighted.
- If the field of RLHF moves further toward economic axiomatization in 2026–2027 (e.g., new AER-Insights publications imposing WARP or Slutsky restrictions on bandit algorithms) → strengthen the case for Theme 3 at JEP and accelerate that submission.

### Avoid these mistakes
- Don't submit cold to Annual Reviews — explicit invitation-only policy ("We do not accept primary research proposals or unsolicited manuscripts").
- Don't list Reviews of Economic Literature among target journals (already submitted).
- Don't double-submit; each FnT/AEA/IMS journal has strict simultaneous-submission bans.
- Don't submit Theme 6 (limitations) as a stand-alone paper — it will be perceived as a "negative" or polemic piece without the constructive frame of a foundations or methodology survey.

## Caveats
- **Editor turnover and reputational risk at JES**: Editorial leadership at JES experienced controversy in early 2024 (mass resignation of the prior board) and again in early 2026 (Retraction Watch coverage of co-EiC Brian Lucey, January 2026, regarding 12 retracted papers in other Elsevier journals he edited). The journal remains operational and Wiley confirmed Lucey remains EiC; the author should weigh whether to factor this in. Mallick and Stanley are uncontroversial.
- **Foundations and Trends EIC for ML**: Ryan Tibshirani is reported as EIC by a third-party indexer (S-Logix); the official Emerald page references the editorial board but the EIC name was not directly visible in publicly fetched pages. The author should verify by emailing the publisher before submission.
- **Annual Reviews "Subscribe to Open" model**: even though invitation-only, content from these journals is now freely accessible online (since 2023) — they are no longer paywalled, just gated at the submission stage.
- **APC sticker shock**: AI Review's full-OA conversion (January 2024) means the author needs to budget £2,490 / $3,390 / €2,790 per Springer's official journal page. Statistics Surveys and Probability Surveys, by contrast, are no-APC OA. WBRO charges OA fees only if the author opts in. ACM CSUR's January 2026 fully-OA pricing ($950/$1,450 subsidized) is the most economical T1 venue.
- **The "survey journal" label is fuzzy**: SIAM Review and Statistical Science occasionally publish what amount to short technical contributions, but their editorial mandates and >90% of published content fit the survey-only criterion. Bulletin of the AMS publishes only expository articles (no original research), but its readership is pure mathematicians — weak fit for an RL-Econ piece.
- **Length mismatch**: a 138-page survey will need to be cut substantially for JEP (~25 pages target) and JEL (~40–60 pages typical), but can be expanded for FnT-ML (~100 pages typical, e.g., Moerland et al. 2023 was 118 pages). Plan thematic carve-outs around the venue's length norms, not vice versa.