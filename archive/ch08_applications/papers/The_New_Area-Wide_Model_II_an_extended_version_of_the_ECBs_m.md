Günter Coenen, Peter Karadi, Sebastian Schmidt, Anders Warne

No 2200 / November 2018

<!-- image -->

## Working Paper Series

The New Area-Wide Model II: an extended version of the ECB's micro-founded model for forecasting and policy analysis with a financial sector

Revised December 2019

<!-- image -->

Disclaimer: This paper should not be reported as representing the views of the European Central Bank (ECB). The views expressed are those of the authors and do not necessarily reflect those of the ECB.

## Abstract

This paper provides a detailed description of an extended version of the ECB's New Area-Wide Model (NAWM) of the euro area (cf. Christoffel, Coenen, and Warne 2008). The extended model-called NAWM II-incorporates a rich financial sector with the threefold aim of (i) accounting for a genuine role of financial frictions in the propagation of economic shocks and policies and for the presence of shocks originating in the financial sector itself, (ii) capturing the prominent role of bank lending rates and the gradual interest-rate pass-through in the transmission of monetary policy in the euro area, and (iii) providing a structural framework useable for assessing the macroeconomic impact of the ECB's large-scale asset purchases conducted in recent years. In addition, NAWM II includes a number of other extensions of the original model reflecting its practical uses in the policy process over the past ten years.

## JEL Classification System: C11, C52, E30, E37, E58

Keywords: DSGE modelling, Bayesian inference, financial frictions, forecasting, policy analysis, euro area

## Non-technical summary

In recent years, the ECB's standard monetary policy operations have been complemented by several non-standard measures (NSMs) which have responded to the challenges posed by the different phases of the financial crisis that had begun in 2007. Asset price reactions suggest that these NSMs had expansionary effects but the quantitative impact on other macroeconomic variables remains uncertain. Hence there is a compelling need to aid the analysis of the quantitative effects of NSMs by developing coherent structural macroeconomic modelling frameworks . Yet standard DSGE models are silent on the pertinent transmission channels of NSMs.

Against this background, this paper provides a detailed description of an extended version of the ECB's New Area-Wide Model (NAWM). In its original version (cf. Christoffel, Coenen, and Warne 2008), the NAWM features only stylised financial frictions in the form of exogenous risk premium shocks but no endogenous financial propagation mechanisms. In contrast, the extended version of the NAWM-called NAWM II-includes a rich financial sector which is centered around two distinct types of financial intermediaries that are exposed to sector-specific shocks: (i) funding-constrained 'wholesale banks' ` a la Gertler and Karadi (2011) which engage in maturity transformation and originate long-term loans, and (ii) 'retail banks' ` a la Gerali, Neri, Sessa, and Signoretti (2010) which distribute these loans to the non-financial private sector and adjust the interest rates on loans only sluggishly. The long-term loans are required by the non-financial private sector to finance capital investments as in Carlstrom, Fuerst, and Paustian (2017). Furthermore, building on Gertler and Karadi (2013), NAWM II includes a set of no-arbitrage and optimality conditions which govern the holdings of domestic and foreign long-term government bonds by the financial and the non-financial private sector, respectively. These conditions give rise to an explicit exchange-rate channel of private and public-sector asset purchases.

The incorporation of these financial extensions into the original model reflects the threefold aim pursued in the development of NAWM II, namely: (i) to account for a genuine role of financial frictions in the propagation of economic shocks and policies and for the presence of shocks originating in the financial sector itself, (ii) to capture the prominent role of bank lending rates and the gradual interest-rate pass-through in the transmission of monetary policy in the euro area, and (iii) to provide a structural framework useable for assessing the macroeconomic impact of the ECB's large-scale asset purchases which have been conducted in recent years, along with other NSMs.

In addition, NAWM II features a number of other extensions of the original model reflecting its practical uses in the policy process at the ECB over the past ten years, including an endogenous mechanism concerning private-sector agents' perceptions of the central bank's inflation objective which permits to capture possible fluctuations in longer-term private-sector inflation expectations.

## 1 Introduction

In recent years, the ECB's standard monetary policy operations have been complemented by several non-standard measures (NSMs) which have responded to the challenges posed by the different phases of the financial crisis that had begun in 2007. Asset price reactions suggest that these NSMs had expansionary effects (see ECB 2017 and the references therein) but the quantitative impact on other macroeconomic variables remains uncertain. Hence there is a compelling need to aid the analysis of the quantitative effects of NSMs by developing coherent structural macroeconomic modelling frameworks . Yet standard DSGE models are silent on the pertinent transmission channels of NSMs.

A new class of DSGE models has extended the standard framework with financial market imperfections, regularly referred to as financial frictions. In an environment with financial frictions, the balance sheet conditions of borrowers and/or lenders influence the conditions through which credit is granted, and credit conditions affect macroeconomic outcomes. Furthermore, in a two-way interaction, the macroeconomic outlook feeds back to the balance sheet conditions through its impact on asset prices. Consequently, the introduction of financial frictions has modified the transmission of standard economic shocks and policies (Kiyotaki and Moore 1997; Bernanke, Gertler, and Gilchrist 1999; Iacoviello 2005), enabled the analysis to account for fluctuations of a financial nature (Jermann and Quadrini 2012; Christiano, Motto, and Rostagno 2014), and could explain the effectiveness of NSMs through distinct mechanisms. NSMs, like the large-scale asset purchase programmes conducted by major central banks, were argued to improve economy-wide credit conditions by circumventing and relieving the stressed financial intermediary sector (C´ urdia and Woodford 2011; Gertler and Karadi 2011), injecting liquidity to the financial system (Del Negro, Eggertsson, Ferrero, and Kiyotaki 2017), or creating asset scarcity in relevant market segments (Chen, C´ urdia, and Ferrero 2012). 1 Calibration and estimation of the relevance of financial frictions with pre-NSM data allowed using these frameworks to assess the likely macroeconomic impact of NSM policies.

Against this background, this paper provides a detailed description of an extended version of the ECB's New Area-Wide Model (NAWM). In its original version (cf. Christoffel, Coenen, and Warne 2008), the NAWM features only stylised financial frictions in the form of exogenous risk premium shocks but no endogenous financial propagation mechanisms. In contrast, the extended version of the NAWM-hereafter called NAWM II-includes a rich

1 Additional research (Bhattarai, Eggertsson, and Gafarov 2015) has argued that asset purchase policies exerted their influence through signalling the central bank's commitment to maintain low interest rates in the future.

financial sector which is centered around two distinct types of financial intermediaries that are exposed to sector-specific shocks: (i) funding-constrained 'wholesale banks' ` a la Gertler and Karadi (2011) which engage in maturity transformation and originate long-term loans, and (ii) 'retail banks' ` a la Gerali, Neri, Sessa, and Signoretti (2010) which distribute these loans to the non-financial private sector and adjust the interest rates on loans only sluggishly. The long-term loans are required by the non-financial private sector to finance capital investments as in Carlstrom, Fuerst, and Paustian (2017). Furthermore, NAWM II includes a set of no-arbitrage and optimality conditions which govern the holdings of domestic and foreign long-term government bonds by the financial and the non-financial private sector, respectively, building on Gertler and Karadi (2013).

The incorporation of these financial extensions into the original model reflects the threefold aim pursued in the development of NAWM II, namely: (i) to account for a genuine role of financial frictions in the propagation of economic shocks and policies and for the presence of shocks originating in the financial sector itself, (ii) to capture the prominent role of bank lending rates and the gradual interest-rate pass-through in the transmission of monetary policy in the euro area, and (iii) to provide a structural framework useable for assessing the macroeconomic impact of the ECB's large-scale asset purchases which have been conducted in recent years, along with other NSMs. In addition, NAWM II features a number of other extensions of the original model reflecting its practical uses for forecasting and policy analysis since its integration into the ECB policy process about ten years ago, including an endogenous mechanism concerning private-sector agents' perceptions of the central bank's inflation objective which permits to capture possible fluctuations in longerterm private-sector inflation expectations.

In estimating NAWM II, we follow the standard Bayesian approach outlined in An and Schorfheide (2007), which was likewise adopted by Christoffel, Coenen, and Warne (2008)-henceforth referred to as CCW-for estimating the original version of the NAWM. In doing so, we retain the 18 macro variables that were used as observed variables for estimating the original model, albeit with two important changes: all real variables have been transformed into per-capita units, and the estimation sample has been extended until 2014Q4. In addition, six variables have been added to the original set of observables in order to provide useful measurements for identifying the model's financial-sector parameters and shocks, to inform the estimation of the perceived inflation objective as well as trend productivity growth, and, last but not least, to establish a link with traditional measures of the euro area output gap. To allow for prior distributions with dependencies among the parameters and, at the same time, to avoid outcomes which assign too much mass to

unrealistically large population standard deviations for the observed variables, we use a socalled system prior , originally proposed by Del Negro and Schorfheide (2008) and further developed by Andrle and Beneˇ s (2013). Apart from guiding the prior distributions of the population standard deviations of a subset of observables, our system prior also exerts a direct influence on the slope coefficient of the domestic price Phillips curve, which is a key reduced-form parameter of the model.

The properties of NAWM II are inspected along multiple dimensions, including by means of impulse-response analysis, by conducting historical as well as forecast-error-variance decompositions, and by providing an evaluation of its relative forecasting performance. While the presence of a financial sector in NAWM II enhances the transmission mechanism underlying the propagation of its structural shocks when compared to the original NAWM, both models feature broadly similar impulse responses to important common shocks, such as a standard monetary policy shock. The newly introduced financial shocks, in turn, provide a more comprehensive picture of how shocks originating in the financial sector are transmitted to the macro economy through the reaction of various asset prices. These shocks are found to play a non-trivial role in the evolution of economic fluctuations in the run-up to, during and in the aftermath of the financial crisis of 2008/09.

NAWM II also expands the range of potential applications for model-based policy analysis, as illustrated by means of two examples. The first application uses the model to conduct counterfactual simulations of the ECB's large-scale asset purchases, one of the main objectives for the development of NAWM II. The second application explores the potential consequences of a de-anchoring of longer-term inflation expectations. With regard to the effects of asset purchases, we find that the model gives rise to bond-yield reactions that are contained, but consistent with the empirical findings of several event studies. Also, the macroeconomic effects, as reflected in the responses of real GDP growth and inflation, are broadly in line with, albeit slightly smaller than, other model-based assessments. A key insight from the counterfactual simulations is that the novel international dimension of the financial frictions in NAWM II, which gives rise to an exchange-rate channel of asset purchases, meaningfully influences the impact of this non-standard instrument.

The remainder of the paper is organised as follows. Section 2 outlines the specification of NAWM II with a focus on the financial-sector extension, and Section 3 reports on the estimation of the model using the Bayesian approach. Section 4 inspects the model properties, while Section 5 presents illustrative applications. Section 6 concludes and discusses directions for future research. An appendix provides details on the log-linear version of the model, including the computation of its steady state.

## 2 The model

In this section we outline the specification of NAWM II. As the model extends the original version of the NAWM with a rich financial intermediary sector but otherwise retains its basic structure, we just provide a non-technical sketch of the original model and present subsequently the new elements, as well as the new equations, that are most relevant for understanding the enhanced role of the financial sector in the extended model. In addition, we briefly describe some other extensions of the original NAWM which reflect its practical uses for forecasting and policy analysis over the past years and which have been incorporated into NAWM II as well. 2

## 2.1 The original model: A bird's eye view

The original NAWM is an open-economy DSGE model of the euro area designed for use in the (Broad) Macroeconomic Projection Exercises regularly undertaken by ECB/Eurosystem staff and for analysis of topical policy issues; see CCW for a detailed description of the model's structure. Its development has been guided by the principal consideration of covering a comprehensive set of core projection variables, including a small number of foreign variables, which, in the form of exogenous assumptions, play an important role in the preparation of the staff projections. 3

The NAWM features four types of economic agents: households, firms, a fiscal authority and the central bank. Households make optimal choices regarding their purchases of consumption and investment goods, the latter determining the economy-wide capital stock. They supply differentiated labour services in monopolistically competitive markets, they set wages as a mark-up over the marginal rate of substitution between consumption and leisure, and they trade in domestic and foreign (short-term) bonds.

As regards firms, the NAWM distinguishes between domestic producers of tradable intermediate goods and domestic producers of three types of non-tradable final goods: a private consumption good, a private investment good, and a public consumption good. The intermediate-good firms use labour and capital services as inputs to produce differentiated goods, which are sold in monopolistically competitive markets domestically and abroad.

2 Another major extension, which is not considered further in this paper, concerns the development of a rich specification of the NAWM's fiscal sector. This extension was triggered by the need to analyse the large-scale response of fiscal policy to the financial crisis in the euro area. For details, see Coenen, Straub, and Trabandt (2012, 2013).

3 For an overview of the ECB/Eurosystem staff macroeconomic projection exercises and a description of the techniques, models and tools used therein, see ECB (2016).

In doing so, they set different prices for domestic and foreign markets as a mark-up over their marginal costs. The final-good firms combine domestic and foreign intermediate goods in different proportions, acting as price takers in fully competitive markets. The foreign intermediate goods are imported from producers abroad, who set their prices in euro in monopolistically competitive markets, allowing for a gradual exchange-rate pass-through. A foreign retail firm in turn combines the exported domestic intermediate goods, with aggregate export demand depending on total foreign demand.

Both households and firms face a number of nominal and real frictions, which have been identified as important in generating empirically plausible dynamics. Real frictions are introduced via external habit formation in consumption, through generalised adjustment costs in investment, imports and exports, and through fixed costs in intermediate-good production. Nominal frictions arise from staggered price and wage-setting ` a la Calvo (1983), in combination with (partial) dynamic indexation of price and wage contracts to past inflation. In addition, there already exist some stylised financial frictions which however enter the model only in the form of exogenous risk premia.

The fiscal authority purchases the public consumption good, issues domestic bonds, and levies different types of distortionary taxes, albeit at constant rates. Nevertheless, Ricardian equivalence holds because of the simplifying assumption that the fiscal authority's budget is balanced each period by means of lump-sum taxes. The central bank sets the shortterm nominal interest rate according to a Taylor (1993)-type interest-rate rule, stabilising inflation in line with the ECB's definition of price stability.

The NAWM is closed by a rest-of-the-world block, which is represented by a structural vector-autoregressive (SVAR) model determining five foreign variables: foreign demand, foreign prices, the foreign interest rate, foreign competitors' export prices and the price of oil. The SVAR model does not feature spill-overs from the euro area, in line with the treatment of the foreign variables as exogenous assumptions in the projections.

## 2.2 The extended model with financial intermediaries

The financial-sector extension of the NAWM modifies the households' decision problem, introduces three new types of agents and five new (nominal) assets. The new assets are deposits, long-term retail and wholesale loans, and domestic and foreign long-term government bonds. The households face a financing constraint for investments in physical capital, which gives rise to a demand for long-term loans, and they can invest in domestic longterm government bonds subject to portfolio adjustment frictions. The investment loans are

supplied by two out of the three new types of agents, which act as financial intermediaries. First, wholesale banks use the deposits raised from the households and their net worth to issue wholesale loans and to purchase domestic and foreign long-term bonds. Second, retail banks purchase the wholesale loans, transform them into retail loans and offer these retail loans to the households. Because of the financing constraint on the part of the households, the production of new physical capital is now separated from the households' decision problem and relegated to the third of the three new types of agents, namely capital producers . In the following, we describe the modified decision problem of the households and the decision problems of the three new agents one by one. They give rise to a set of modified and new model equations which we state below.

In terms of notation, the extended model with financial intermediaries directly builds on the original model. That is, details on the variables and parameters that are common to both the extended and the original model can be found in CCW to the extent that they are not separately provided in this paper.

## 2.2.1 Households

We first describe the new assets households can invest in, then we set up the households' modified decision problem and detail their choice of allocations.

## Assets

Households can acquire long-term government bonds and obtain retail investment loans.

- Long-term nominal government bonds, B L,t +1 , are modeled as perpetuities with geometrically decaying coupons as in Woodford (2001). In particular, the asset pays 1, /rho1 L , /rho1 2 L , . . . units of the num´ eraire in consecutive periods, where /rho1 L ∈ ( 0 , 1) is the rate of decay. We denote their discount price as Q L,t . Its (ex post) nominal rate of return can be expressed as

<!-- formula-not-decoded -->

The assumption of geometric decay facilitates the aggregation of bonds issued in different periods: given the time pattern of repayment, loans issued in period t -p have the same value as /rho1 p L assets issued in period t . Relatedly, the amount of outstanding bonds decays with the rate /rho1 L , and the new issuance equals CF L,t = B L,t +1 -/rho1 L B L,t .

- We introduce long-term nominal retail investment loans in an analogous way. The

loans, B I,r,t +1 , are raised from a continuum of retail banks r ∈ [ 0 , 1]. An investment loan generates an upfront payoff to the household in the amount Q I,r,t for each unit of the loan and requires a geometrically decaying coupon repayment from the household with a rate of decay /rho1 I ∈ ( 0 , 1). Consequently, these assets represent liabilities of the households, whereas long-term government bonds are assets. The aggregate amount of new loans issued in period t equals CF I,t = B I,t +1 -/rho1 I B I,t .

In addition, households hold short-term (one-period) government bonds, B t +1 , which are purchased with a discount in period t and repayed at par in period t +1, as well as interestbearing nominal deposits D t +1 with the wholesale banks. Short-term government bonds and deposits are both nominally riskless, and they are perfect substitutes. In equilibrium they earn the same interest, which we denote as R D,t = /epsilon1 RP t R t . As in the original model, /epsilon1 RP t represents a serially correlated risk premium shock which drives a wedge between the return on the household's deposits with the wholesale banks and, in equal measure, between the short-term government bond yield and the short-term nominal interest rate R t controlled by the central bank. In economic terms, this shock can also be interpreted as a liquidity premium reflecting the distinct liquidity services provided by central bank assets versus government bonds and wholesale bank deposits.

## Decision problem

An individual household h ∈ [ 0 , 1] maximises its expected discounted lifetime utility

<!-- formula-not-decoded -->

subject to three constraints. The first is the period-by-period budget constraint, which is modified relative to that in the original NAWM:

<!-- formula-not-decoded -->

We describe the differences of this constraint relative to the original model. First, house-

hold h purchases investment goods in quantity I h,t at a relative price ˜ p I,t from the capital producers. Second, it places deposits in the amount of D h,t +1 to the wholesale banks and receives a (gross) return R D,t -1 on previous-period deposits. Third, it purchases new longterm domestic government bonds in quantity B L,h,t +1 -/rho1 I B L,h,t at the market price Q L,t and receives a coupon payment equal to unity on each outstanding unit of government bonds B L,h,t . As in Gertler and Karadi (2013), we assume that the household faces a cost when adjusting its portfolio holdings of these bonds. The cost function takes the form

<!-- formula-not-decoded -->

so that the household can hold a fixed quantity B L,h of bonds costlessly, but needs to pay a quadratic cost to raise its holdings above this value. The parameter γ h L determines the sensitivity of the cost to the size of the adjustment.

And fourth, the household raises a basket of differentiated investment loans supplied by the continuum of retail banks r , which operate in monopolistically competitive markets as in Gerali et al. (2010). 4 The basket of new loans, CF I,h,t , is a CES aggregate of individual retail loans:

<!-- formula-not-decoded -->

where the possibly time-varying parameter ϕ I t &lt; 1 is inversely related to the elasticity of substitution between the differentiated loans. 5

Cost minimisation by households implies a demand function for individual retail bank loans:

<!-- formula-not-decoded -->

where Q I,r,t is the price of a new loan from retail bank r , while the aggregate loan-price index is given by

<!-- formula-not-decoded -->

As can be seen from the budget constraint (3), household h raises an amount of Q I,t CF I,h,t = ∫ 1 0 Q I,r,t CF I,h,r,t dr of investment loans in period t and repays a coupon

4 Our approach extends the sticky-loan-rate assumption of Gerali et al. (2010), who consider one-period loans, to long-term loans.

5 As detailed in Section 2.2.4, the parameter ϕ I t determines the mark-down on loan prices that monopolistically competitive retail banks charge in the process of intermediating wholesale loans to the households.

of unity for its outstanding loans B I,h,t . 6

Finally, it should be noted that the short-term foreign bond holdings have disappeared from the household's budget constraint because we assume now that the trade balance is financed by long-term foreign government bonds, which are intermediated by the wholesale banks; see Section 2.2.3. Also the term D h,t in the budget constraint now denotes the totality of profits from different sources which are assumed to be transferred to the household in a lump-sum fashion.

The second constraint the household faces is a capital accumulation equation: 7

<!-- formula-not-decoded -->

where /epsilon1 I t represents a serially correlated investment-specific technology shock.

The third constraint of the household is a Loan-in-Advance (LIA) constraint on financing its investment expenditures as in Carlstrom, Fuerst, and Paustian (2017):

<!-- formula-not-decoded -->

According to this constraint, household h needs to finance the costs of purchasing investment goods, expressed in nominal terms, by raising new retail loans in quantity CF I,h,t = B I,h,t +1 -/rho1 I B I,h,t with unit payoffs Q I,t . The constraint establishes a special role for retail bank loans: they are the only source of funding for new investment projects. Furthermore, the constraint exposes investment to the evolution of the retail loan price Q I,t , which is determined in general equilibrium.

## Choice of allocations

The household maximises its lifetime utility (2) subject to its budget constraint (3), the

6 The basket of investment loans B I,h,t +1 requires a periodic repayment ˜ B I,h,t +1 , which is only equal to B I,h,t +1 up to a first-order approximation. We abstract from this distinction and, instead, impose the equality of the two quantities, based on the understanding that we only conduct first-order analysis. More specifically, the repayment evolves as ˜ B I,h,t +1 = /rho1 I ˜ B I,h,t +1 + ∫ 1 0 CF I,h,r,t dr , while the evolution of the basket of loans is described in equation (5). The two variables are different because the repayment is derived from the simple sum of retail-bank-level loans (unit repayment for each unit of loan from each retail bank), while the measure for the basket of investment loans is derived from the CES aggregate of loans across retail banks. However, the two variables are equal up to a first-order approximation around a non-stochastic steady state, which, as in our case, has no dispersion in retail loan prices.

7 The investment adjustment cost, which is part of the household's problem in the original model, is now paid by the capital producers; see Section 2.2.2. This modification has no impact on the investment choice, ceteris paribus, but simplifies the investment problem in the presence of financial frictions. For example, the log-linearised capital accumulation equation (abstracting from growth), ̂ K h,t +1 = (1 -δ ) ̂ K h,t + δ ( ̂ /epsilon1 I t + ̂ I h,t ), is equivalent to the log-linear capital accumulation equation in the original model because the investment adjustment cost and its first derivative are both zero in steady state.

capital accumulation equation (8), and the LIA constraint (9). As in the original NAWM, we define Λ h,t /P C,t and Λ h,t Q h,t to be the Lagrange multipliers associated with the budget constraint (3) and the capital accumulation equation (8), respectively. Consequently, Λ h,t represents the shadow value of a unit of the consumption good and Q h,t the shadow value of a unit of the investment good; that is, Tobin's Q . In addition, let ς h,t Λ h,t /P C,t be the Lagrange multiplier associated with the LIA constraint (9). This multiplier represents the shadow value of retail loans and measures the utility costs of financial frictions.

The first-order condition with respect to investment is

<!-- formula-not-decoded -->

As the left-hand side of the equation shows, the marginal cost of investment is now not given by the relative price ˜ p I,t of purchasing investment goods from the capital producers alone, but it includes the additional cost caused by the extra retail loans the household needs to raise to finance the purchase. 8 The gains from an extra investment unit, indicated on the right-hand side, are equal to the shadow value of investment Q h,t scaled by the investment-specific technology shock /epsilon1 I t .

The first-order condition with respect to the basket of retail investment loans is

<!-- formula-not-decoded -->

According to this equation, the periodt gains from an extra retail loan unit, which raises Q I,t for each unit of the new loan and eases the household's LIA constraint by the same amount, must be equal to the discounted present value of the future costs of this loan. The stochastic nominal discount factor is β Λ h,t +1 / Λ h,t P C,t /P C,t +1 and the future cost is the sum of the unit coupon the household needs to repay to the retail bank in the next period and the next-period value of the loan. The loan quantity decays with parameter /rho1 I , and the next-period value of the loan is in turn determined by the next-period replacement value of the loan Q I,t +1 and the tightness of the LIA constraint in the next period, ς h,t +1 .

The household's demand for long-term government bonds comes from the consolidated first-order conditions with respect to short and long-term government bond holdings:

<!-- formula-not-decoded -->

8 Compared to the original model, ˜ p I,t replaces the relative price p I,t = P I,t /P C,t in the first-order conditions with respect to capital K h,t +1 and capital utilisation u h,t , which are not separately stated here.

That is, the demand of household h for long-term government bonds depends on the expected present value of the excess bond return relative to the short-term government bond yield (which is equal to the deposit rate), R L,t +1 -R D,t . The household would demand a fixed amount B L,h of government bonds if this premium were zero, and otherwise, when the premium increases, it raises its demand with an elasticity equal to the inverse of the portfolio adjustment cost parameter γ h L . As this parameter declines towards zero, the demand elasticity increases and the household responds ever more flexibly to a change in the premium. Accordingly, the adjustment cost is a crucial determinant of the overall level of financial frictions in the model. If the adjustment cost were zero, the households' behaviour would neutralise the impact of financial frictions. More concretely, when the wholesale banks' balance sheet constraint tightens, the premium of long-term government bonds rises, as we show below. If the adjustment cost were zero, the households would react to this rise in the premium by flexibly purchasing bonds from the wholesale banks. This would immediately ease the banks' balance sheet constraint and offset any transmission of their balance sheet problems to the economy.

## 2.2.2 Capital producers

The problem of the capital producers is standard. They face investment adjustment costs: in particular, in order to produce investment goods of a quantity I t , they need inputs in quantity (1 + Γ I ( I t /I t -1 )) I t . They purchase their investment inputs at the relative price P I,t /P C,t , and sell their investment output at the relative price ˜ p I,t . They discount future income using the stochastic discount factor of the households, who are their owners. Therefore, they maximise

<!-- formula-not-decoded -->

The first-order condition for the capital producers' maximisation problem with respect to the investment good is:

<!-- formula-not-decoded -->

The first-order condition equates the relative selling price of the marginal unit of the

investment good to its marginal cost. The marginal cost depends on the relative price of the investment inputs and the change in the adjustment costs this period and in the next brought about by the production of the marginal unit of the investment good. The profits of the capital producers are redistributed to the households in a lump-sum fashion.

## 2.2.3 Wholesale banks

We introduce wholesale banks into the model, which face an agency problem as in Gertler and Karadi (2011). Accordingly, each household h is composed of a measure f of wholesale bankers and a measure (1 -f ) of workers. New wholesale bankers obtain a fixed start-up fund from the household, which forms the basis of their net worth. They invest their net worth and deposits, which they collect from other households, into investment loans and government bonds, and build their net worth over time from their retained earnings. An agency friction limits the wholesale banks' leverage, and, therefore, the amount of deposits they can collect. In each period, wholesale bankers become workers with time-varying probability 1 -θ t and return their net worth to the household they are a member of. This setup ensures finite expected lifetime for wholesale bankers and thereby limits their ability to free themselves from their binding leverage constraints. To keep the share of the bankers constant, each period f (1 -θ t ) workers become wholesale bankers. Wholesale bankers and workers enjoy perfect consumption risk sharing.

## Assets

The wholesale banks are indexed by b ∈ [ 0 , 1] and hold three types of long-term nominal assets: wholesale investment loans and domestic as well as foreign long-term government bonds.

- The wholesale banks issue nominal long-term investment loans B I,b,t +1 at wholesale price ˜ Q I,t . The wholesale investment loans follow the same repayment schedule as the retail loans, with a repayment of unity in the first period, and a geometrically decaying coupon with parameter /rho1 I . The nominal rate of return of the wholesale investment loan is

<!-- formula-not-decoded -->

- They purchase domestic and foreign nominal long-term government bonds, B L,t +1 and B ∗ L,t +1 . Analogously to the domestic long-term government bonds that we introduced in the previous section, foreign government bonds (denominated in foreign currency) have a geometrically decaying coupon with a rate of decay /rho1 L ∗ and new issuance equals

CF ∗ L,t = B ∗ L,t +1 -/rho1 ∗ L B ∗ L,t . The price of the bond in foreign currency is denoted by Q ∗ L,t , and the rate of return of the foreign long-term government bond denominated in foreign currency is

## Decision problem

The nominal balance sheet of a wholesale bank b (at the end of period t ) is

<!-- formula-not-decoded -->

where the superscript p indicates privately intermediated assets. 9

The balance sheet identity imposes the equality of the value of assets, including investment loans and domestic as well as foreign bonds, with the bank's liabilities, which are its nominal net worth NW b,t and the households' nominal deposits with the bank, D b,t +1 .

The net worth of bank b accumulates through retained earnings:

<!-- formula-not-decoded -->

As can be seen from the first equality in this accumulation equation, the bank obtains revenue from the returns on its asset holdings, and it pays interest on its deposits. The return on the foreign currency holdings in local currency is S t +1 /S t R ∗ L,t +1 . Combining this equation with the balance sheet identity (17), we can express the law of motion for banklevel net worth as a function of excess asset returns over the deposit rate and asset-level leverage (asset-to-net-worth) ratios, as shown by the second equality. The net worth of banks, if uninvested, would earn the deposit rate R D,t (which is equal to the short-term government bond yield and is shown as the last term on the right-hand side), but they earn

9 The wholesale banks hold foreign currency assets and finance them from domestic currency deposits. This formulation is different from alternative setups like the one in Aoki, Benigno, and Kiyotaki (2016), which were developed to describe emerging market economies and in which domestic assets are financed by foreign currency deposits. The difference is motivated by the observation that the euro area maintains a positive net-foreign-asset position.

<!-- formula-not-decoded -->

leveraged excess returns when invested in risky assets.

The wholesale banker b maximises its discounted net worth upon exiting the wholesale market, which happens with probability 1 -θ t +1 . As the bank can earn excess returns on its net worth, it is optimal to continue to invest it and to postpone all net worth payouts to the household till the bank exits. Accordingly, the value function associated with the banker's decision problem is

<!-- formula-not-decoded -->

with Λ t,t + k = β Λ h,t + k / Λ h,t and Π C,t + k = P C,t + k /P C,t .

We assume that each wholesale bank b faces an agency problem: in particular, it can abscond with a share Ψ of the assets. If this happens, the bank forgoes its franchise value and its depositors are left with the remaining (1 -Ψ) share of the assets. To guarantee that such event never takes place, the depositors impose an incentive compatibility constraint on the bank:

<!-- formula-not-decoded -->

Under this constraint, no bank will ever abscond in equilibrium because the franchise value never gets below the gains from absconding. The bank's value function is maximised subject to this constraint, as well as the net worth accumulation equation (18).

The incentive compatibility constraint also shows that the absconding rate varies across assets: we assume that the bank can abscond with share Ψ of the investment loans, with share ω L Ψ of the domestic and with share ω ∗ L,t Ψ of the foreign government bonds, where ω L , ω ∗ L,t &lt; 1. The idea here is that the domestic and foreign governments have superior ability relative to the private sector to recover the assets. The relative absconding rate for foreign government bonds is assumed to depend on the country's net-foreign-asset position: the higher the foreign bond holdings, the harder it is to recover a sizeable share of the assets. In particular, the relative absconding rate for the foreign bonds takes the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where s B ∗ L ,t +1 = S t Q ∗ L,t B ∗ L,t +1 /P Y,t Y t is the share of the country's aggregate foreign bond holdings in domestic output and /epsilon1 RP ∗ t represents a serially correlated risk premium shock.

We estimate both the relative absconding rate parameters ω L , ˜ ω ∗ L , and the parameter γ ∗ L determining the elasticity of the relative absconding rate for foreign bonds with respect to the country's net-foreign-asset position.

## Portfolio allocation

To solve the banker's decision problem, we postulate, and later verify, that the banker's value function is linear in net worth and symmetric across banks; that is, V b,t ( NW b,t ) = v t NW b,t .

The bank's decision problem leads to three first-order conditions with respect to its assets, an envelope condition with respect to its net worth and the incentive compatibility constraint:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, the bank's stochastic discount factor is modified with the factor Ω t +1 = (1 -θ t +1 ) + θ t +1 v t +1 , which measures the value of an extra unit of net worth next period: it is worth a value of unity in case of exit (with probability 1 -θ t +1 ), and it is worth v t +1 otherwise. The bank's discount factor Λ t,t +1 Ω t +1 Π -1 C,t +1 is different from the households' discount factor Λ t,t +1 Π -1 C,t +1 , because a payoff for the bank in the presence of a binding leverage constraint is more valuable than the same payoff for the household; that is, v t +1 &gt; 1.

The first three equations equalise the excess returns of marginal investments into the individual assets to the marginal costs accruing through their impact on the incentive constraint. The excess returns measure the gains from holding the assets relative to their costs, which include payouts on the deposits and foregone returns on net worth that were used

to finance them. The marginal costs of extra investments are influenced by the absconding rates Ψ, ω L Ψ and ω ∗ L,t Ψ for the individual assets and the tightness of the incentive constraint. The latter is measured by µ t = ˜ µ t / (1 + ˜ µ t ), where ˜ µ t is the Lagrange multiplier of the incentive constraint. We assume that the incentive constraint always binds. Under the calibration used later in the paper, the constraint binds in the model's non-stochastic steady state, so for small enough shocks the assumption is indeed satisfied.

Combining the first-order conditions (22) to (24), we can obtain two no-arbitrage relationships between the expected excess returns of investment loans and domestic government bonds, and between domestic and foreign government bonds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

These no-arbitrage conditions imply that the expected excess returns of the bank's assets weighted by the inverse of the relative absconding rates must be equal. The relative absconding rates enter these conditions because they determine the relative impact of the assets on the bank's incentive constraint. This reflects the fact that the relative absconding rates influence the market-determined 'haircuts' of the assets, i.e. how easy it is to raise outside funding to purchase them.

The dynamics of the exchange rate are governed by the second no-arbitrage condition, instead of the standard uncovered interest parity (UIP) relationship. According to this no-arbitrage condition, expected exchange-rate revaluations need to equalise expected discounted excess returns of domestic over foreign long-term government bonds. 10 This interaction between excess returns and the exchange rate opens the door to a direct exchange-rate channel of central bank asset purchase policies: to the extent that these policies improve the balance sheet conditions of banks, they will depreciate the domestic currency. More precisely, because asset purchases ease banks' balance sheet constraint, the excess return on domestic government bonds declines. As the foreign currency return R ∗ L,t +1 on foreign bonds is determined internationally and remains unaffected by domestic developments, the exchange rate needs to depreciate so that its expected appreciation reduces the domestic

10 It is straightforward to show that, without financial frictions, this condition simplifies to the UIP condition.

currency return on foreign bonds.

It remains to determine the value of v t . To this end, we define the bank's 'risk-weighted' leverage Φ t as

<!-- formula-not-decoded -->

where the relative absconding rates determine the relative risk weights, reflecting the fact that the bank can raise more deposits for a unit of net worth to purchase government bonds compared to the issuance of investment loans. Therefore, government bonds can be downweighted in an effective leverage measure.

This definition of leverage is particularly useful since, evoking the incentive constraint (20), bank leverage can be expressed as Φ t = v t / Ψ. Substituting this relationship into the first-order condition (22), we get:

<!-- formula-not-decoded -->

We can then substitute this equation into the envelope condition (25) to obtain an expression for v t :

<!-- formula-not-decoded -->

From v t = ΨΦ t we finally get:

<!-- formula-not-decoded -->

This equation gives a closed-form expression for the bank's endogenous leverage limit Φ t . Intuitively, the obtainable leverage is smaller, the larger Ψ, i.e. the larger the share of the assets the bank can abscond with. Yet the leverage limit is higher if the bank's franchise value gets higher and the incentives to abscond, therefore, becomes lower, either as a result of higher expected excess returns or a higher expected deposit rate. The value of v t , furthermore, is independent of bank-level characteristics, including the bank-level net worth, so the bank's value function is indeed linear in its net worth, as postulated above.

## Aggregation

As we have shown above, the banks' leverage Φ t does not depend on bank-level factors, so

we can easily aggregate the bank-level balance sheet conditions into an aggregate balance sheet condition for the wholesale banking sector:

<!-- formula-not-decoded -->

The aggregate law of motion of the net worth of the wholesale banking sector becomes:

<!-- formula-not-decoded -->

where the second term on the right-hand side is the start-up fund Θ t +1 which the new wholesale bankers receive from the households. This start-up fund is assumed to evolve in line with actual consumer price inflation Π C,t +1 and trend productivity growth g z,t +1 . The first term represents the retained earnings of the existing wholesale bankers, a share θ t +1 of which remain bankers in period t +1.

## 2.2.4 Retail banks

As discussed in Section 2.2.1, there exists a continuum of monopolistically competitive retail banks indexed by r . Households raise Q I,r,t CF I,r,t new investment loans from an individual retail bank r , which satisfies the households' demand for new loans by purchasing new loans from the wholesale banks in an amount equal to ˜ Q I,t CF I,r,t . The retail bank differentiates the new wholesale loans and sells the differentiated loans to the households subject to a mark-down . In doing so, the retail bank obtains an upfront profit and then channels the scheduled repayments fully from the households to the wholesale banks. The profits are redistributed to the households in a lump-sum fashion.

Under the maintained assumption of a loan-price mark-down, which is embodied in the steady-state calibration of the households' basket of new investment loans with ϕ I &lt; 1 (see equation (5) in Section 2.2.1) and holds in the neighbourhood of the steady state, the retail bank's amount of loan purchases from the wholesale banks is larger than the loan amount extended to the households; that is ˜ Q I,t CF I,r,t &gt; Q I,r,t CF I,r,t . Moreover, the mark-down assumption implies a steady-state mark-up of the retail-loan-rate spread over the wholesale-loan-rate spread; see Section 3.2.1.

To capture the empirically documented sluggish adjustment of loan rates applied to new

issuance, we further assume that the retail banks face rigidities constraining their ability to re-set the prices of their loans. In particular, like in Calvo (1983), a retail bank can re-set the price of a new loan only with probability 1 -ξ I . All retail banks r that can re-set their prices in period t choose the same price Q ◦ I,t = Q ◦ I,r,t maximising

<!-- formula-not-decoded -->

This expression measures the present discounted value of profits from setting the loan price Q ◦ I,t . With probability ξ k I , the retail bank r will still offer loans at the same price in period t + k . The households' demand for its loans in period t + k is ( Q ◦ I,t / Q I,t + k ) ϕ I t + k / (1 -ϕ I t + k ) CF I,t + k , and its profit comes from the difference between the amount of payoff it receives for obtaining a unit loan ˜ Q I,t + k from the wholesale bank and the amount Q ◦ I,t it pays to the household. The retail bank is owned by the households, so it discounts the future profits with the households' nominal stochastic discount factor.

After some straightforward simplifications, the first-order condition is

<!-- formula-not-decoded -->

which implicitly determines the optimal loan price Q ◦ I,t .

A measure ξ I of the retail banks keep their previous-period price unchanged, and a measure (1 -ξ I ) set the new optimal price. Hence, the aggregate loan price index is

<!-- formula-not-decoded -->

Finally, for future use we define the nominal loan rate as the yield to maturity of the investment loans. They can be expressed as the internal rate of return of a loan with immediate payoff Q I,t and a geometrically decaying repayment schedule 1 , /rho1 I , /rho1 2 I , . . . . It is

easy to show that the gross loan rate R I,t is

<!-- formula-not-decoded -->

noting that this rate needs to be distinguished from the investment loans' ex post return.

## 2.2.5 Foreign assets and domestic bond supply

Foreign assets

Foreign assets in the form of foreign long-term government bonds are assumed to be intermediated by the wholesale banks. The idea is that foreign trade is conducted through financial intermediaries which invest the trade surplus into foreign currency assets and, in return, provide domestic households with local currency deposits. In this process, the wholesale banks enter an open foreign exchange position. As a result, exchange-rate fluctuations influence the balance sheet condition of wholesale banks (see equation (17)) and, in a two-way interaction, banks' balance sheet condition will influence the exchange rate. This is the consequence of the no-arbitrage condition between the spreads of domestic and foreign government bonds (see equation (28)).

The foreign long-term government bond holdings of the wholesale banks evolve according to:

That is, the wholesale banks invest the country's current account surplus, which encompasses the trade balance TB t /S t (expressed in foreign currency) and the coupon payments B ∗ L,t on the banks' foreign bond holdings, into long-term foreign bonds. To this end, they purchase new foreign bonds in quantity B ∗ L,t +1 -/rho1 ∗ L B ∗ L,t , which are supplied flexibly at the foreign currency price Q ∗ L,t .

<!-- formula-not-decoded -->

## Domestic government bonds

We assume an exogenous supply of domestic long-term government bonds B L,t +1 , which is financed by lump-sum taxes. We also assume that short-term government bonds B t +1 are in zero supply.

## 2.2.6 Central bank asset purchases and market clearing

## Central bank asset purchases

The central bank can purchase long-term wholesale investment loans, as well as domestic

and foreign long-term government bonds from the wholesale bank. It issues short-term excess reserves D g t +1 to finance these purchases. We assume that wholesale banks hold these excess reserves and that they cannot divert them. As a result, holding excess reserves does not tighten the balance sheet constraints of wholesale banks, because they can finance them with extra deposits. Accordingly, the central bank's balance sheet identity is

<!-- formula-not-decoded -->

where the superscript g indicates central bank assets and liabilities.

Importantly, the central bank can issue excess reserves and purchase loans and bonds without the agency friction that banks face, as in Gertler and Karadi (2011). The idea is that the public is confident that the central bank does not abscond with any of the assets. At the same time, the central bank is arguably less efficient than banks in intermediating assets, which we model simply by assuming that it faces efficiency costs ϑ I , ϑ L and ϑ ∗ L associated with its purchases of the respective asset (expressed as a share of domestic output). These asset-specific efficiency costs enter the model's aggregate resource constraint.

## Market clearing

Clearing of the loan and bond markets requires that the assets intermediated by the central bank and the wholesale banks and the assets held by the households (domestic long-term government bonds only) are equal to their aggregate supply:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The formulation of the market clearing conditions concludes the description of the financial-sector extension of the NAWM. An appendix towards the end of the paper provides details on the log-linear version of the extended model, including the derivation of its steady state, with a focus on the amendments that relate to the model's financial-sector extension.

## 2.3 Other model extensions

Reflecting the practical experience gained with using the NAWM in the policy process at the ECB over the past ten years, a number of other extensions have been made to its

original specification. In the following, we briefly describe those extensions that have been incorporated into NAWM II.

## 2.3.1 Non-zero import content of exports

The extended model allows for a non-zero import content of exports. This is an empirically important feature of the euro area economy, and the incorporation of this feature into the model has a noticeable bearing on the propagation of a variety of shocks via trade flows. Below, we describe the changes to the original model specification implied by the introduction of a non-zero import content of exports. Further details, including on the computation of the model's steady state and on the derivation of its log-linear equations, are provided in Coenen and Vetlov (2009).

Unlike in the original model, the differentiated intermediate goods Y f,t produced by its continuum of monopolistically competitive firms f ∈ [ 0 , 1] can either be directly sold to domestic producers of final goods or be combined with imported foreign intermediate goods, IM X f,t , and then be sold abroad. The non-zero import content of the intermediate goods sold abroad is modelled by assuming that the production of the exported good X f,t features a CES production technology:

<!-- formula-not-decoded -->

where H X f,t and IM X f,t are respectively the domestic and the foreign intermediate-good inputs used in its production. The parameter ν X ∈ [ 0 , 1] determines the share of domestic inputsthat is, the home bias-in the production of the exported good; and the parameter µ X &gt; 1 represents the elasticity of substitution between the domestic and the foreign intermediategood inputs.

The foreign intermediate-good input is given by the index

<!-- formula-not-decoded -->

where the possibly time-varying parameter ϕ ∗ t &gt; 0 is inversely related to the elasticity of substitution between the differentiated goods supplied by foreign intermediate-good firms f ∗ ∈ [ 0 , 1]. 11

To determine the optimal demand for the domestic and foreign inputs in the production

11 The parameter ϕ ∗ t also represents the price mark-up charged by the monopolistically competitive foreign intermediate-good firms selling their goods in domestic markets; see CCW, Section 2.2.2.

of exports, an intermediate-good firm f must solve the problem of minimising total input cost MC t H X f,t + P IM,t IM X f,t subject to the technology constraint (44), taking the prices of the inputs, MC t and P IM,t , as given. Here, MC t denotes the marginal cost of producing a unit of the domestic intermediate good H X f,t (identical across firms as they face the same input prices and since they have access to the same production technology), and P IM,t is the import price index of the foreign intermediate goods.

Defining as MC X f,t the Lagrange multiplier associated with the technology constraint (44), the first-order conditions of the firm's cost minimisation problem with respect to domestic and foreign inputs are given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The above optimality conditions determine the firm's demand for domestic and imported intermediate goods:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substituting the demand equations (48) and (49) into the production technology (44), we can express the Lagrange multiplier MC X f,t , or nominal marginal cost, in terms of the given input prices:

<!-- formula-not-decoded -->

Since all firms f face the same input prices and the same production technology, nominal marginal cost MC X f,t will be identical across firms; that is, MC X f,t = MC X t .

Finally, recalling that IM X f,t represents an index of imported differentiated intermediate goods, the demand by firm f for the imported good IM f ∗ ,t is given by

<!-- formula-not-decoded -->

where P IM,f ∗ ,t and P IM,t denote the price of the differentiated imported good f ∗ and the aggregate import price index, respectively. Hence, the aggregate demand by domestic producers for the imported good is given by

<!-- formula-not-decoded -->

where IM t X = ∫ 1 0 IM X f,t df .

## 2.3.2 Generalised price and wage Phillips curves

The specifications of the price and wage Phillips curves in NAWM II have been generalised, in part because of conceptual considerations and partly on empirical grounds. In the following, the nature of the respective generalisation is outlined and the resulting modified price and wage Phillips curves are stated. Further details are available in Coenen, Levin, and Christoffel (2007) and Coenen (2009), respectively.

## Price Phillips curves

Besides the usual nominal rigidity due to the assumption of staggered price contracts on the part of firms, the generalised price-setting framework of NAWM II incorporates a source of real rigidity to better account for the very low cyclical sensitivity of prices at the aggregate level. In particular, following Kimball (1995), the demand curves for the differentiated intermediate goods Y f,t , which are produced by a continuum of monopolistically competitive firms f ∈ [ 0 , 1] and then sold in both domestic and foreign markets, are assumed to exhibit a high degree of curvature (approximating a kinked demand curve ) as a function of the deviation of the individual firm's price from the average price level. Thus, when a firm is re-optimising its domestic and foreign price contracts, its optimal prices will be relatively less sensitive to changes in the firm's marginal cost.

Noting that the intermediate-good firms charge different prices for the goods sold in domestic markets, H f,t , and for those sold in foreign markets, X f,t , we will focus by means of example on the firms' optimal price-setting decisions for the goods sold domestically, H f,t . 12 These goods are aggregated by a distinct set of competitive firms into a final good,

12 Besides the pricing decision of the domestic intermediate-good firms f concerning their goods sold abroad, there is yet another pricing decision concerning the imported goods produced by foreign intermediategood firms f ∗ with an analogous treatment.

using the following technology:

<!-- formula-not-decoded -->

where the function G ( · ) is increasing and strictly concave with G (1) = 1. 13 Under this assumption, the steady state of aggregate output, H , is identical to the steady-state output of each individual firm, H f .

Henceforth we use µ H to denote the steady-state elasticity of demand; that is, µ H = -G ′ (1) / G ′′ (1) &gt; 1. Furthermore, we use /epsilon1 H to denote the relative slope of the demand elasticity around its steady-state value; that is, /epsilon1 H = µ H G ′′′ (1) / G ′′ (1) + µ H + 1. Thus, the special case /epsilon1 H = 0 corresponds to the specification of Dixit and Stiglitz (1977) with a constant demand elasticity, for which G ( · ) = ( · ) ( µ H -1) /µ H , with µ H representing the elasticity of substitution of the differentiated intermediate goods.

Under these assumptions, each firm f faces the following implicit demand curve for its output as a function of its price P H,f,t relative to the aggregate price index P H,t :

<!-- formula-not-decoded -->

The concavity of G ( · ) ensures that the demand curve is downward-sloping; that is, dH f,t /dP H,f,t &lt; 0. The price index P H,t can be obtained explicitly by multiplying both sides of equation (54) by the factor H f,t /H t and then integrating over the unit interval:

<!-- formula-not-decoded -->

As in the original model, there is sluggish price adjustment due to staggered price contracts ` a la Calvo (1983). Accordingly, in a given period t each firm f receives permission to optimally reset the price of the output sold in the domestic market with probability 1 -ξ H . All firms that receive permission to reset their price contracts in a given period t choose the same optimal price, P ◦ H,t = P ◦ H,f,t . Those firms which do not receive permission are allowed to adjust their prices according to the following scheme:

<!-- formula-not-decoded -->

13 It is assumed that the competitive final-good firms producing the private consumption good Q C t , the private investment good Q I t and the public consumption good Q G t have the same implicit demand function for the domestic intermediate good f . Accordingly, it is sufficient to consider the aggregate demand for this good from the three final-good firms, H f,t = H C f,t + H I f,t + H G f,t .

That is, the price contracts are indexed to a geometric average of past (gross) intermediategood inflation, Π H,t -1 = P H,t -1 /P H,t -2 , and the central bank's possibly time-varying (gross) inflation objective, ¯ Π t , where χ H ∈ [0 , 1] is an indexation parameter determining the weight on past inflation.

We use ̂ π H,t = log(Π H,t / ¯ Π) to denote the logarithmic deviation of the rate of domestic price inflation from the central bank's long-run inflation objective, while ̂ ¯ π t = log( ¯ Π t / ¯ Π) represents the logarithmic deviation of the possibly time-varying inflation objective from its long-run value. Then, combining the log-linearised first-order condition characterising the firms' optimal price-setting decision and the log-linearised aggregate price index yields the following log-linear domestic price Phillips curve: 14

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ̂ mc H t and ̂ ϕ H t denote, respectively, firms' real marginal cost (identical across firms as they face the same input prices and since they have access to the same production technology) and a time-varying price mark-up. Both variables are expressed as logarithmic deviations from their respective steady-state values, with the steady-state price mark-up ϕ H being inversely related to the steady-state elasticity of demand µ H .

It should be emphasised that the real rigidity coefficient Ψ( µ H , /epsilon1 H ) depends solely on the relative curvature of the firms' demand function and has a value of unity in the special case with constant demand elasticity; that is, for the Dixit-Stiglitz specification with /epsilon1 H = 0. Furthermore, it should be noted that the parameter /epsilon1 H is not separately identified from the Calvo parameter ξ H determining the degree of nominal rigidity in firms' price-setting behaviour. Nevertheless, as will be shown in Section 3 below, a reasonable calibration of the curvature of the demand function will help to account for the strong empirical evidence on the existence of a very flat price Phillips curve for the euro area, without necessarily relying on an implausibly large Calvo parameter.

14 Details of the derivation of the log-linear price Phillips curve are available upon request.

with

Wage Phillips curve

In generalising the wage-setting framework of the original NAWM, we allow for less than full indexation of wages to trend labour productivity. Like in the case of price setting by firms, there is sluggish wage adjustment by a continuum of households h ∈ [ 0 , 1] due to staggered wage contracts ` a la Calvo. Accordingly, in a given period t each household h receives permission to optimally reset its nominal wage contract W h,t with probability 1 -ξ W . All households that receive permission to reset their wage contracts in a given period t choose the same optimal wage rate W ◦ t = W ◦ h,t . Those households which do not receive permission are allowed to adjust their wage contracts according to the following scheme:

<!-- formula-not-decoded -->

where g † z,t = g ˜ χ W z,t g 1 -˜ χ W z and Π † C,t = Π χ W C,t -1 ¯ Π 1 -χ W t . That is, the nominal wage contracts are indexed both to a geometric average of the current (gross) rate of productivity growth, g z,t = z t /z t -1 , and the steady-state (gross) rate of productivity growth, g z , and to a geometric average of past (gross) consumer price inflation, Π C,t -1 = P C,t -1 /P C,t -2 , and the central bank's possibly time-varying (gross) inflation objective, ¯ Π t , where ˜ χ W ∈ [0 , 1] and χ W ∈ [0 , 1] are the respective indexation parameters.

In analogy to the derivation of the log-linear price Phillips curve above, we use ̂ π C,t = log(Π C,t / ¯ Π) to denote the logarithmic deviation of the rate of consumer price inflation from the central bank's long-run inflation objective, while ̂ ¯ π t = log( ¯ Π t / ¯ Π) represents the logarithmic deviation of a possibly time-varying inflation objective from its long-run value. Similarly, we define the logarithmic productivity growth deviation as ̂ g † z,t = log( g † z,t /g z ) = log( g ˜ χ W z,t /g ˜ χ W z ). Then, combining the log-linearised first-order condition characterising the households' optimal wage-setting decision and the log-linearised aggregate wage index yields the following log-linear wage Phillips curve: 15

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

where ̂ w t ( ̂ w τ t ) represents the productivity-adjusted (after-tax) real wage deflated by the consumer price index, while ̂ mrs t and ̂ ϕ t W denote, respectively, the households' marginal rate of substitution between consumption and leisure (identical across households as we assume that there exists a mechanism which provides insurance against household-specific income risk) and a time-varying wage mark-up. All variables are expressed as logarithmic deviations from their respective steady-state values. The parameter ζ is the inverse of the households' Frisch elasticity of labour supply which can be interpreted as a source of real rigidity influencing the slope of the wage Phillips curve.

Finally, it should be noted that for the original NAWM with full indexation of wages to trend productivity (that is, with ˜ χ W = 1) the two separate productivity terms in the above log-linear wage Phillips curve vanish.

## 2.3.3 Uncertainty about shifts in trend productivity

In the original NAWM, labour productivity is shifted recurrently, and lastingly, by shocks to the intermediate-good firms' production technology. These permanent technology shocks z t are assumed to evolve according to a serially correlated process,

<!-- formula-not-decoded -->

where g z,t = z t /z t -1 represents the (gross) growth rate of trend labour productivity. The parameters g z and ρ g z determine, respectively, the unconditional mean (that is, the drift ) and the degree of persistence of the process, and η g z t is a serially uncorrelated innovation.

While future shocks to trend labour productivity are unknown, a shock occurring in the current period is perfectly observed by all economic agents, including the central bank. Following a positive shock, forward-looking households and firms predict a rise in their future income and profits and, hence, they increase their current level of spending. The strong wealth effect of the anticipated rise in income and profits eventually boosts aggregate demand beyond supply in the short run and may give rise to a temporary increase in firms' costs of production. As a result, inflationary, rather than disinflationary, pressures can emerge following a positive permanent technology shock.

In NAWM II, the strength of the wealth effect associated with a permanent technology

shock is mitigated by introducing uncertainty on the part of economic agents about the degree of persistence of the shock following the approach in Edge, Laubach, and Williams (2007). In particular, it is assumed that the permanent technology shock ̂ g z,t (expressed as the logarithmic deviation from its deterministic component) is the composite of two distinct components: a persistent component, ̂ g p z,t , and a transitory component, ̂ g tr z,t . Both components have a permanent impact on the level of trend labour productivity. They differ, however, to the extent that they have either a protracted or a one-off impact on overall trend labour productivity growth:

<!-- formula-not-decoded -->

Here, ̂ η g p z t and ̂ η g tr z t are serially uncorrelated innovations to the persistent and to the transitory component of the permanent technology shock with variances σ 2 g p z and σ 2 g tr z , respectively. The parameter ρ g p z determines the degree of persistence of the former component.

<!-- formula-not-decoded -->

The agents in the extended model do not directly observe the two individual components. Instead, they only observe the composite permanent technology shock,

<!-- formula-not-decoded -->

and need to estimate the values of the individual components by recursively solving a signalextraction problem.

As in Edge, Laubach, and Williams (2007) we assume that, to solve the signal-extraction problem and to update the estimates of the persistent and the transitory component of the permanent technology shock over time, the agents employ the Kalman filter. The resulting updating rules for the (steady-state) Kalman filter are as follows:

<!-- formula-not-decoded -->

where K g p z and K g tr z are the Kalman gain parameters governing the rate of updating,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with the ratio of variances σ 2 g p z /σ 2 g tr z representing the signal-to-noise ratio associated with the signal-extraction problem.

In order to solve the extended model jointly with the signal-extraction problem, we append the Kalman filter updating equations to the structural equations of the model as a separate recursive block. To achieve recursiveness, we introduce a change in notation from ̂ g p z,t and ̂ g tr z,t to ˜ g p z,t and ˜ g tr z,t . Moreover, to ensure that the appended Kalman filter block is complete, we complement the updating equations with laws of motion for ˜ g p z,t and ˜ g tr z,t , which duplicate equations (61) and (62) above.

A suitable model solution algorithm-e.g., the AiM algorithm developed in Anderson and Moore (1985), or the generalised Schur form-based algorithm of Klein (2000) 16 -can then be used in a first step to solve for agents' forward-looking expectations so as to obtain the observed structure of the model. This structure respects the information set available to agents in period t and is not influenced by the Kalman filter updating equations (64) and (65). In particular, it generically depends on agents' periodt estimates of the individual shock components, ̂ g p z,t | t and ̂ g tr z,t | t . In the same step, the solution algorithm provides recursive Kalman filter-based periodt estimates of the shock components, ˜ g p z,t | t and ˜ g tr z,t | t , which reflect the new information about the composite technology shock that has become available while moving from period t -1 to period t . In a second step, agents' estimates of the shock components in the observed structure are set equal to the Kalman filter-based estimates, ̂ g p z,t | t = ˜ g p z,t | t and ̂ g tr z,t | t = ˜ g tr z,t | t . This requires appropriately modifying the coefficient matrices of the observed structure obtained from the model solution algorithm. The modified coefficient matrices allow to compute the recursive reduced-form representation of the full model, including the solution of the signal-extraction problem, in a straightforward way.

## 2.3.4 Time-varying equilibrium real interest rate

In the original NAWM, the central bank is assumed to adjust the logarithmic deviation of the (riskless) short-term nominal interest rate from its long-run level, ̂ r t = log( R t /R ), in response to logarithmic deviations of some key macroeconomic variables-inflation and output (growth)-from their respective target or equilibrium levels, i.e. by following a Taylor (1993)-type interest-rate feedback rule. The long-run nominal interest rate R is exogenously determined as the product of the non-stochastic steady-state rate of inflation Π (equal to the central bank's long-run inflation objective ¯ Π) and the long-run equilibrium real interest

16 Both algorithms are available in YADA, the Matlab programme used for solving, estimating and simulating NAWM II; see Warne (2018).

rate R r , with the latter being influenced by long-run trend productivity growth, R r = β -1 g z . The potential shortcoming of this specification, however, is that, in case there is a slowdown in economic activity due to lower productivity growth and if this slowdown is expected to persist well into the future, the model tends to prescribe, all else equal, a mistakenly tight monetary policy stance for current and future periods.

Given the slowing of euro area productivity growth over the past decades, the interestrate rule in NAWM II is now amended by relating the central bank's estimate of the equilibrium real interest rate to perceived medium-term fluctuations in productivity growth, as captured by the estimate of the persistent component of the model's permanent technology shock g p z,t | t , with R r t | t = β -1 g p z,t | t . The resulting log-linear interest-rate rule is:

<!-- formula-not-decoded -->

This specification of the model's interest-rate rule implies that the central bank recognises persistent deviations of the long-run productivity growth rate from its steady-state value and alters the short-term nominal interest rate accordingly, namely by lowering (increasing) the interest rate in response to negative (positive) shifts in the perceived value of the persistent component of the model's permanent technology shock.

where ̂ r r t | t = ̂ g p z,t | t , ̂ π C,t = log(Π C,t / ¯ Π) denotes the logarithmic deviation of (gross) consumer price inflation Π C,t = P C,t /P C,t -1 from the central bank's long-run inflation objective ¯ Π, and ̂ ¯ π t = log( ¯ Π t / ¯ Π) represents the logarithmic deviation of the central bank's possibly timevarying inflation objective from its long-run value. Similarly, ̂ y t = ̂ Y t /z t is the logarithmic deviation of aggregate output Y t from the trend output level z t , which is determined by the history of the composite permanent technology shock { g z,t , g z,t -1 , . . . } . This deviation will be referred to as output gap hereafter. 17 Finally, ̂ η R t is a serially uncorrelated shock to the short-term nominal interest rate.

## 2.3.5 Time-varying perceived inflation objective

NAWM II also accounts for situations in which persistent fluctuations in inflation outcomes eventually result in a drift of private-sector agents' longer-term inflation expecta-

17 We acknowledge that this output gap measure has no strong theoretical basis (see Gal´ ı and Gertler 2007). Exploring the model with a theory-consistent measure of the output gap-that is, the deviation of actual output from the output level that can be attained in an environment of full nominal flexibility (see Woodford 2003)-is left for the future.

tions. Within the model, a possible drift of longer-term inflation expectations is implemented through a gradual shift in private-sector agents' perception of the central bank's inflation objective, ¯ Π p t , which provides the long-run anchor for their formation of inflation expectations. At the same time, it is assumed that the actual inflation objective of the central bank, which is guiding its policy decisions, remains unchanged, ¯ Π t = ¯ Π. 18

The perceived inflation objective is determined via a simple adaptive scheme (formulated in terms of logarithmic deviations from the central banks's invariant long-run inflation objective):

<!-- formula-not-decoded -->

According to the reformulation of the adaptive scheme in the second equality, the current value of the perceived inflation objective is determined as a weighted average of past annual consumer price inflation and the value of the perceived inflation objective in the previous period. 19 If the weight on past inflation /pi1 ¯ Π p is zero, and in the absence of shocks both in the present and in past periods, the perceived inflation objective is equal to the actual inflation objective, with ̂ ¯ π p t = 0. Otherwise, and depending on the magnitude of /pi1 ¯ Π p , the perceived inflation objective is influenced by movements in actual inflation, with possibly adverse economic consequences. For example, households and firms will respond to a downward shift in the perceived inflation objective by downwardly adjusting their desired wage and price contracts. 20 Eventually, this will lead to the materialisation of self-reinforcing secondround effects and an ensuing fall in inflation expectations. This in turn will push up the real interest rate and, as a consequence, curtail aggregate demand. The implied demand shortfall will exacerbate the downward pressure on wages and prices, over and above the with ̂ ¯ π p t = log( ¯ Π p t / ¯ Π) and ̂ π (4) C,t -1 = log(( P C,t -1 /P C,t -5 ) / ¯ Π 4 ) / 4. The parameter /pi1 ¯ Π p ∈ [0 , 1] measures the sensitivity of the perceived inflation objective to past inflation, and ̂ η ¯ Π p t is a serially uncorrelated shock to the perceived objective.

18 Linking the anchoring of longer-term inflation expectations to private-sector agents' perceptions of the central bank's inflation objective is motivated by the widespread notion that stable inflation expectations are inherently linked to the central bank's commitment to achieving, and the ability to achieve, its inflation objective over the medium term.

19 For further discussion and for details on modelling inflation expectations using the adaptive scheme, see G¨ urkaynak, Levin, Marder, and Swanson (2007).

20 Within NAWM II, this adjustment is implemented through replacing the time-varying inflation objective term ̂ ¯ π t with the perceived inflation objective term ̂ ¯ π p t in the price and wage Phillips-curve equations (57) and (59) stated in Section 2.3.2 above.

direct effects resulting from the fall in inflation expectations.

Another important consequence that arises from misperceptions of the central bank's inflation objective is that the central bank's actual interest-rate rule differs from the interestrate rule assumed by households and firms in their decision-making: 21 when forming expectations of current and future short-term nominal interest rates households and firms will over-predict the interest rate as the perceived inflation objective falls below the actual objective, and vice versa. This mismatch between actual and perceived interest rates can have material effects on aggregate demand on its own.

To solve the augmented model, the adaptive scheme determining the perceived inflation objective (69) is appended to the model's structural equations as a separate recursive block, like in the case of solving NAWM II with the signal-extraction problem concerning the unobserved components of its permanent technology shock (see Section 2.3.3). Accordingly, in the employed two-step solution procedure, the model's observed structure is first determined conditional on the central bank's possibly time-varying inflation objective ̂ ¯ π t . The latter is then set equal to the perceived objective ̂ ¯ π p t by means of appropriate adjustments to the coefficient matrices of the observed structure, which thereafter allows to compute the model's reduced-form representation.

## 3 Bayesian estimation

As in the case of the NAWM, we adopt the approach outlined in An and Schorfheide (2007) and Schorfheide (2000) and estimate NAWM II using Bayesian inference methods. This involves obtaining the posterior distribution of the model's estimated parameters based on its log-linear state-space representation. 22 For a detailed exposition of these methods, see Herbst and Schorfheide (2016). The computations are performed with YADA, a Matlab programme for Bayesian analysis of DSGE models; see Warne (2018).

An extensive discussion of the estimation results for NAWM II is beyond the scope of this paper. Here we report selectively on the data used in the estimation, on the model's structural shocks, on the calibration of its steady-state expenditure shares and of a subset of financial and non-financial-sector parameters, and, as a last step, on the prior and posterior

21 Specifically, the interest-rate rule of the central bank in equation (68) of Section 2.3.4 above includes the actual inflation objective, with ̂ ¯ π t = 0 in case the actual objective remains unchanged, whereas the interest-rate rule assumed by households and firms in forming their expectations includes the time-varying perceived inflation objective, ̂ ¯ π p t .

22 For details on the derivation of the model's log-linear representation around its non-stochastic steady state, see the appendix towards the end of the paper.

distributions of the financial and non-financial-sector parameters that we chose to estimate. The main focus of the discussion is on those aspects that help to understand the enhanced role of the financial intermediary sector in our extended model. Some additional discussion relates to the novel use of system priors. For details concerning the estimation of the original model, see CCW, Section 3.

## 3.1 Data and shocks

## 3.1.1 Data

In estimating the original version of the NAWM, CCW used times series for 18 macroeconomic variables which feature prominently in the ECB/Eurosystem staff projection exercises: real GDP, private consumption, total investment, government consumption, extraeuro area exports and imports, the GDP deflator, the consumption deflator, the extra-euro area import deflator, total employment, nominal wages (measured as compensation per employee), the short-term nominal interest rate (corresponding to the 3-month EURIBOR), the nominal effective exchange rate of the euro, foreign demand, foreign prices, the foreign shortterm interest rate, competitors' export prices, and the price of oil. For estimating NAWM II, we retain the same variables, but with an extended estimation sample ranging from 1985Q1 to 2014Q4 (again using the period 1980Q2-1984Q4 as training sample). The time series of the variables are taken from an updated version of the AWM database (see Fagan, Henry, and Mestre 2001), except for the time series of the extra-euro area trade variables. 23 The last five variables of the above list form part of a structural vector-autoregressive (SVAR) model of the euro area's external environment, the estimated parameters of which are kept fixed throughout the analysis in this paper. Similarly, government consumption is specified by means of an autoregressive (AR) model with fixed parameters.

The time series of the 18 macro variables are displayed in Figure 1, with real GDP, private consumption, total investment, exports, imports, the GDP deflator, the consumption deflator, the import deflator, nominal wages, foreign demand, and foreign prices being all transformed into quarterly growth rates (approximated by the first difference of their logarithm). All other variables are measured as logarithmic levels, except for the domestic and foreign short-term nominal interest rates which are expressed in annualised percentage terms. In contrast to the estimation of the original version of the NAWM, all real variables

23 Note that the foreign interest rate is proxied by the US Dollar 3-month LIBOR instead of the federal funds rate used in the estimation of the original NAWM, and that the other foreign variables and the euro nominal effective exchange rate are constructed as a weighted average for the euro area's 38 main trading partners instead of the smaller group of countries covered in earlier vintages of the AWM database.

used in the estimation of NAWM II have beforehand been transformed into per-capita units by scaling them with a measure of the euro area working age population (aged 15 to 64). This scaling is motivated by the deceleration in working age population growth observed over the extended estimation period.

A number of further transformations are made to ensure that variable measurement is consistent with the properties of the balanced-growth path for NAWM II and that the underlying modelling assumption of stationarity of all relative prices is satisfied. First, the sample mean growth rates of extra-euro area exports and imports as well as foreign demand are matched with the sample mean growth rate of real GDP. Second, we take the logarithm of government consumption and remove a linear trend consistent with the model's steady-state growth rate for aggregate output of 1.5% per annum. Third, for the logarithm of employment we remove a linear trend consistent with an increase in the labour force of 0.3% per annum. 24 Fourth, we construct a measure of the real effective exchange rate from the nominal effective exchange rate, the domestic GDP deflator and foreign prices (computed as a weighted average of foreign GDP deflators) and then remove the sample mean. Finally, competitors' export prices and oil prices (both expressed in the currency basket underlying the construction of the nominal effective exchange rate) are deflated with foreign prices before linear trends are removed from the deflated variables.

For the estimation of NAWM II, we use six additional time series, which are displayed in Figure 2. First, the added financial data used in the estimation comprise time series for 10-year government bond yields for the euro area and for the United States and a composite euro area long-term lending rate. The euro area 10-year government bond yield series covers, in changing composition, all sovereign issuers whose rating is AAA. It is available from 2004Q3 onwards, while the earlier part of the series concerns the German 10-year government bond yield. The 10-year government bond yield for the United States corresponds to the US 10-year treasury yield, covering the full estimation sample. The euro area long-term lending rate is available from 2003Q1 onwards and comprises (new business) lending with an original maturity of over one year to households for house purchases and to non-financial corporations. The sources of the financial data are the ECB's Statistical Data Warehouse, the Deutsche Bundesbank database, and the FRED database of the Federal Reserve Bank of St. Louis. All data are expressed as annualised percentages.

24 In other words, the steady-state output growth rate of 1.5% is assumed to have two components: trend labour productivity growth of 1.2% and trend labour force growth of 0.3%; see Section 3.2.1 for details. The former component is broadly in line with average labour productivity growth over the estimation period, whereas the latter one approximates the steady increase in labour force participation which causes an upward trend in our (per-capita) employment measure.

Second, we use survey data on long-term inflation expectations and long-term growth expectations from the ECB's Survey of Professional Forecasters (SPF) as proxy measures for the unobserved perceived inflation objective and trend productivity growth, respectively. These measures refer to expectations five (calendar) years ahead and are expressed in terms of annual percentages. They are available in quarterly frequency from 2001 onwards, as well as for the respective first quarters of the years 1999 and 2000. And third, we use a measure of the euro area output gap, which is constructed using the European Commission's estimate of potential output taken from the Autumn 2014 vintage of its AMECO database (after having interpolated the annual potential output data into quarterly data using a cubic spline) and our real GDP data. The output gap measure is expressed in percent of potential output and covers the period from 1999Q1 onwards.

## 3.1.2 Shocks

Commensurate with the number of time series used in the estimation, NAWM II features 24 distinct structural shocks. 25 In particular, we retain the 12 structural shocks incorporated in the original NAWM (see CCW, Section 3.2.2), including the domestic and external risk premium shocks, which can be interpreted as (exogenous) liquidity premia driving a wedge between the interest rate controlled by the central bank and the returns required by privatesector agents, as well as the permanent technology shock, capturing fluctuations in trend labour productivity growth. The permanent technology shock in NAWM II is however split into a persistent and a transitory component, giving rise to a signal-extraction problem on the part of all agents; see Section 2.3.3.

In addition, we distinguish two structural shocks originating in the financial intermediary sector of NAWM II: a shock to the survival rate of the model's wholesale banks, and a shock to the mark-down parameter of its retail banks. A shock to the survival rate affects the net worth of the wholesale banking sector and, because of the prevailing funding constraint, their ability to originate loans, whereas a shock to the mark-down parameter of retail banks captures shifts in their market power and drives a wedge between the wholesale lending rate and the retail lending rate in the model. We also add shocks to the two updating equations for the private sector's perceived inflation objective of the central bank and for the perceived trend growth rate. Finally, we augment the SVAR model of the foreign variables with an

25 Like in the original model, NAWM II also allows for measurement error in extra-euro area trade data as they are prone to sizeable revisions. It also accounts for small errors in the measurement of real GDP and the GDP deflator to alleviate discrepancies between the national accounts framework underlying the construction of official GDP data and the model's aggregate resource constraint.

equation for the US 10-year government bond yield and an associated yield shock. 26

All shocks are assumed to follow first-order autoregressive processes, except for the shock to the transitory component of the permanent technology shock, the shock to the central bank's interest-rate rule, the shocks in the two updating equations, and the shocks in the SVAR model of the foreign variables and in the AR model of government consumption. These shocks are assumed to be serially uncorrelated innovations.

## 3.2 Calibration and prior distributions

## 3.2.1 Calibration

In NAWM II, all real variables are assumed to evolve along a balanced-growth path with a steady-state growth rate of 1.5% per annum, which roughly matches average per-capita real GDPgrowth in the estimation sample. Consistent with the balanced-growth assumption, we calibrate the steady-state expenditure shares of the model's aggregate demand components by broadly matching their empirical counterparts over the sample period; see Table 1 for details. Specifically, the shares of private consumption, total investment and government consumption are set equal to, respectively, 57.5%, 21.0% and 21.5% of nominal GDP, while the extra-euro area export and import shares are set to 16%, ensuring balanced trade in steady state. Regarding the import content of the individual expenditure components, we choose values for the quasi-share parameters ν C and ν I in the model's aggregators for final private consumption and final investment-good production that allow the model to replicate the import content of private consumption and total investment expenditures, which amount to roughly 8.5% and 4.5% of nominal GDP according to data from inputoutput tables for the euro area. Similarly, we choose the quasi-share parameter ν X in the export aggregator to calibrate the import content of exports at 3%, whereas government consumption is assumed to relate to domestic goods only ( ν G = 0).

Information on the calibration of selected non-financial-sector parameters of NAWM II is presented in Table 2. Concerning preferences, the households' discount factor β is chosen to imply a steady-state equilibrium (net) real interest rate of 2.0% per annum, conditional on the model's steady-state rate of trend productivity growth. The inverse of the Frisch elasticity of labour supply ζ , which is inherently difficult to identify empirically, is assigned a value of 2 as in CCW and in line with the range of available estimates in the literature.

26 Identification of the shocks in the SVAR model is achieved by a Choleski decomposition of its estimated variance-covariance matrix, the ordering of variables being: foreign prices, foreign demand, foreign shortterm interest rate, foreign long-term interest rate, oil prices and competitors' export prices.

As regards technology parameters, we set the depreciation rate of physical capital δ equal to 10% per annum. The capital share in the domestic firms' intermediate-good production technology α is set to 36%, while the fixed-cost parameter ψ is calibrated such that firms' profits are zero in steady state. Labour productivity is assumed to rise at a trend rate of 1.2% per annum, which is broadly in line with average productivity growth over the sample period, while employment is assumed to grow with a trend rate of 0.3% per annum, approximating an increasing trend in labour force participation.

Our calibration of the steady-state mark-ups influencing households' and firms' wage and price-setting decisions in the model follows the calibration in CCW, while the slope of the demand elasticity implied by the Kimball aggregator used in the derivation of the model's generalised price Phillips curves is set uniformly equal to 10, following Coenen, Levin, and Christoffel (2007). We also keep the values of the tax rates at the values chosen in CCW, noting that we now allow for a wedge between the return on capital and the shortterm government bond yield which is chosen in the computation of the model's steady state so as to be consistent with a capital income tax rate of 30%.

With regard to monetary policy, the central bank's long-run (net) inflation objective is set equal to 1.9% per annum, consistent with the ECB's quantitative definition of price stability. The long-run inflation objective determines the model's steady-state inflation rate, which, in combination with a steady-state equilibrium real interest rate of 2.0% per annum, implies a steady-state short-term nominal interest rate of 3.9% per annum.

Details on the calibration of key financial-sector parameters of NAWM II are provided in Table 3. The behaviour of the wholesale banks in the model are governed by three central parameters: the absconding rate Ψ, the steady-state value of the normalised start-up fund, ϑ , 27 and the steady-state value of the survival rate, θ . We calibrate these parameters to match certain steady-state moments. First, we set the steady-state leverage of the wholesale banks equal to 6, which corresponds to the average asset-over-equity ratio of monetary and other financial institutions as well as non-financial corporations, with weights equal to their share of assets in total assets between 1999Q1 and 2014Q4 according to the euro area sectoral accounts. 28 Second, by making use of the fact that R D = R , we set the annualised steady-state spread of the retail lending rate over the riskless rate, R I -R , equal to 2.17

27 The need for the normalisation of the start-up fund Θ t arises from the fact that the start-up fund is assumed to evolve in line with consumer price inflation and trend productivity growth; see Section 2.2.3. For details on the normalisation, see the appendix.

28 We include the leverage of non-financial corporations because, in practice, both banks and firms hold leveraged positions. In our model, firms do not hold leveraged positions, so we assign the average composite leverage to financial intermediaries.

percentage points, which is the average spread between the long-term cost of private-sector borrowing and the short-term nominal interest rate over the period from 2003Q1 to 2014Q4. Third, we set the wholesale banks' planning horizon equal to 5 years, corresponding to a survival rate of 0.95. The calibrated parameters are in line with those used in the related literature (see, e.g., Gertler and Karadi 2011). As regards the behaviour of the retail banks, we target a 10% steady-state mark-up of the retail lending rate spread, R I -R , over the wholesale lending rate spread, ˜ R I -R . In particular, we set the mark-down parameter ϕ I to a value such that α I ( R I -R ) = ˜ R I -R , where 1 /α I = 1 . 1.

The decay parameters /rho1 I , /rho1 L and /rho1 ∗ L of the long-term assets in the model are calibrated so that these assets have the same duration as long-term bonds with 10-year maturities. The evolution of the yields of the model's domestic and foreign government bonds can therefore be directly matched with the developments in 10-year yields, which are among the observed variables used in the estimation. We choose the same benchmark value for the long-term investment loans because there are no better generally accepted measures for the maturity of long-term loans in the euro area. Our predictions about the impact of asset purchases presented below are not sensitive to this calibration.

The supply of domestic long-term government bonds B L is calibrated such that the steady-state bond supply Q L B L equals 70% of nominal GDP, whereas the constant B L,h in the portfolio adjustment cost function of the households is chosen such that, in steady state, 25% of the domestic long-term government bonds are held by wholesale banks. This is broadly consistent with the sovereign debt holdings of financial intermediaries according to the euro area sectoral accounts. Finally, we set the efficiency costs ϑ I , ϑ L , and ϑ ∗ L of asset purchases equal to zero. Our results are not sensitive to these parameter choices.

## 3.2.2 Marginal priors and system priors

## Choice of marginal priors

Concerning the choice of the marginal prior distributions for the estimated parameters that are common to NAWM II and the original NAWM, we use broadly the same priors as CCW (see Section 3.3 and Table 1). 29 So our discussion here focuses on the prior distributions of the parameters characterising the financial intermediary sector in the extended model, as detailed in the third column of Table 4.

29 An exception is the prior distribution for the investment adjustment cost parameter, for which we have chosen a somewhat higher mean so as to contain the heightened role of the interest-rate channel in determining investment spending relative to consumption spending. In addition, the tightness of several prior distributions has been moderately increased to better cope with the computational challenges that result from the model's high-dimensional parameter space.

The means of the priors for some of the financial-sector parameters are informed by longterm averages of the relevant observables used in the estimation of NAWM II. Specifically, the prior mean of the relative absconding rate of the domestic government bond ω L is set equal to the ratio of the spread between the euro area 10-year AAA government bond yield and the short-term nominal interest rate over the period from 2003Q1 to 2014Q4 (being equal to 1.4 percentage points) and the spread between the long-term cost of borrowing and the short-term rate without the 10% retail mark-down. The prior mean of the absconding rate of the foreign government bond relative to the domestic bond ˜ ω ∗ L is calibrated to equal unity to reflect the broad similarity of the average US and euro area 10-year AAA yields between 2003Q1 and 2014Q4.

The choice of priors for the other financial-sector parameters is more agnostic. The prior mean of the Calvo parameter ξ I governing the retail banks' loan re-pricing decisions is set equal to 0.75, consistent with a rather flexible re-pricing policy, while the means of the portfolio adjustment cost parameters γ h L and γ ∗ L of households and wholesale banks are conservatively assigned rather small values, containing the role of financial frictions a priori. Finally, the priors for the parameters of the model's financial shock processes follow standard specifications of priors for structural shock processes in the literature.

## Choice of system priors

The marginal prior distributions discussed so far are based on the assumption that the parameters are independent, a property which is often at odds with what is a priori envisaged. For example, certain parameters of the central bank's interest-rate rule (68), such as φ Π and φ Y , are expected to be related since they are arguably influenced by the weights given to inflation and output fluctuations in the central bank's implicit objective function. While these parameters should perhaps be negatively correlated in a realistic joint prior distribution, there is typically a lack of guidance from economic theory on how to model dependence between parameters.

Although correlations between parameters can be introduced in many ways, one natural approach is to consider particular model or system features that one would like to condition the empirical analysis on. Several papers have introduced ways that such information can be accounted for in DSGE models, albeit with different terminology. Del Negro and Schorfheide (2008) suggest an approach for introducing beliefs about steady-state relationships and second moments of the endogenous variables to avoid priors which, among other things, assign too much mass to steady-state values and other functions of the parameters that are eventually unreasonable. In this context, Christiano, Trabandt, and Walentin (2011) refer

to endogenous priors and focus on the population standard deviations of observed variables used in the estimation. Their choice of a joint Gaussian distribution for these moments is motivated by asymptotic arguments. Andrle and Beneˇ s (2013), on the other hand, extend the ideas in Del Negro and Schorfheide (2008) by discussing what they call system priors , a concept which includes priors about, for instance, conditional and unconditional population moments, impulse-response functions, or outcomes of hypothetical policy experiments such as the sacrifice ratio.

In this paper we follow the terminology of Andrle and Beneˇ s (2013) and refer to a system prior as a set of prior beliefs about certain system characteristics which can be modelled with a suitable density function conditional on the model parameters θ . Let h ( θ ) be a vector-valued function of the system features concerned and h S a vector of fictitious measurements such that with ω S being a vector of measurement errors. This relation can be expressed as a likelihood function (conditional density), L ( h ( θ ) | h S ) = p ( h S | h ( θ )), which, when combined with a marginal prior density p ( θ ) and Bayes theorem, implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, the density p ( θ | h S ) is the system prior , while the constant of proportionality is given by the marginal system prior likelihood p ( h S ).

To formulate a system prior for NAWM II, we focus on two system features for which the original prior p ( θ ) may not be sufficiently flexible. First, the population covariance matrix of the observed variables used in the estimation of the model is a complex function of its parameters and having reasonable prior distributions for this matrix is difficult in practice. Second, a particularly important parameter determining the relationship between real economic developments and inflation is the 'slope' coefficient (that is, the reducedform coefficient on firms' marginal cost) in the model's domestic price Phillips curve. It is deemed useful to exert some control over the prior for this parameter, instead of formulating merely marginal priors for the underlying structural parameters.

A common prior for covariance matrices is the inverted Wishart distribution. However this distribution is often inadequate because of its restrictive form, implying a common number of degrees of freedom. Barnard, McCulloch, and Meng (2000) suggest to begin from the standard decomposition of a covariance matrix into standard deviations σ and

correlations c , and to formulate the joint prior as a marginal density for σ times a density for c conditional on σ . Specifically, they consider a multivariate log normal prior for the standard deviations and a joint, or marginal, uniform prior for the correlations.

We follow this approach to formulating a joint prior for the population covariance matrix for the following observed variables: real GDP, private consumption, total investment, the GDP deflator and the private consumption deflator (with variable measurements as detailed in Section 3.1.1). In particular, we assume that each standard deviation has a log normal prior, while each correlation has a uniform prior with lower and upper bounds of -1 and 1, respectively. For the log normal priors we use pre-sample data for the period 1980Q2-1984Q4 to suggest a suitable mean. Specifically, we assume that the population standard deviation of real GDP has prior mean 0.55 and standard deviation 0.1. For private consumption we set these hyperparameters equal to 0.45 and 0.1, respectively. For total investment they are 1.35 and 0.2, while for the GDP and private consumption deflators we set them to 0.45 and 0.05.

Turning next to the prior distribution for the slope coefficient of the domestic price Phillips curve (see equation (57) in Section 2.3.2), we consider the transformation S mc = 100 s mc , where

<!-- formula-not-decoded -->

We assume that S mc has a gamma distribution with mean 1 and standard deviation 0.15. The selected value of the mean is based on the estimate of the slope coefficient of the price Phillips curve in Smets and Wouters (2007), computed from the posterior mode estimates of the underlying parameters reported in Lind´ e, Smets, and Wouters (2016, Table 7; No ZLB model) using US data until 2014Q2.

## Comparing marginal and system priors

To assess the effects on the parameters from using the system prior information, we compare the marginal system prior densities to the marginal prior densities. While the marginal priors p ( θ ) have analytical expressions and allow for direct sampling, the system prior densities p ( θ | h S ) are unknown, but can be estimated via standard Markov-Chain MonteCarlo (MCMC) methods. Specifically, in our application of these methods we use the random-walk Metropolis sampler with 200,000 draws to enable kernel density estimation of the marginal densities of the system prior; see Herbst and Schorfheide (2016, Chapter 4.1) for details about the posterior sampler.

The two sets of densities for the individual parameters of NAWM II are plotted in Appendix Figure A.1. It can be seen that imposing the system prior tightens the density of the Calvo parameter ξ H entering the slope coefficient of the model's domestic price Phillips curve. The standard deviation of the marginal system prior is estimated at 0.015, while the standard deviation from the original marginal prior is equal to 0.0375. Furthermore, the density of the indexation parameter χ H shifts somewhat to the left due to a lower mean of 0.72 for the system prior, compared with a mean of 0.75 for the original prior. An explanation for these effects is arguably the fairly tight system prior on the slope coefficient of the domestic price Phillips curve.

As regards the parameters of the interest-rate rule in equation (68), the system prior information mainly affects the coefficient φ Π on contemporaneous inflation. For this parameter, the system prior has a higher mean and a lower standard deviation than the original prior. Finally, and as expected, the system prior tightens the densities for some of the standard deviations of the shock innovations, in particular for the persistent component of the permanent technology shock, the investment-specific technology shock, and the price and wage mark-up shocks. It is also worth pointing out that the marginal prior densities of the financial-sector parameters are not visibly affected by the system prior information.

## 3.3 Posterior distributions

The posterior distributions of NAWM II have been estimated using the random-walk Metropolis sampler, taking into account the system priors. The estimation results are based on a Markov chain with 1,000,000 draws, with the first 500,000 draws being discarded as burn-in draws. The marginal posterior densities of the individual parameters are plotted against their marginal system prior densities in Appendix Figure A.2.

## Convergence

A standard approach to monitoring convergence of the posterior sampler is to employ multivariate analysis of variance (MANOVA). Based on the method discussed by Brooks and Gelman (1998), we have calculated the so-called multivariate potential scale reduction factor (MPSRF) that summarises the information about within-chain and between-chain variation for a number of parallel Markov chains. The factor is greater than unity for a finite number of posterior draws per chain and a finite number of chains. As the sampling algorithm converges, the factor declines towards 1, with values less than 1.1 being interpreted as signs of convergence. In the upper panel of Figure 3 we have plotted sequential estimates of the MPSRF based on five Markov chains (including the Markov chain underlying the estima-

tion results presented subsequently in this section) each having 1,000,000 draws with the first 500,000 draws being discarded as burn-in draws. The sequential estimates start with 10,000 post burn-in draws and are updated every 10,000 draws. After around 175,000 post burn-in draws we find that the MPSFR has dropped below 1.1.

In addition to monitoring the MPSRF, Brooks and Gelman (1998, p. 447) suggest to monitor the determinants of the within-chain covariance matrix and the pooled covariance matrix. These matrices are used in the computation of the MPSRF and monitoring their determinants allows also checking if they both stabilise as functions of the number of posterior draws. Since the determinants can assume very small values in models with a large number of parameters and where some are highly correlated, we have instead plotted the sequential estimates of the trace of these two matrices in the lower panel of Figure 3. If the posterior sampler has converged, they should have the same trace when the number of posterior draws has become sufficiently large. As can be seen from this panel, the sequential trace estimates indeed stabilise, with the trace of the pooled covariance matrix being slightly larger than the trace of the within covariance matrix.

## Financial-sector parameters

In Table 4, we first present estimation results for the parameters characterising the financial sector in NAWM II. The entries in the posterior-mode column refer to the values of the parameters that are obtained by maximising the model's joint posterior density. The remaining two columns report the 5% and 95% percentiles of the marginal posterior densities (see the plots in Appendix Figure A.2).

Comparing the posterior and prior densities of the financial-sector parameters gives an indication of how informative the observed data used in the estimation, including the additional financial data, are about the individual parameters. That is, in the case of those parameters for which the posterior deviates from the prior, the data are likely to be informative. Accordingly, our estimation results suggest that the observed data are generally informative about the financial-sector parameters. The posterior estimate of the relative absconding rate for domestic (foreign) government bonds, ω L (˜ ω ∗ L ), turns out to be somewhat higher (lower) than its prior. This finding implies a larger (smaller) relative 'riskiness' of domestic (foreign) government bonds, compared to what is implied by the choice of the prior mean on the basis of historical spreads (see Section 3.2.2). The posterior mode estimate of the households' portfolio adjustment cost parameter with respect to the holdings of domestic government bonds, γ h L , is 0.009, which is lower, but close to the prior mean of 0.01. Volume data on (sectoral) government bond holdings or bank lending, which

are currently not used in the estimation, could arguably be informative and help improve the estimation of this parameter. However, our posterior mode estimate for γ h L is very close to the posterior mode estimate of 0.007 reported in Darracq Pari` es and K¨ uhl (2016), even though this study uses data on bank loan volumes and, moreover, employs a prior with a higher mean and a larger standard deviation. 30

## Non-financial-sector parameters

In Tables 5 and 6, we report the posterior mode estimates of the parameters that are common to NAWM II and the original NAWM, or that relate to the other non-financial extensions of the original model, as outlined in Section 2.3. In particular, columns 3 to 5 in the two tables show the posterior mode estimates of the individual parameters as well as the 5% and 95% percentiles of the corresponding marginal posterior densities (see Appendix Figure A.2), whereas columns 6 and 7 show for comparison purposes the posterior mode estimates of the parameters of the original model as reported in CCW, Table 1, and when re-estimated using (per-capita) data until 2014Q4.

Overall, the posterior mode estimates of the parameters characterising households' preferences, wage and price-setting behaviour, final-good production, adjustment costs and interest-rate setting are found to be broadly similar across the two models. That is, the estimation of the common parameters appears rather robust to the extensions of the original model and to the inclusion of the additional variables in the set of observables used in the estimation. Nevertheless a number of estimation results are noteworthy. First, on the real side NAWM II appears to feature, ceteris paribus, a somewhat higher degree of intrinsic persistence when compared to the original model. The estimated habit formation parameter κ is noticeably larger, and even more so the investment adjustment cost parameter γ I . In the case of κ , the comparison with the re-estimated version of the original model suggests that this finding may be explained by using an extended sample. Second, the estimate of the Calvo parameter ξ H entering the reduced-form slope coefficient of the domestic price Phillips curve is quite a bit lower. This, however, reflects primarily the presence of the additional element of real rigidity represented by the calibrated Kimball parameter /epsilon1 H , which, ceteris paribus, lowers the slope coefficient. Taking into account the system prior, the estimate of the slope coefficient is actually 0 . 007 and, hence, only slightly higher than

30 We have conducted sensitivity analysis using the same prior specification as in Darracq Pari` es and K¨ uhl (2016), which resulted in a somewhat larger posterior mode estimate of 0.037. The findings of the quantitative analysis in this paper would not be notably different for the model's non-financial variables if this posterior mode estimate were to be used, except for the effects of shocks that directly affect the funding constraint of wholesale banks, such as shocks to the banks' survival rate or their asset holdings. Yet even for these shocks the differences in the effects remain contained.

the implied estimate of 0 . 005 for the original model. Third, the estimate of the parameter ˜ χ W , which determines the degree of indexation of wages to the model's permanent technology shock, is 0.66, compared to the implicitly calibrated value of unity in the original model. This implies a lower sensitivity of wages to shifts in trend productivity growth in the extended model. Fourth, different from the original model, the data seem to prefer a specification of NAWM II with variable capital utilisation, with the estimate of the adjustment cost parameter γ u, 2 being of moderate size. Fifth, all else equal, the estimates of the interest-rate rule parameters, which now account for a positive albeit small interest-rate response to the level of the output gap, imply a higher degree of interest-rate smoothing and a stronger response to inflation. Sixth, the estimates of the sensitivity parameters in the updating equations for the perceived inflation objective and the perceived growth rate of trend output are rather small, with values of 0.06 each. And finally, the estimated signal-to-noise ratio of the signal-extraction problem concerning the nature of the permanent technology shocks determining trend output is rather small, thereby attenuating the otherwise much stronger wealth effects on aggregate demand.

## 4 Model properties

In this section, we inspect the empirical properties of NAWM II by (i) examining the modelimplied sample moments of the observed variables used in its estimation, (ii) discussing selected impulse-response functions to shed light on the propagation of its structural shocks, with a focus on the new financial channels, (iii) interpreting the historical and the forecasterror-variance decomposition of per-capita real GDP growth in terms of structural shock contributions, and (iv) assessing the model's forecasting performance for a small set of the observed variables, namely per-capita real GDP growth, GDP deflator inflation and the short-term nominal interest-rate.

## 4.1 Sample moments

As the first element of our model inspection, Table 7 reports the sample means and standard deviations implied by NAWM II for the observed variables used in its estimation, along with the sample moments based on the observed data for the estimation sample 1985Q5-2014Q4. The model-based sample moments have been estimated through simulation. Specifically, for each one of the 5,000 draws of parameter values from the posterior distribution that we

use, 31 a sample of the observed variables is simulated by drawing shock innovations and measurement errors such that the number of simulated observations matches the number of observed data points. In doing so, the missing observations are taken into account. For each such sample of simulated data, the sample moments of the observable variables have thereafter been calculated. The table shows the averages of these moments as well as their 5% and 95% percentiles. In addition, Figure 4 displays model and data-based sample autocorrelations between real GDP and other observed variables that are endogenously determined within NAWM II. 32 In particular, the figure shows the averages of the modelbased sample autocorrelations and the 70% and 90% equal-tail credible intervals, along with the sample autocorrelations based on the observed data.

Regarding the sample means, we conclude from Table 7 that the NAWM II-based means are broadly in line with the data-based means. In particular, concerning the real variables the data means are close to the model-based means with the exception of long-term growth expectations. For the latter, the data mean is substantially higher than the model-based mean but nevertheless well below the 95% percentile. For the nominal variables the data means are greater than the model-based means albeit below the corresponding 95% percentiles, with the exception of the import deflator for which the data mean is smaller than the corresponding 5% percentile. Finally, the data means for the 10-year government bond yield and, to a lesser extent, for the short-term nominal interest rate are close to the modelbased means, whereas the data mean for the composite long-term lending rate is smaller than the corresponding 5% percentile.

Turning to the sample standard deviations, we find for the real variables that real GDP, consumption, investment, exports, imports and the output gap have a data-based volatility measure that is below the 5% percentile generated by simulating NAWM II. Among the nominal variables, we likewise obtain data-based sample standard deviations less than the model-based 5% percentiles for GDP deflator, consumption deflator and wage inflation, as well as for long-term inflation expectations. Among the remaining seven variables only the data-based sample standard deviation of the 10-year government bond yield is above the model-based 95% percentile. All in all, the use of the system prior for the population covariance matrix of the observed variables (see Section 3.2.2) has been an important factor in

31 The 5,000 posterior parameter draws are taken from the 500,000 available post-burn-in draws as draw number 1, 101, 201, etc. Compared to using all 500,000 draws, the selected sub-sample has the advantage of combining moderate computational costs with a lower autocorrelation among the draws. At the same time, the length of the sub-sample retains a sufficiently high estimation accuracy.

32 We have only excluded the SPF-based long-term growth and inflation expectations from the figure for the sake of limiting its size to one page.

limiting the discrepancies between the model and data-based sample standard deviations, most notably for the observed real variables. Likewise the system prior for the slope coefficient of the model's domestic price Phillips curved helped containing the implied standard deviations for the observed nominal variables.

As to the dynamics between real GDP and 15 of the other endogenous observed variables, as captured by their autocorrelations, Figure 4 shows that the data-based correlations are generally inside the 90% equal-tail credible intervals of the model-based correlations. At the same time, it is noteworthy that the data-based autocorrelations with respect to both GDP and consumption deflator inflation are greater than the 95% percentiles of the model-based correlations for some leads and lags around zero. Nevertheless, the overall findings suggest that NAWM II is able to match reasonably well the fluctuations in the observations of the endogenous variables over the sample period. Moreover, these results are broadly consistent with the findings reported in CCW for the original NAWM.

## 4.2 Propagation of structural shocks

To portray key features of the propagation mechanisms of NAWM II, Figures 5 to 10 show the model's impulse responses to a subset of its structural shocks. This subset comprises a shock to the short-term nominal interest rate (i.e., a standard monetary policy shock), three financial shocks-a shock to the survival rate of the wholesale banks in the model, a shock to the market power of the retail banks, and a domestic risk premium shock-and, finally, a shock to the persistent component of the permanent technology shock (i.e., a shock to the growth rate of trend labour productivity). Unless otherwise stated, the figures depict the means and the 70% and 90% equal-tail credible intervals of the impulse responses to shocks equal to one standard deviation for the main observed variables that are endogenously determined within NAWM II and for two unobserved state variables, namely the net worth of the wholesale banks and the Lagrange multiplier for the households' Loan-in-Advance (LIA) constraint. These state variables play a central role in the model's propagation mechanism via financial channels. Like the sample moments, the credible intervals are based on 5,000 draws from the posterior distribution of the parameters. Where available, the figures also display the mean of the impulse responses for the original NAWM, as presented in CCW. All impulse responses are reported as percentage deviations from the model's non-stochastic balanced-growth path, except for those of inflation and interest rates which are reported as annualised percentage-point deviations.

## Shock to the short-term nominal interest rate

In Figure 5, we depict impulse responses to a shock to the short-term nominal interest rate. The temporary increase in the short-term rate leads to a hump-shaped decline in domestic demand, with investment falling to a considerably larger extent than private consumption. Reflecting the enhanced transmission mechanism of NAWM II, the increase in the shortterm interest rate is propagated to households' investment decisions through a rise in the long-term wholesale lending rate (which is not shown in the figure but closely linked to the government bond yield via a no-arbitrage condition) and its gradual pass-through to the long-term retail lending rate for investment loans. The demand for domestic goods is further dampened by an appreciation of the domestic currency which leads, via a deterioration of the terms of trade, to a switch of domestic and foreign demand towards goods produced abroad. In response to the broad-based decline in aggregate demand for domestic goods, firms reduce their demand for labour. In equilibrium, both, employment and wages fall. The fall in wages contributes to a decline in firms' marginal cost which, in turn, puts downward pressure on domestic prices. 33 The decrease in domestic prices, together with the drop in import prices triggered by the appreciation of the domestic currency, leads to a decline in consumer prices. The trough in economic activity and inflation is reached after about one year. Interestingly, the real effects are somewhat stronger in NAWM II than in the original NAWM, while the impact on inflation seems marginally weaker once accounting for the differential exchange-rate effect. This finding is consistent with empirical evidence that inflation has become less sensitive to fluctuations in economic activity in recent years. Finally, it is noteworthy that the responsiveness of both exports and imports is markedly higher in NAWM II because of the non-zero export content of imports.

Figure 6 sheds light on the role of the financial frictions, and the implied 'financial accelerator' mechanism, in the propagation of the interest-rate shock in NAWM II. To this end, the impulse responses of the standard version of the model are compared to the responses of a version where the financial frictions are shut off. For computing the two sets of impulse responses, the posterior mode estimates for the parameters of the standard model are used. 34 As the central bank raises the short-term interest rate, the present value

33 In addition, firms utilise fewer capital services due to both a gradual decline in the physical capital stock and a downward adjustment in the capital utilisation rate. The ensuing fall in the rental rate of capital amplifies the negative impact on marginal cost.

34 To shut off the financial frictions, the households' portfolio adjustment cost parameter γ h L and the Calvo parameter ξ I governing the loan pricing of the retail banks are both set to (values close to) zero. Without portfolio adjustment costs, the expected excess return on long-term bonds over the deposit rate is zero to rule out any arbitrage by the households. Similarly, the excess returns on investment bonds and foreign bonds also stay zero by the no-arbitrage conditions of the wholesale bank. The zero Calvo parameter of the

of the long-term fixed-coupon assets (i.e., investment loans and government bonds) in the portfolio of the wholesale banks in the model declines. This triggers a drop in banks' net worth, and the agency friction brings about an accompanying withdrawal of deposits. The ensuing scarcity of funding forces the wholesale banks to tighten credit supply, which leads to curtailed wholesale lending and elevated lending rates on wholesale investment loans. Higher wholesale lending rates pass through sluggishly to retail lending rates, generating a hump-shaped response of the latter. The evolution of the tightness of credit conditions for new investment is best represented by the multiplier of the households' LIA constraint. Consistent with the sluggish increase of retail lending rates and the declining credit demand due to the downturn, the multiplier eases initially, but it increases persistently in the medium term signalling a deterioration in lending conditions. This leads to an amplification of the decline in investment in the standard version of NAWM II relative to the version without financial frictions.

Two adjustment channels attenuate the strength of the financial accelerator mechanism amplifying the propagation of the interest-rate shock. The first is related to the availability of long-term government bonds and the households' ability to invest directly into these bonds. The expected excess return on government bonds increases in parallel with the excess return on investment bonds to ensure the absence of arbitrage opportunities for the wholesale banks. This requires an immediate drop in the price of long-term government bonds, which can further exacerbate the capital losses of wholesale banks. At the same time, and mirroring the decline in bond prices, yields on long-term government bonds go up. The higher expected return on government bonds induces households to increase their long-term government bond holdings through purchasing bonds from the wholesale banks. This mitigates the tightening of the wholesale banks' capacity to issue investment loans and partially offsets the negative impact of the initial shock. The second effect, which also dampens the accelerator mechanism, comes from the enhanced future profitability of the wholesale banks due to higher excess returns, which allows them to maintain a higher riskweighted leverage without violating their incentive constraint. The higher leverage eases their balance sheet constraint ceteris paribus, but it is insufficient to offset the impact of the much larger drop in their net worth.

Finally, the financial frictions also bear international ramifications. In order for the wholesale banks' no-arbitrage conditions to hold, the increase in the expected excess return

retail banks ensures that the changes in wholesale lending rates are passed through to retail lending rates immediately. These conditions jointly imply that the relevant cost of borrowing is the deposit rate as in an environment without financial frictions.

on domestic long-term government bonds has to be accompanied by an increase in the expected excess return on foreign long-term government bonds. For this to happen, agents have to expect a future depreciation of the domestic currency. Consequently, the initial appreciation of the exchange rate gets amplified in the standard version of NAWM II relative to the version without financial frictions.

## Shock to the survival rate of wholesale banks

Figure 7 shows impulse responses to a shock to the survival rate of the wholesale banks in NAWM II. The temporary increase in the survival rate leads to a transitory increase in wholesale banks' aggregate net worth since fewer existing banks, which have accumulated retained earnings in the past, leave the financial sector. The shock, therefore, can be interpreted also as a temporary drop in banks' propensity to pay dividends. The increase in net worth improves their leverage ratio and allows for an easing of lending conditions. Both the yield on long-term government bonds and the rate on long-term wholesale loans decline, the latter being only gradually passed through to the long-term retail lending rate, which hence displays a hump-shaped pattern. In turn, private investment is stimulated and triggers an inflationary increase in aggregate demand, despite a temporary crowding out of private consumption. The central bank, following its interest-rate feedback rule, raises the short-term nominal interest rate, thereby curtailing private demand and mitigating the increase in overall economic activity and inflation.

## Shock to the mark-down parameter of retail banks

Figure 8 shows impulse responses to a shock to the mark-down parameter of the retail banks. The temporary fall in the market power of the retail banks leads to a hump-shaped decline in the long-term retail lending rate. Private investment is stimulated and boosts aggregate demand, while private consumption is crowded out over the shorter term. The central bank responds by raising the short-term interest rate, thereby dampening the stimulative effect from lower retail lending rates. At the same time, and on the back of an equally sized increase in the rate on deposits placed by households, the implied increase in the refinancing costs of the wholesale banks triggers a decline in the price of their long-term assets. The fall in asset prices in turn leads to a decline in their net worth, impairing the scope for issuing new investment loans. In sum, while the retail price of investment loans increases on account of the fall in retail banks' market power, both the wholesale price of loans and the price of long-term government bonds decrease. Mirroring the decrease in the price of government bonds, the yield on government bonds rises.

## Shock to the domestic risk premium

Figure 9 shows impulse responses to a domestic risk premium shock. The shock drives a wedge between the interest rate on households' deposits with wholesale banks and, in equal measure, between the one-period domestic government bond yield and the short-term nominal interest rate controlled by the central bank. The increase in the wedge leads to an inter-temporal reallocation of private consumption expenditures towards the future, similar to the effects of a shock to the short-term policy rate (see Figure 5). At the same time, the present discounted value of future income from renting capital services to firms declines and consequently the shadow price of a unit of the investment good falls, resulting in a reduction of investment. Both, the decline in consumption and the drop in investment causes aggregate output to fall. Accordingly, firms reduce their demand for labour, thereby putting downward pressure on wages. In equilibrium, both, employment and wages fall. The decline in wages contributes to a decline in firms' marginal cost which in turn puts downward pressure on domestic prices. The no-arbitrage condition between domestic and foreign long-term bonds induces an appreciation of the domestic currency that curtails exports and leads foreign firms to lower the price of goods sold domestically. The decrease of domestic prices and import prices, in turn, induces a decline in consumer prices. The central bank responds to the decline in aggregate output and inflation by lowering the shortterm interest rate, but the implied degree of policy accommodation is insufficient to fully offset the contractionary effect of the domestic risk premium shock. Like for the interestrate shock, the individual impulse responses across NAWM II and the NAWM are very similar, with the responses being somewhat stronger and more drawn out for the former, primarily because of the higher estimated degree of persistence of the shock process.

## Shock to the persistent component of the permanent technology shock

Figure 10 shows impulse responses to an increase in the persistent component of the permanent technology shock. Whereas agents observe the realisation and know the persistent nature of the permanent technology shock in the original NAWM, agents in NAWM II only observe the realisation of the composite of its persistent and transitory components and need to solve a signal-extraction problem to infer the nature of the particular component driving the observed realisation of the composite shock. In the original model, forward-looking households anticipate a persistent rise in their future income following the permanent technology shock and, hence, they already increase the current levels of their consumption and investment expenditures. While transitory technology shocks are typically associated with a temporary decline in firms' labour demand and a muted response

of wages, the strong demand effect triggered by the persistent increase in the permanent technology shock is sufficiently stimulative to push labour demand and wages up. Firms' marginal cost of production eventually increases giving rise to upward rather than downward pressure on domestic goods prices. The central bank raises the short-term interest rate, but the monetary policy tightening is not aggressive enough to offset the expansionary and inflationary effect of the shock. In NAWM II, the responses are quantitatively much more muted, reflecting the fact that agents initially do not know whether they are witnessing a persistent or a one-off increase in the permanent technology shock. Hence, expected future income increases by less than in the original model, resulting in a more attenuated increase in households' expenditures. Moreover, the inflation response is very subdued, also on account of the merely partial indexation of wages to the permanent technology shock further moderating the wage response in the shorter term.

## 4.3 Historical decomposition of real GDP growth

Historical decompositions are widely used to identify the drivers of observed economic developments through the lens of a structural model in terms of the contributions of its estimated shocks. In accordance with this practice, we examine the contributions of the structural shocks obtained from NAWM II to the observed fluctuations in per-capita real GDPgrowth. 35 In order to facilitate the presentation of the shock contributions, we focus on the historical period from 2000 to 2014 and bundle the structural shocks into seven groups: technology shocks, demand shocks, mark-up shocks, financial shocks, foreign shocks and perception shocks, plus a monetary policy shock. 36 The technology shock group comprises the persistent and the transitory component of the model's permanent technology shock (which are not separately observed by the agents in the model and estimated by solving a signal-extraction problem), the transitory technology shock and the investment-specific technology shock. The demand shock group includes the shocks to government consumption, to import demand and to export demand preferences, while the mark-up shock group consists of the shocks to the wage mark-up, the domestic price and export price mark-ups, as well as the import price mark-up. The financial shock group comprises the domestic and external risk premium shocks, the shock to the survival rate of the wholesale banks

35 The smoothed estimates of the model's structural shocks and of their standardised innovation components are shown in Appendix Figures A.3 and A.4, respectively.

36 For the sake of convenience, we do not report the small contribution of the measurement errors accounted for in the estimation of the model and the fading effect of the estimated initial conditions for its unobserved state variables.

and the shock to the mark-down parameter of the retail banks. The foreign shock group assembles the innovations in the SVAR model of the foreign variables, while the perception shocks correspond to the innovations in the updating equations for the perceived inflation objective of the central bank and the perceived trend growth rate. Finally, the monetary policy shock represents unanticipated deviations of the short-term nominal interest rate from the prescriptions of the central bank's interest-rate feedback rule. 37

The historical decomposition of per-capita real GDP growth depicted in the upper panel of Figure 11 (in terms of annual growth rates and in deviation from the model-implied mean growth rate of 1.5%) shows that the growth acceleration both in the early years of the period under consideration and during the run-up to the financial crisis of 2008/09 is explained to a considerable extent by positive contributions of financial shocks. These positive contributions are expression of the very favourable financing conditions at the time, fostering domestic demand. In both sub-periods, the financial shocks more than offset the persistently negative contribution from adverse technology shocks, on the back of a positive contribution from (wage) mark-up shocks reflecting a protracted period of wage moderation. According to the decomposition, the episode of subdued growth in the interim period from 2002 to 2004 is primarily attributed to the waning of favourable financing conditions, as captured by a reversal in the previously positive contribution from financial shocks. This reversal was arguably triggered by the global asset market disruptions around the year 2001. Throughout this interim period, favourable monetary policy shocks (i.e., a looser stance than suggested by the interest-rate rule) supported domestic demand and prevented a stronger slowing of real GDP growth.

The sharp and protracted decline in real GDP growth caused by the unfolding of the financial crisis in 2008 is attributed primarily, and in roughly equal proportions, to a large negative contribution of foreign shocks, capturing the dramatic fall in world trade and the implied contraction in euro area foreign demand, and a large negative contribution from financial shocks. The latter contribution arguably reflects the sudden freezing of financial markets which gave rise to a surge in risk premia. The protracted slump in GDP growth is accompanied by a lasting weakness in investment and a slowing of trend growth, being associated with an increasing negative contribution of technology shocks. Among the individual technology shocks (not reported), investment-specific technology shocks and

37 Note that the grouping of the shocks differs from the shock grouping for the original NAWM in CCW. Specifically, the domestic and external risk premium shocks have been moved from the demand shock group and the foreign shock group, respectively, to the group of financial shocks, whereas the export preference shock has been moved from the foreign shock group to the demand shock group. Similarly, the import price mark-up shock has been moved from the foreign shock to the mark-up shock group.

permanent technology shocks are particularly important. The subsequent recovery in GDP growth in late 2009 is explained by a marked turnaround in the beforehand negative contributions of the foreign and financial shocks, reflecting the stabilisation of both the world economy and financial markets. At the same time, price and wage rigidities hampered the recovery, as reflected in the negative contribution of the mark-up shocks. While the decomposition suggests that monetary policy shocks contributed positively to the recovery, this contribution is found to be negative in the immediate aftermath of the crisis despite the fact that the ECB, like other major central banks around the world, rapidly reduced its key interest rates to historically low levels in order to bolster domestic demand and stabilise financial markets. Clearly, when interpreting this finding due account should be taken of the fact that the ECB also swiftly implemented a number of non-standard monetary policy measures, including the provision of unlimited liquidity to the banking system, in order to sustain financial intermediation and to maintain the availability of credit to the private sector (see ECB 2010). The effects of these non-standard measures are not directly captured by the model-based contribution analysis, but only indirectly and, arguably, via positive contributions of the financial shocks.

The setback in the recovery in the course of 2010 and 2011-owing to the reintensification of the financial crisis on account of elevated tensions in euro-area sovereign debt markets-is explained to a great extent by the waning of the previously positive contribution from financial shocks (tantamount to a fresh surge in risk premia), in conjunction with a renewed increase in the negative contribution of technology shocks against the backdrop of a further weakening of investment and a resumed slowing of trend growth. Similarly, insufficient downward adjustment of prices and wages continued to weigh adversely on GDP growth, as indicated by the protracted negative contribution of the mark-up shocks.

The decomposition of the contribution of the financial shock group in the lower panel of Figure 11 reveals that, amongst the individual financial shocks, the domestic risk premium shock explains, by and large, the greater part of the historical fluctuations in real GDP growth. This feature reflects the fact that, when compared to the other financial shocks, the domestic risk premium shock is the only shock that generates a degree of co-movement amongst the different demand components of real GDP which is commensurate with the strong co-movement in the observed data (see the corresponding impulse-response functions in Figures 7, 8 and 9). Nevertheless, the shock to the survival rate of the model's wholesale banks is a significant factor in explaining the acceleration of GDP growth both in the run-up to the financial crisis of 2008 and during the recovery following the crisis on the back of a

declining government bond yield spread (recalling that the long-term government bond yield in the model is matched to the euro area 10-year AAA bond yield). In contrast, the shock to the mark-down parameter of the model's retail banks makes a more nuanced contribution to GDP growth. In particular, its contribution is negative prior to and during the unfolding of the financial crisis on account of a widening spread between the long-term (euro areawide composite) lending rate and the long-term government bond yield, turns positive in the recovery (arguably reflecting the positive effects of the ECB's liquidity measures aimed at sustaining longer-term bank funding), and is persistently negative during the evolving sovereign debt crisis owing to the renewed rise in the lending-rate spread. 38

All in all, this section demonstrates that the contribution analysis based on the shocks identified through the lens of NAWM II can guide the development of a compelling narrative for the fluctuations in real GDP growth observed in the past, including the crisis years. Yet given the particular choice of observables for the government bond yield and the lending rate, and without a proper model mechanism that separates risks emerging in the government bond market from risks in the financial intermediary sector, or that accounts for the inherently non-linear nexus between these two types of risk, 39 the shock contributions identified for the different phases of the crisis need to be interpreted with due caution.

## 4.4 Forecast-error-variance decomposition for real GDP growth

Forecast-error-variance decompositions allow to combine, on the one hand, the statistical approach employed in the comparison of the simulation-based sample moments of NAWM II with those based on the observed data and, on the other hand, the structural approach to identifying the contributions of its estimated shocks to fluctuations in the observed data by means of historical decompositions. In particular, we can exploit the Bayesian setting of NAWM II to construct credible sets for the contributions of its shocks to the implied forecast error variances of the observed variables at different horizons. As an example, Table 8 reports posterior mean estimates and 90% equal-tail credible intervals for the shock contributions to the forecast error variance of per-capita real GDP growth in the short

38 In this context it is noteworthy that a shock to the mark-down parameter of the retail banks in the model, which brings about a fall in the lending rate, results in a marked increase in the government bond yield (see the impulse responses in Figure 8 and the accompanying explanations). This negative conditional correlation between the lending rate and the government bond yield is a critical factor in explaining developments in the lending-rate spread and hence for the identification of the individual financial shocks.

39 In this respect see, for example, Bocola (2016), who examines the macroeconomic implications of sovereign credit risk in a model where banks are exposed to domestic government debt. The model features time-variation in risk premia and occasionally binding financial constraints and therefore necessitates the use of non-linear solution methods which are not yet applicable to large-scale models such as NAWM II.

run (4-quarter horizon) and in the long run, based on 5,000 draws from the posterior distribution of the model's structural parameters. For the sake of simplifying the exposition, the structural shocks have again been bundled into seven groups: technology shocks, demand shocks, mark-up shocks, foreign shocks, financial shocks and perception shocks, plus a monetary policy shock.

In the short run, the observed fluctuations of real GDP growth are primarily driven by technology and financial shocks, with shares of 28% and 22%, respectively. Among the individual financial shocks, the domestic risk premium shock explains most of the forecast error variance (18%), while the shocks to the survival rate of wholesale banks and to the mark-down parameter of retail banks are less important (3% and 1%, respectively). 40 Concerning the individual technology shocks (not reported), the share of the investmentspecific technology shock is particularly large, while the share of the two components of the permanent technology shock is smaller. Moreover, the demand and the mark-up shock groups explain 12% and 14% of the variance, while the shares due to monetary policy and foreign shocks are modest.

In the long run, the forecast error variance of real GDP growth is equal to its variance. It is interesting to note how small the differences in the shares are when compared with the short-run shares. The main difference is that the structural shocks account for 99% of the variance in the long run, with the remaining 1% being due to measurement errors, rather than 90% as is recorded for the short run. Furthermore, the importance of technology shocks increases somewhat to roughly one-third of the variance, and investment-specific shocks remain the main contributor at 21%, while the two permanent technology shock components account for approximately 8%.

## 4.5 Evaluation of real GDP, inflation and interest-rate forecasts

As the last element of our model inspection, we conduct an evaluation of the relative forecasting performance of NAWM II for a small set of its observed variables over the forecast sample period 2006Q1-2014Q4. To this end, we compute root-mean-squared errors (RMSEs) of one to eight-quarter-ahead point forecasts for annual per-capita real GDP growth, annual GDP deflator inflation and the annualised short-term nominal interest rate. The forecasts of NAWM II are compared to forecasts from an updated version of the original NAWM and two na¨ ıve forecast benchmarks. The updated NAWM takes into account the

40 Like for the historical decomposition of real GDP growth discussed above, this pattern of relative importance is rooted in the high degree of co-movement amongst the demand components generated by the domestic risk premium shock.

use of per-capita data and has been estimated on the same data sample as NAWM II. The na¨ ıve forecasts are given by the no-change/random-walk assumption (according to which the last pre-forecast sample observation represents the forecast over the full forecast horizon) and the pre-forecast sample mean (corresponding to a random walk with drift for the level of real GDP).

The point forecasts are made out-of-sample with the end of the estimation sample being gradually extended from 2005Q4 to 2013Q4. For NAWM II and the updated NAWM, the point forecasts for computing the RMSEs are given by the mean forecasts conditional on the parameters being evaluated at the posterior mode. Both models are re-estimated in the fourth quarter of each calendar year, which is a reasonable representation of how often such medium-scale DSGE models are re-estimated at policy institutions. 41

The mean forecast paths underlying the computation of the RMSEs, also known as spaghetti plots , are displayed in Figure 12. Concerning the real GDP growth forecasts in Panel A, we find that the paths look broadly similar for NAWM II and the NAWM. Like most macroeconomic forecasts at the time, both models substantially overpredict real GDP growth during the unfolding of the financial crisis in 2008/09 and in the wake of its reintensification in 2012. At the same time, the NAWM has a clear tendency to underpredict GDP growth over the years prior to the crisis and during the short-lived recovery in the interim period from 2010 to 2012. 42 Turning to the forecast paths for GDP deflator inflation in Panel B, the behaviour of the two models is quite different. Overall, the paths for NAWM II are downward-sloping prior to the crisis and upward-sloping thereafter, while those for the NAWM are typically hump shaped. In Panel C, we depict the forecast paths for the short-term interest rate. It appears that the paths from NAWM II lie closer to the actual data than those from the NAWM, especially prior to the period of very low interest rates starting towards the end of 2012. During the low interest-rate period, both models strongly overpredict actual interest-rate developments, thereby exposing the limitations of the simple interest-rate feedback rules assumed in the models.

In Figure 13, we have plotted the RMSEs of the model-based point forecasts relative to the corresponding pre-forecast sample mean. Hence, values below (above) unity indicate a better (worse) point forecasting performance than this na¨ ıve benchmark. Regarding the real GDPgrowth forecasts, we observe that NAWM II has slightly smaller RMSEs for all forecast

41 The recursively computed posterior mode estimates for the parameters of NAWM II are shown in Appendix Figure A.5.

42 It is interesting to observe that the share of the squared mean errors in the overall mean squared errors for real GDP growth is greater for NAWM II than for the NAWM.

horizons up to six quarters ahead when compared to the NAWM, while the random walk is out-performed at all horizons. Interestingly, at the outer forecast horizons the sample mean is competitive with both NAWM II and the NAWM, with the corresponding relative RMSEs being close to unity. As regards the forecasts of GDP deflator inflation, NAWM II performs again slightly better than the NAWM for the forecasts up to four quarters ahead, while the situation is reversed for the longer horizons. The random walk-based forecasts, however, are generally found to have smaller RMSEs, except for the two-year horizon. Compared to the pre-sample forecast mean, all model-based forecasts do better, with their relative RMSEs lying markedly below unity. 43 For the short-term nominal interest rate, NAWM II yields somewhat better point forecasts than the NAWM and the random walk, for which we have obtained broadly similar RMSEs. Like in the case of GDP deflator inflation, the relative RMSEs of the model-based forecasts are substantially lower than unity.

All in all, on the basis of our limited forecast evaluation exercise, the comparative evidence suggests that NAWM II fares quite well relative to the updated version of the original NAWM and the two na¨ ıve benchmarks.

## 5 Applications

We finally consider two applications to illustrate the potential contributions that NAWM II can make to conducting counterfactual policy analysis at the ECB. 44 The first application shows how the model can be used to assess the macroeconomic impact of large-scale central bank asset purchases which were implemented by major central banks, including the ECB, in the aftermath of the financial crisis once policy rates had reached their effective lower bound. In this context, we highlight the role of two critical modelling assumptions that influence the overall impact of asset purchases: the central bank's promise to keep the policy rate unchanged over a number of quarters-that is, the use of forward guidance on interest-rate policy -and the mechanism underlying the determination of the exchange rate and, thus, the strength of the exchange-rate channel of asset purchases . In the second application we give an example of how the model can be used in a real-time setting to analyse the adverse impact of a possible de-anchoring of longer-term inflation expectations

43 As pointed out by CCW, the good performance of the random walk-based forecasts can be explained by the fact that inflation has been relatively stable during the EMU period. By contrast, the bad performance of the sample mean reflects the protracted period of higher average inflation rates in the pre-EMU period.

44 For an exemplification of the basic need in policy analysis to address questions of a counterfactual nature and of the comparative advantage of structural models, notably DSGE models, to satisfy this need, see Coenen, Motto, Rostagno, Schmidt, and Smets (2017).

on the macro economy in an environment where the lower bound on the short-term interest rate is binding. From a methodological perspective, this application illustrates how existing DSGE models can be modified to allow for both small deviations from rational expectations and non-linear constraints on interest rates.

## 5.1 Macroeconomic effects of central bank asset purchases

This section illustrates how NAWM II can be used to assess the macroeconomic impact of central bank asset purchases by means of counterfactual simulations. The simulations are designed to capture some of the key features of the ECB's expanded asset purchase programme (EAPP) as announced in January 2015. In the simulations, the total amount of asset purchases equals 11% of nominal GDP. The share of long-term domestic government bonds in the purchases is assumed to equal 20% and the share of long-term investment loans is assumed to equal 80%. This composition aims to account for the fact that, in the estimation of NAWM II, the observed variable corresponding to the long-term domestic government bond yield only comprises the 10-year government bond yield series of sovereign issuers whose rating is AAA at a given point in time. The share of 20% government bond holdings is broadly consistent with the share of AAA government bond purchases as a share of all purchases (70% government bond purchases, and a 30% capital key for Germany, the Netherlands and Luxembourg among the euro area countries). We consider sovereign issuers eligible for the EAPP with an investment grade rating below AAA as well as corporate bond purchases as investment loans in the simulations. We furthermore assume that the stock of purchased assets follows a hump-shaped path, peaking 2 years after the start of the programme and with a gradual reduction thereafter as the purchased assets mature. This pattern is approximated with a second-order autoregressive process with parameters set equal to 1.7 and -0.71.

Figure 14 shows the paths of key model variables for a benchmark variant of the asset purchase simulation under two alternative assumptions about interest-rate policy. The solid blue lines depict the variable paths when the short-term nominal interest rate is assumed to be unconstrained and set according to the model's interest-rate feedback rule. These paths are reported in the form of percentage deviations from the model's non-stochastic balancedgrowth path, except for the paths of the inflation and interest rates which are reported as annualised percentage-point deviations. All in all, the model-based benchmark simulation suggests that the EAPP is effective because it improves credit conditions. In particular, the central bank asset purchases, the evolution of which is shown in the upper-left panel

of the figure, remove assets from the wholesale banks' balance sheets and create excess balance sheet capacity that banks can use to extend new credit to the private sector. As a consequence, lending conditions improve and stimulate private investment on the back of a lower lending rate. Concomitantly, the lending rate spread and the expected excess returns on government bonds fall and asset valuations rise. This generates windfall gains for the wholesale banks, raising their net worth, and allows them to loosen credit conditions further in a positive feedback loop. The ensuing increase in aggregate demand puts upward pressure on firms' marginal cost of production and leads to a rise in domestic prices. The decline in expected excess returns on domestic long-term government bonds goes hand in hand with a decrease in expected excess returns on foreign long-term government bonds, which is brought about by an instantaneous depreciation of the domestic currency, boosting exports. Since the evolution of the short-term nominal interest rate is governed by a standard feedback rule, the central bank partly counteracts the increase in inflation and economic activity by raising the short-term nominal rate. The monetary policy tightening translates into a rise in the short-term real interest rate and dampens private domestic demand, leading to a temporary decline in private consumption.

The blue dashed lines in Figure 14 depict the variable paths for the benchmark variant of the simulation when the central bank promises to keep the short-term nominal interest rate unchanged for 8 quarters of the simulation. The 8-quarter interest-rate peg coincides with the period over which net central bank asset purchases are positive. This simulation is thus tantamount to a scenario in which the policy rate is temporarily constrained by its lower bound and the central bank employs asset purchases and forward guidance on future interest-rate policy to stimulate the economy. In principle, this type of simulation is prone to the so-called forward guidance puzzle (see Del Negro, Giannoni, and Patterson 2012, and Carlstrom, Fuerst, and Paustian 2015). That is, in standard DSGE models credible nonstate-contingent announcements about future interest-rate policy can have unrealistically large macroeconomic effects. To attenuate the forward guidance puzzle in our alternative benchmark simulation, we assume that only a fraction of households and firms believe in the central bank's announcement about unchanged future interest rates. 45 In the absence of the immediate interest-rate hike observed in the simulation with an unconstrained nominal interest rate, the nominal and real effects of the central bank asset purchases get elevated.

45 This approach of modelling limited credibility of policy announcements follows Coenen and Wieland (2004). To conduct the asset-purchase simulation with the imperfectly credible temporary interest-rate peg, we make use of the code developed by Montes-Gald´ on (2018). In the simulation shown, it is assumed that only 50% of households and firms find the interest-rate announcement credible. The remaining households and firms do observe the short-term interest rate set in the current period but build expectations on the assumption that in future periods the central bank follows the standard interest-rate rule.

Both real GDP and inflation rates increase more strongly, albeit still moderately, and the currency depreciation is larger than in the benchmark simulation without the temporary interest-rate peg.

Figure 15 compares the benchmark variant of the asset purchase simulation to an alternative variant where the riskiness of the long-term investment bonds has been increased, reflecting an environment with a more fragile financial intermediary sector. 46 To focus on the role of the riskiness of assets as such, the short-term nominal interest rate is assumed to be unconstrained in both simulations. In the alternative simulation with heightened riskiness of investment bonds (red dashed lines), the overall absorption of risk from banks' balance sheets through the asset purchases is significantly higher than in the benchmark simulation (blue solid lines). The decline in the long-term lending rate therefore gets amplified, boosting the rise in investment and real GDP.

Figure 16 sheds light on the role that the modelling assumptions concerning international financial markets have on the exchange-rate channel of central bank asset purchases. The simulation of the standard version of NAWM II with an unconstrained short-term nominal interest rate (blue solid lines) is compared to a version where the foreign long-term government bonds are replaced by foreign one-period bonds which can be traded directly by domestic households (red dashed lines). In this alternative model version, the standard uncovered interest parity condition (UIP) holds. For the counterfactual asset purchase simulation, the path of the exchange rate differs across the two versions of the model. Whereas the exchange rate depreciates on impact in the standard NAWM II, as discussed above, there is almost no instantaneous response of the exchange rate in the alternative version of the model. The lack of an immediate exchange-rate depreciation also translates into a more muted rise in consumer prices in the alternative model. At the same time, in the absence of an exchange-rate depreciation net exports initially decline, rather than rise, mitigating the amplification in the response of real GDP all else equal. Reflecting less severe inflationary pressures, the central bank raises the short-term nominal interest rate by less than in the standard version, thereby attenuating the decline in private consumption and exacerbating the rise in investment.

The results of the benchmark and variant simulations of the asset purchase counterfactual for real GDP growth and consumer price inflation are summarised in Table 9. In the benchmark simulation with an 8-quarter interest-rate peg, average annual real GDP growth increases by 0.34 percentage point at the peak, and average annual inflation by 0.19

46 In this alternative simulation, the annualised steady-state spread of the retail lending rate over the deposit rate was raised by 64 basis points compared to the benchmark simulation.

percentage point. These results are broadly consistent with other quantitative model-based assessments of the macroeconomic effects of central bank asset purchases for the euro area, albeit at the lower end of the range of results. For example, Burlon, Gerali, Notarpietro, and Pisani (2017) use a calibrated open-economy New Keynesian model of the euro area with preferred habitat motive as in Chen, C´ urdia, and Ferrero (2012) to compare alternative simulations of the EAPP. In their benchmark simulation with an 8-quarter interest-rate peg, inflation increases by up to 0.8 percentage point in annualised terms and annualised real GDP growth by about 1 percentage point. The inflation effects are more sizeable than in the simulations based on NAWM II, most likely because Burlon et al. (2017) assume that the central bank's announcement to keep the short-term nominal interest rate constant is perfectly credible. Andrade, Breckenfelder, De Fiore, Karadi, and Tristani (2016) use a modified version of the New Keynesian model developed in Gertler and Karadi (2013) to simulate the effects of the EAPP in a scenario with an endogenous lower bound constraint. In their benchmark simulation, the central bank asset purchases increase annualised inflation by up to 0.4 percentage point and the level of real GDP by 1.1%.

Finally, the quantitative impact on the 10-year government bond yield in the modelbased asset purchase simulations is around 10 basis points. This is in the ballpark of the effects of the EAPP announcement of 22 January 2015 on 10-year government bond yields found in event studies. Altavilla, Brugnolini, G¨ urkaynak, Motto, and Ragusa (2019), for example, identify effects between 13 to 15 basis points for the 10-year German, French, Italian and Spanish government bonds. As the announcement came only partly as a surprise, the overall yield impact of the programme is likely to be somewhat higher; see the metaanalysis of the literature in Andrade et al. (2016). 47 Thus we conclude that our estimates based on NAWM II are at the low end, but within the range, of the available empirical findings concerning bond-yield reactions to large-scale asset purchases.

## 5.2 Consequences of a de-anchoring of longer-term inflation expectations

Longer-term inflation expectations are generally seen to be an indicator of the credibility of central banks in achieving their inflation objectives and should, therefore, remain solidly 'anchored'. In this section, we employ NAWM II to conduct a counterfactual simulation with the aim of illustrating the macroeconomic consequences of a decline in longer-term inflation expectations, over and above the moderate fall in inflation expectations observed on

47 Additional sensitivity analysis, which is not reported in the paper, shows that, all else equal, a higher pace of asset purchases, commensurate with a less persistent process for the EAPP shock, results in a stronger initial bond-yield reaction.

the basis of the ECB's Survey of Professional Forecasters (SPF) during the year 2014 (see the lower-left panel in Figure 2). 48 Such a decline in expectations may reflect a growing privatesector concern that the central bank's ability commitment to achieving, and the ability to achieve, its inflation objective over the medium term has weakened against the backdrop of the prolonged period of low inflation in the aftermath of the financial crisis. The deanchoring of longer-term expectations is modelled within NAWM II through a gradual shift in the private sector's perceptions of the central bank's inflation objective, which provides the long-run 'anchor' for the formation of inflation expectations (see Section 2.3.5). At the same time, it is assumed that the central bank's actual inflation objective guiding its monetary policy decisions remains unchanged.

The counterfactual de-anchoring simulation is conducted relative to a baseline which represents actual economic developments until the end of 2014 (corresponding to the historical data used in the estimation of NAWM II) and economic predictions for the following years that arguably incorporate the anticipated macroeconomic effects of the ECB's EAPP announcement in January 2015 and its subsequent implementation. 49 In the baseline, the private sector's inflation anchor is assumed to follow the movements in the observed SPF measure of longer-term inflation expectations and, hence, it declines by about 0.1 percentage point below the central bank's maintained inflation objective of 1.9% in the course of 2014. Thereafter, the announcement and implementation of the EAPP is assumed to have re-anchored longer-term inflation expectations. Accordingly, in the baseline, the decline in the inflation anchor comes to a halt at the turn of the year 2014/15, and it gradually recovers along an increasing path towards 1.9% in subsequent years. 50 In the counterfactual, the evolution of the inflation anchor from the beginning of 2015 onwards is instead determined endogenously. Specifically, the inflation anchor is assumed to evolve in an adaptive manner and to adjust downwards in response to consumer price inflation outcomes running persistently below the perceived inflation objective over the simulation horizon. 51

The four panels in Figure 17 portray the adverse consequences of the counterfactual deanchoring of longer-term inflation expectations in comparison with the baseline paths for

48 For an interpretation of this counterfactual simulation that attributes a key role to the announcement of the ECB's asset purchases in actually preventing a de-anchoring of longer-term inflation expectations, see Coenen and Schmidt (2016).

49 That is, to avoid double counting, the effects of the EAPP purchases, as gauged in the previous section, are not separately accounted for in the counterfactual simulation.

50 For a description of the developments in both survey and market-based indicators of longer-term inflation expectations in early 2015, see ECB (2015).

51 In the adaptive scheme for the inflation anchor, the weight on lagged consumer price inflation is set equal to the posterior mode estimate of 0.058.

annual consumer price inflation (measured in terms of the private consumption deflator), annual per-capita real GDP growth, the output gap, and the annualised short-term nominal interest rate (corresponding to the EONIA). 52 These baseline paths were extended beyond the year 2014 using commensurate vintages of Consensus Forecasts surveyed from financial and economic forecasters, the Commission's output gap forecast, as well as market-based interest-rate expectations. 53 They are represented by the blue solid lines, while the modelbased outcomes of the counterfactual de-anchoring simulation are depicted by the red dashed lines. The green solid and dash-dotted lines in the upper-left panel indicate the baseline and the counterfactual path of the inflation anchor, respectively.

In the counterfactual, the persistently low inflation outcomes over the simulation horizon lead to a sizeable additional downward shift in longer-term inflation expectations. The forward-looking private-sector agents respond to the decline in expected inflation rates by reducing their price and wage claims, giving rise to self-reinforcing second-round effects. The resulting moderate but lasting slowing of price and wage inflation towards levels consistent with the lower inflation anchor is exacerbated by the binding effective lower bound on nominal interest rates. The latter (marked by the pink shaded area in the lower-right panel) prevents the central bank from counteracting the further decline in inflation by lowering its policy rate. As a consequence, the real interest rate rises (and the real effective exchange rate of the euro appreciates), so aggregate demand is dampened and real GDP grows more slowly than in the baseline. The emerging slack (as indicated by the widening output gap) lowers price pressures over and above the direct effects resulting from the fall in expectations and further hampers the inflation adjustment process.

Thus, the counterfactual simulation illustrates the importance of solidly anchoring private-sector inflation expectations to avoid the prolongation of a period of low inflation outcomes through expectations-driven second-round effects.

## 6 Conclusion

In this paper, we have outlined the specification of NAWM II, which extends the original version of the NAWM by adding a rich financial sector. Our specification of the financial

52 For conducting the counterfactual, the 3-month EURIBOR used in the estimation has been mapped into the EONIA (Euro OverNight Index Average) so as to account more realistically for the existence of an effective lower bound on the riskless short-term nominal interest rate in the model. In the post-financialcrisis environment of excess liquidity in money markets, the latter is given by the rate on the ECB's deposit facility that banks may use to make overnight deposits with the Eurosystem.

53 For details on the methodology for extending the baseline paths with predictions for a restricted set of observed variables, see Appendix B in Coenen and Warne (2014).

sector distinguishes between wholesale banks, which face funding constraints inhibiting their ability to originate new loans, and retail banks, which distribute these new loans subject to staggered loan-rate setting. This specification comes close to meeting our threefold objective of accounting for a genuine role of the financial sector in the wider economy, capturing the prominent role of sluggish lending rates in the monetary transmission mechanism of the euro area, and providing a structural framework for assessing the macroeconomic impact of large-scale asset purchases by central banks.

We have presented estimation results for NAWM II, which are obtained by employing Bayesian methods and using an enhanced data set including financial time-series data. The properties of the model have been examined on the basis of selected impulse-response functions, by studying the implied historical and forecast-error-variance decompositions for real GDP growth, by comparing its implied sample moments with those based on the data, and by evaluating its relative forecasting performance. Overall, the estimated model is found to have economically and empirically plausible properties, including with regard to the nature and strength of the propagation of monetary and financial shocks to the wider economy and concerning the identification of the main sources of economic fluctuations in the run-up to, during and in the aftermath of the financial crisis of 2008/09. At the same time, the established features and properties of the original NAWM have been largely preserved, including the quantitative effects of standard monetary policy shocks on key macro variables, even though they are propagated through a far richer set of transmission channels in NAWM II. Furthermore, in terms of forecasting ability, NAWM II fares quite well compared to the NAWM as well as na¨ ıve benchmarks. Finally, illustrative applications demonstrate that the new model can make valuable contributions to the conduct of counterfactual policy analysis, including the quantitative assessment of the macroeconomic impact of the ECB's large-scale asset purchases carried out in recent years and the analysis of pertinent risks to the economic outlook, such as a possible de-anchoring of longer-term inflation expectations in an environment of persistently low inflation.

In the NAWM II, large-scale asset purchases exert their influence through credit easing and the exchange-rate channel. While these channels clearly played a prominent role, other non-modelled channels arguably also influenced the impact of these policy interventions. For example, asset purchase programmes, which reduce long-term government bond yields, can generate policy space that a fiscal authority can use to influence the effectiveness of the programmes. On the one hand, without violating deficit limits, the fiscal authority can reinforce the stimulative impact of the asset purchases by spending the gains from lower debt

financing costs on additional fiscal expenditures or tax cuts. On the other hand, the fiscal authority might offset part of the stimulative impact, if it were to use the flatter yield curve to extend the maturity and, therefore, the duration risk of its outstanding debt. Such a fiscal reaction would re-elevate the risk in private-sector financial portfolios, mitigating the initial risk-absorbing impact of the asset purchases. 54 Explicit modelling of fiscal policy reactions can shed light on the relative importance of these mechanisms. Additionally, asset purchase programmes can have a direct impact on interest-rate expectations, so they can reinforce the strength of accompanying forward guidance announcements (Bhattarai, Eggertsson, and Gafarov 2015). Furthermore, they can forestall a possible de-anchoring of inflation expectations through signalling the central bank's commitment to its medium-term inflation objective, as illustrated by the de-anchoring counterfactual in this paper. The empirical assessment of the strength of these channels in the euro area would be an interesting exercise. We leave this for future research.

54 For an early assessment of this channel in the euro area, see Andrade et al. (2016).

## References

- Altavilla, C., L. Brugnolini, R. S. G¨ urkaynak, R. Motto, and G. Ragusa (2019). Measuring euro area monetary policy. ECB Working Paper No. 2281.
- An, S. and F. Schorfheide (2007). Bayesian analysis of DSGE models. Econometric Reviews 26 (2-4), 113-172.
- Anderson, G. and G. Moore (1985). A linear algebraic procedure for solving linear perfect foresight models. Economics Letters 17 (3), 247-252.
- Andrade, P., J. Breckenfelder, F. De Fiore, P. Karadi, and O. Tristani (2016). The ECB's asset purchase programme: An early assessment. ECB Working Paper No. 1956.
- Andrle, M. and J. Beneˇ s (2013). System priors: Formulating priors about DSGE models' properties. IMF Working Paper No. 13/257.
- Aoki, K., G. Benigno, and N. Kiyotaki (2016). Monetary and financial polices in emerging markets. Manuscript, Princeton University.
- Barnard, J., R. McCulloch, and X.-L. Meng (2000). Modelling covariance matrices in terms of standard deviations and correlations, with application to shrinkage. Statistica Sinica 10 (4), 1281-1311.
- Bernanke, B. S., M. Gertler, and S. Gilchrist (1999). The financial accelerator in a quantitative business cycle framework. In J. B. Taylor and M. Woodford (Eds.), Handbook of Macroeconomics , Volume 1A, Chapter 21, pp. 1341-1393. New York: Elsevier.
- Bhattarai, S., G. Eggertsson, and B. Gafarov (2015). Time consistency and the duration of government debt: A signalling theory of quantitative easing. NBER Working Paper 21336.
- Bocola, L. (2016). The pass-through of sovereign risk. Journal of Political Economy 124 (4), 879-926.
- Brooks, S. P. and A. Gelman (1998). General methods for monitoring convergence of iterative simulations. Journal of Computational and Graphical Statistics 7 (4), 434455.
- Burlon, L., A. Gerali, A. Notarpietro, and M. Pisani (2017). Macroeconomic effectiveness of non-standard monetary policy and early exit. A model-based evaluation. International Finance 20 (2), 155-173.

- Calvo, G. A. (1983). Staggered prices in a utility-maximizing framework. Journal of Monetary Economics 12 (3), 383-398.
- Carlstrom, C. T., T. S. Fuerst, and M. Paustian (2015). Inflation and output in New Keynesian models with a transient interest rate peg. Journal of Monetary Economics 76 (C), 230-243.
- Carlstrom, C. T., T. S. Fuerst, and M. Paustian (2017). Targeting long rates in a model with segmented markets. American Economic Journal: Macroeconomics 9 (1), 205242.
- Chen, H., V. C´ urdia, and A. Ferrero (2012). The macroeconomic effects of large-scale asset purchase programmes. Economic Journal 122 (564), 289-315.
- Christiano, L. J., R. Motto, and M. Rostagno (2014). Risk shocks. American Economic Review 104 (1), 27-65.
- Christiano, L. J., M. Trabandt, and K. Walentin (2011). Introducing financial frictions and unemployment into a small open economy model. Journal of Economic Dynamics and Control 35 (12), 1999-2041.
- Christoffel, K., G. Coenen, and A. Warne (2008). The New Area-Wide Model of the euro area: A micro-founded open-economy model for forecasting and policy analysis. ECB Working Paper No. 944.
- Coenen, G. (2009). Extending the NAWM with a partial indexation mechanism linking wages and trend productivity. Manuscript, European Central Bank, available as MPRA Paper 86153.
- Coenen, G., A. T. Levin, and K. Christoffel (2007). Identifying the influences of nominal and real rigidities in aggregate price-setting behavior. Journal of Monetary Economics 54 (8), 2439-2466.
- Coenen, G., R. Motto, M. Rostagno, S. Schmidt, and F. Smets (2017). DSGE models and counterfactual analysis. In R. S. G¨ urkaynak and C. Tille (Eds.), DSGE Models in the Conduct of Policy: Use as Intended, Chapter 7, pp. 70-82. London: CEPR Press.
- Coenen, G. and S. Schmidt (2016). The role of the ECB's asset purchases in preventing a potential de-anchoring of longer-term inflation expectations. Research Bulletin No. 25. European Central Bank, July.
- Coenen, G., R. Straub, and M. Trabandt (2012). Fiscal policy and the Great Recession in the euro area. American Economic Review, Papers and Proceedings 102 (3), 71-76.

- Coenen, G., R. Straub, and M. Trabandt (2013). Gauging the effects of fiscal stimulus packages in the euro area. Journal of Economic Dynamics and Control 37 (2), 367386.
- Coenen, G. and I. Vetlov (2009). Extending the NAWM with a non-zero import content of exports. Manuscript, European Central Bank, available as MPRA Paper 76490.
- Coenen, G. and A. Warne (2014). Risks to price stability, the zero lower bound and forward guidance: A real-time assessment. International Journal of Central Banking 10 (2), 7-54.
- Coenen, G. and V. Wieland (2004). Exchange-rate policy and the zero bound on nominal interest rates. American Economic Review, Papers and Proceedings 94 (2), 80-84.
- C´ urdia, V. and M. Woodford (2011). The central bank balance sheet as an instrument of monetary policy. Journal of Monetary Economics 58 (1), 54-79.
- Darracq Pari` es, M. and M. K¨ uhl (2016). The optimal conduct of central bank asset purchases. ECB Working Paper No. 1973.
- Del Negro, M., G. Eggertsson, A. Ferrero, and N. Kiyotaki (2017). The Great Escape? A quantitative evaluation of the Fed's liquidity facilities. American Economic Review 107 (3), 824-57.
- Del Negro, M., M. Giannoni, and C. Patterson (2012). The forward guidance puzzle. Federal Reserve Bank of New York, Staff Report No. 574.
- Del Negro, M. and F. Schorfheide (2008). Forming priors for DSGE models (and how it affects the assessment of nominal rigidities). Journal of Monetary Economics 55 (7), 1191-1208.
- Dixit, A. K. and J. E. Stiglitz (1977). Monopolistic competition and optimum product diversity. American Economic Review 67 (3), 297-308.
- ECB (2010). The ECB's response to the crisis. Monthly Bulletin, Issue 10. European Central Bank, October.
- ECB (2015). Developments in longer-term inflation expectations in the euro area. Economic Bulletin, Issue 3, Box 4. European Central Bank, April.
- ECB (2016). A guide to the Eurosystem/ECB staff macroeconomic projection exercises. European Central Bank, July.
- ECB (2017). Impact of the ECB's non-standard measures on financing conditions: Taking

stock of recent evidence. Economic Bulletin, Issue 2, Box 3. European Central Bank, March.

- Edge, R. M., T. Laubach, and J. C. Williams (2007). Learning and shifts in long-run productivity growth. Journal of Monetary Economics 54 (8), 2421-2438.
- Fagan, G., J. Henry, and R. Mestre (2001). An Area-Wide Model (AWM) for the euro area. ECB Working Paper No. 42.
- Gal´ ı, J. and M. Gertler (2007). Macroeconomic modeling for monetary policy evaluation. Journal of Economic Perspectives 21 (4), 25-45.
- Gerali, A., S. Neri, L. Sessa, and F. M. Signoretti (2010). Credit and banking in a DSGE model of the euro area. Journal of Money, Credit and Banking 42 (s1), 107-141.
- Gertler, M. and P. Karadi (2011). A model of unconventional monetary policy. Journal of Monetary Economics 58 (1), 17-34.
- Gertler, M. and P. Karadi (2013). QE 1 vs. 2 vs. 3 . . . : A framework for analyzing largescale asset purchases as a monetary policy tool. International Journal of Central Banking 9 (1), 5-53.
- G¨ urkaynak, R. S., A. T. Levin, A. N. Marder, and E. T. Swanson (2007). Inflation targeting and the anchoring of inflation expectations in the Western Hemisphere. Federal Reserve Bank of San Francisco Economic Review , 25-47.
- Herbst, E. P. and F. Schorfheide (2016). Bayesian Estimation of DSGE Models . Princeton: Princeton University Press.
- Kimball, M. S. (1995). The quantitative analytics of the basic neomonetarist model. Journal of Money, Credit and Banking 27 (4), 1241-1277.
- Kiyotaki, N. and J. Moore (1997). Credit cycles. Journal of Political Economy 105 (2), 211-48.
- Klein, P. (2000). Using the generalized Schur form to solve a multivariate linear rational expectations model. Journal of Economic Dynamics and Control 24 (10), 1405-1423.
- Lind´ e, J., F. Smets, and R. Wouters (2016). Challenges for central banks' macro models. In J. B. Taylor and H. Uhlig (Eds.), Handbook of Macroeconomics , Volume 2B, Chapter 28, pp. 2185-2262. Amsterdam: North Holland.
- Iacoviello, M. (2005). House prices, borrowing constraints, and monetary policy in the business cycle. American Economic Review 95 (3), 739-764.

- Jermann, U. and V. Quadrini (2012). Macroeconomic effects of financial shocks. American Economic Review 102 (1), 238-271.
- Montes-Gald´ on, C. (2018). Imperfect credibility and the forward-guidance puzzle: An empirical evaluation. Manuscript, European Central Bank.
- Schorfheide, F. (2000). Loss function-based evaluation of DSGE models. Journal of Applied Econometrics 15 (6), 645-670.
- Smets, F. and R. Wouters (2007). Shocks and frictions in US business cycles: A Bayesian DSGE approach. American Economic Review 97 (3), 586-606.
- Taylor, J. B. (1993). Discretion versus policy rules in practice. Carnegie-Rochester Conference Series on Public Policy 39 (1), 195-214.
- Warne, A. (2018). YADA manual - Computational details. Manuscript, European Central Bank, available at https://www.texlips.net/download/yada.pdf .
- Woodford, M. (2001). Fiscal requirements for price stability. Journal of Money, Credit and Banking 33 (3), 669-728.
- Woodford, M. (2003). Interest and Prices: Foundations of a Theory of Monetary Policy . Princeton: Princeton University Press.

## Appendix: The log-linearised model

In this appendix we provide details on the derivation of the log-linear version of NAWM II, including the computation of its non-stochastic steady state, with a focus on the equations that relate to the model's financial-sector extension. The exposition follows closely the ordering of the equations in Section 2.2 in the main text, which outlines the financial-sector extension in non-linear form.

## Transformation of variables

We start by casting the model's structural equations into stationary form. The reason is twofold: First, because of the assumed unit-root process underlying the evolution of labour productivity, and consistent with the balanced-growth property of the model, all real variables, with the exception of hours worked, have a real stochastic trend in common. And second, as the central bank aims at stabilising inflation, rather than the price level, all nominal variables share a nominal stochastic trend.

To render the model stationary, we therefore scale all variables that share the common real trend with the level of productivity, z t , while we divide all nominal variables by the price of the consumption good, P C,t . In order to simplify the notation, we introduce the convention that all transformed variables are represented by lower-case letters, instead of the upper-case letters employed for the original variables. For example, we use y t = Y t /z t to denote the stationary level of aggregate output, while we use p I,t = P I,t /P C,t to represent the price of the investment good relative to the price of the consumption good.

There are, however, a few exceptions from these conventions that are noteworthy. First, since the nominal wage rate is assumed to grow in line with productivity, it not only needs to be transformed with the price of the consumption good, P C,t , but also with the productivity level, z t , in order to become stationary; and accordingly we define w t = W t / ( z t P C,t ). Second, as the model's endogenous state variables, such as the capital stock, are predetermined in a given period t, they will be scaled with the lagged value of productivity; that is, k t = K t /z t -1 . Third, the marginal utility of consumption needs to be scaled up with the level of productivity to become stationary; and hence we define λ t = z t Λ t . And fourth, we scale the foreign real variables with the productivity trend prevailing abroad, z ∗ t , while maintaining the assumption that z t and z ∗ t share the same stochastic trend. Thus, we treat ˜ z t = z t /z ∗ t as a stationary process that captures the degree of asymmetry in productivity developments in the domestic versus the foreign economy.

## Log-linearisation of equations around the non-stochastic steady state

After having made the necessary stationary-inducing transformations, we proceed with the derivation of the new, or modified, non-stochastic steady-state equations for NAWM II and with the log-linearisation of the re-scaled model equations around the steady state. In so doing, we indicate the logarithmic deviation of a variable from its steady-state value by a hat (' ̂ ') and define the latter implicitly by dropping the time subscript t . For example, the log-deviation from steady state for the scaled output variable is y ̂ t = log ( y t /y ).

## A.1 Re-scaled equations

Applying the transformations and conventions outlined above, this section lists the re-scaled stationary equations of the financial-sector block of NAWM II.

## A.1.1 Households

## A.1.2 Capital producers

<!-- formula-not-decoded -->

where p I,t = P I,t /P C,t

## A.1.3 Wholesale banks

<!-- formula-not-decoded -->

where b I,t +1 = B I,t +1 / ( P C,t z t ), b L,t +1 = B L,t +1 / ( P C,t z t ), b L,t +1 = B L,t +1 / ( P ∗ Y,t z ∗ t ), s t S t P ∗ Y,t /P Y,t , ˜ z t = z ∗ t /z t and p Y,t = P Y,t /P C,t

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where b I,t +1 = B I,t +1 / ( P C,t z t )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where b L,h,t +1 = B L,h,t +1 / ( P C,t z t )

<!-- formula-not-decoded -->

where ϑ t +1 = Θ t +1 / ( P C,t +1 z t +1 ) = ϑ

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.1.5 Foreign assets and domestic bond supply

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where b L,t +1 = B L,t +1 / ( P C,t z t ) and b t +1 = B t +1 / ( P C,t z t )

## A.1.4 Retail banks

## A.1.6 Central bank asset purchases and market clearing

<!-- formula-not-decoded -->

where d g t +1 = D g t +1 / ( P C,t z t )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.2 Non-stochastic steady state

This section first lists the new, or modified, steady-state equations for the financial-sector block of NAWM II. Modifications due to the new exchange-rate channel do not have any influence on the steady state as b ∗ L = 0 and s = 1. Moreover, central bank holdings of financial assets are zero in steady state, b g I = b g L = b ∗ ,g L = 0, so the efficiency costs related to holding these assets have no influence on the steady state either. Thereafter, the section explains how the steady state of NAWM II can be calculated without adding any new equation to the system of non-linear equations that enters the computation of the steady state of the original NAWM.

## A.2.1 Households and capital producers

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

noting that the latter equation is obtained by combining the households' first-order conditions with respect to K h,t +1 and u h,t and substituting in equation (A.22), with r K spread representing a free parameter used in the empirical implementation to calibrate the investmentto-output ratio for a given capital tax rate

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have used the fact that R D = R

<!-- formula-not-decoded -->

## A.2.2 Wholesale banks

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Λ = βg -1 z

## A.2.3 Retail banks

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The retail loan rate mark-up is a proportion α I of the loan rate spread:

<!-- formula-not-decoded -->

## A.2.4 Steady-state calculation

For the calculation of the steady state of NAWM II, we assume that the steady-state loan rate spread, R I -R , and the wholesale banks' leverage in steady state, Φ, are given, and that the absconding rate Ψ and the normalised start-up funds of new wholesale bankers ϑ are adjusting. In the following, we first state the equations that allow to recursively compute the state-state values of key financial-sector variables conditional on the solution of a small system of non-linear steady-state equations. We then briefly explain how the system of non-linear equations for the original NAWM needs to be modified to obtain the relevant system of equations for computing the steady state of NAWM II.

First, the risk-less (gross) interest rate equals

<!-- formula-not-decoded -->

From equations (A.24) and (A.27) we get the steady-state amount of investment loans,

<!-- formula-not-decoded -->

where the steady-state capital stock k is one of the variables which needs to be solved for numerically.

The amount of privately intermediated government bonds are

<!-- formula-not-decoded -->

where b L,h is given by equation (A.26).

The wholesale banks' net worth is given by equation (A.28),

<!-- formula-not-decoded -->

The start-up funds of new wholesale bankers are obtained from equation (A.29),

<!-- formula-not-decoded -->

From equations (A.30) and (A.33), and after some algebra, we get the absconding rate:

<!-- formula-not-decoded -->

From equation (A.40), the steady-state mark-down of the retail banks is

<!-- formula-not-decoded -->

From the definitions of R I and ˜ R I and from equation (A.38) we can get that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have made use of the fact that net-foreign-asset holdings are zero in steady state and hence ˜ ω ∗ L = ω ∗ L /ω L .

Finally, concerning interdependencies amongst the above steady-state equations, we note that the relationship between the input price of investment p I and the price of capital Q is given by equation (A.22). It depends on the Lagrange multiplier ς on the households' LIA constraint, which can be obtained from equation (A.25). The Lagrange multiplier enters the combined first-order conditions for households' physical capital holdings and the capital utilisation rate, resulting in equation (A.23).

The combined first-order conditions for households' capital holdings and the capital utilisation rate are also part of the system of non-linear steady-state equations of the original NAWM that needs to be solved using numerical methods. Hence, taking into account the dependency on the Lagrange multiplier (as well as the introduction of the free spread parameter r K spread ) the otherwise unchanged system of non-linear steady-state equations of the original NAWM allows to numerically determine the steady state of NAWM II. 55

## A.3 Log-linearised equations

Finally, this section lists the log-linearised equations of the financial-sector block of NAWM II, which are derived as first-order approximations around the model's nonstochastic steady state.

## A.3.1 Households and capital producers

where Σ t = (1 + ς t )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

55 For necessary modifications of the steady-state computations that are due to the incorporation of a non-zero import content of exports, see Coenen and Vetlov (2009).

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have used the fact that ̂ R D,t = ̂ /epsilon1 RP t + ̂ R t and R D = R

## A.3.2 Wholesale banks

<!-- formula-not-decoded -->

where ̂ b p I,t +1 = ( b p I,t +1 -b p I ) /y , ̂ b p I,t +1 = ( b p L,t +1 -b p L ) /y and ̂ b ∗ ,p L,t +1 = ( b ∗ ,p L,t +1 -b ∗ ,p L ) /y are the wholesale banks' asset holdings expressed as a share of steady-state output in deviation from the steady-state share

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have used the fact that ̂ ϑ t +1 = 0

<!-- formula-not-decoded -->

where we have used the fact that ̂ ω ∗ L,t = γ ∗ L ̂ s ∗ L,t +1 + ̂ /epsilon1 RP ∗ t , ˜ ω ∗ L = ω ∗ L /ω L and s ∗ L,t +1 = S t Q ∗ t B ∗ L,t +1 /P Y,t Y t .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.3.3 Retail banks

<!-- formula-not-decoded -->

where we have used the fact that Q ◦ I = ϕ I ˜ Q I

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.3.4 Foreign assets and domestic bond supply

<!-- formula-not-decoded -->

where both ̂ b ∗ L,t +1 = ( b ∗ L,t +1 -b ∗ L ) /y and ̂ tb t = ( tb t -tb ) /y are expressed as a share of steadystate output in deviation from the steady-state share, and we assume that b ∗ L = tb = 0

<!-- formula-not-decoded -->

## A.3.5 Central bank asset purchases and market clearing

<!-- formula-not-decoded -->

˜ Q I ̂ b g I,t +1 + Q L ̂ b g L,t +1 + s Q ∗ L p Y ˜ z ̂ b ∗ ,g L,t +1 = ̂ d g t +1 , (A.73) where ̂ b g I,t +1 = ( b g I,t +1 -b g I ) /y , ̂ b g L,t +1 = ( b g L,t +1 -b g L ) /y , ̂ b ∗ ,g L,t +1 = ( b ∗ ,g L,t +1 -b ∗ ,g L ) /y and ̂ d g t +1 = ( d g t +1 -d g ) /y are expressed as a share of steady-state output in deviation from the steady-state share, and we furthermore assume that b g I = b g L = b ∗ ,g L = d g = 0

<!-- formula-not-decoded -->

Table 1: Calibrated expenditure shares

| Share     | Description                      | Value   |
|-----------|----------------------------------|---------|
| s C       | Private consumption              | 57 . 5  |
| s I       | Investment                       | 21 . 0  |
| s G       | Government consumption           | 21 . 5  |
| s X       | Exports                          | 16 . 0  |
| s IM      | Imports                          | 16 . 0  |
| of which: | of which:                        |         |
| s IM C    | Import share of pr. consumption  | 8 . 5   |
| s IM I    | Import share of investment       | 4 . 5   |
| s IM G    | Import share of gov. consumption | 0 . 0   |
| s IM X    | Import share of exports          | 3 . 0   |

Note: This table provides information on the calibration of the steady-state ratios for the different expenditure categories in NAWM II, expressed as a share of GDP. The expenditure shares are computed using national accounts data and data from inputoutput tables, with the import shares being obtained by appropriately adjusting the quasi-share parameters ν C , ν I , ν G , and ν X in the respective final and intermediate-good aggregators of the model.

Table 2: Calibrated non-financial-sector parameters

| Parameter          | Description                         | Value     |
|--------------------|-------------------------------------|-----------|
| A. Preferences     |                                     |           |
| β †                | Discount factor                     | 0 . 998   |
| ζ                  | Inverse Frisch elasticity           | 2 . 0     |
| B. Technology      |                                     |           |
| δ                  | Depreciation rate                   | 0 . 025   |
| α                  | Capital share                       | 0 . 36    |
| ψ ‡                | Fixed cost in production            | 0 . 738   |
| g z                | Gross labour productivity growth    | 1 . 003   |
| C. Employment      |                                     |           |
| g E                | Gross labour force growth           | 1 . 00075 |
| D. Wage and price  | setting                             |           |
| ϕ W                | Wage mark-up                        | 1 . 30    |
| ϕ H ,ϕ X ,ϕ ∗      | Price mark-up                       | 1 . 35    |
| η H ,η X ,η ∗      | Slope of demand elasticity          | 10 . 0    |
| E. Tax rates       |                                     |           |
| τ C                | Consumption tax                     | 0 . 183   |
| τ N                | Labour income tax                   | 0 . 122   |
| τ W h              | Employees' social security contr.   | 0 . 118   |
| τ W f              | Employers' social security contr.   | 0 . 219   |
| τ K                | Capital income tax                  | 0 . 30    |
| τ D                | Profit income tax                   | 0 . 0     |
| F. Monetary policy |                                     |           |
| ¯ Π                | Gross inflation objective           | 1 . 00475 |
| R                  | Gross short-term nom. interest rate | 1 . 00975 |

Note: This table provides information on the calibrated non-financial-sector parameters of NAWM II. The superscript ' † ' indicates that the discount factor is calibrated so that it is consistent with a steady-state net real interest rate of 2% per annum and with steady-state net labour productivity growth of 1.2% per annum, while the superscript ' ‡ ' indicates that the fixed cost in production is calibrated such that the intermediate-good firms' profits are zero in steady state.

Table 3: Calibrated financial-sector parameters

| Parameter                                      | Description                                    | Value   |
|------------------------------------------------|------------------------------------------------|---------|
| A. Wholesale banks                             | A. Wholesale banks                             |         |
| Ψ †                                            | Absconding rate                                | 0 . 380 |
| ϑ †                                            | Start-up funds                                 | 0 . 049 |
| θ                                              | Survival rate                                  | 0 . 950 |
| B. Retail banks                                | B. Retail banks                                |         |
| ϕ I ‡                                          | Mark-down parameter                            | 0 . 987 |
| C. Duration of assets (= 1 / (4(1 - /rho1/R )) | C. Duration of assets (= 1 / (4(1 - /rho1/R )) |         |
| /rho1 I /sharp                                 | Decay parameter: inv. loans                    | 0 . 974 |
| /rho1 L /sharp                                 | Decay parameter: dom. gov. bonds               | 0 . 976 |
| /rho1 ∗ L /sharp                               | Decay parameter: foreign gov. bonds            | 0 . 976 |
| D. Long-term government bonds                  | D. Long-term government bonds                  |         |
| Q L s B L                                      | Supply of bonds over GDP                       | 0 . 7   |
| B L,h /B L                                     | Share of household bond holdings               | 0 . 75  |
| E. Cost of asset purchases                     | E. Cost of asset purchases                     |         |
| ϑ I ,ϑ L ,ϑ ∗ L                                | Cost of asset purchases                        | 0 . 0   |

Note: This table provides information on the calibrated financial-sector parameters of NAWM II. The superscript ' † ' indicates that the absconding rate and the start-up funds are calibrated so that the steady-state leverage ratio Φ equals 6, and the retail lending rate spread over the deposit rate, R I -R , equals 2.17 percentage points on an annualised basis. The superscript ' ‡ ' indicates that the retail banks' mark-down parameter is calibrated to generate a 10% mark-up in the excess return in the retail lending rate relative to the excess return in the wholesale lending rate. The superscript ' /sharp ' indicates that the decay parameters of the loans and bonds are calibrated to reflect assets with 10-year maturity.

Table 4: Prior and posterior distributions of estimated financial-sector parameters

| Parameter                                       | Description                                     | Prior distribution                              | Posterior distribution                          | Posterior distribution                          | Posterior distribution                          |
|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|
|                                                 |                                                 |                                                 | mode                                            | 5%                                              | 95%                                             |
| A. Wholesale banks                              | A. Wholesale banks                              | A. Wholesale banks                              | A. Wholesale banks                              | A. Wholesale banks                              | A. Wholesale banks                              |
| ω L                                             | Absconding of dom. bonds                        | gamma(0.72,0.05)                                | 0.83                                            | 0.78                                            | 0.91                                            |
| ˜ ω ∗ L                                         | Absconding of foreign bonds                     | gamma(1.00,0.05)                                | 0.96                                            | 0.90                                            | 1.03                                            |
| B. Retail banks                                 | B. Retail banks                                 | B. Retail banks                                 | B. Retail banks                                 | B. Retail banks                                 | B. Retail banks                                 |
| ξ I                                             | Calvo: price of inv. loans                      | beta(0.75,0.0375)                               | 0.74                                            | 0.69                                            | 0.78                                            |
| C. Portfolio adjustment costs                   | C. Portfolio adjustment costs                   | C. Portfolio adjustment costs                   | C. Portfolio adjustment costs                   | C. Portfolio adjustment costs                   | C. Portfolio adjustment costs                   |
| γ h L                                           | Households                                      | gamma(0.01,0.0025)                              | 0.009                                           | 0.006                                           | 0.013                                           |
| γ ∗ L                                           | Wholesale banks                                 | gamma(0.01,0.0025)                              | 0.004                                           | 0.003                                           | 0.006                                           |
| D. Autoregressive parameters of shock processes | D. Autoregressive parameters of shock processes | D. Autoregressive parameters of shock processes | D. Autoregressive parameters of shock processes | D. Autoregressive parameters of shock processes | D. Autoregressive parameters of shock processes |
| ρ θ                                             | Survival rate                                   | beta(0.75,0.05)                                 | 0.81                                            | 0.71                                            | 0.87                                            |
| ρ ϕ I                                           | Mark-down parameter                             | beta(0.75,0.05)                                 | 0.72                                            | 0.64                                            | 0.79                                            |
| E. Scaling parameters of shock processes        | E. Scaling parameters of shock processes        | E. Scaling parameters of shock processes        | E. Scaling parameters of shock processes        | E. Scaling parameters of shock processes        | E. Scaling parameters of shock processes        |
| σ θ                                             | Survival rate                                   | invgamma(0.1,2)                                 | 5.12                                            | 3.75                                            | 6.93                                            |
| σ ϕ I                                           | Mark-down parameter                             | invgamma(0.1,2)                                 | 1.22                                            | 0.98                                            | 1.64                                            |

Note: This table provides information on the marginal prior and posterior distributions of the estimated parameters for the financial sector of NAWM II. The prior distributions are characterised by the parameters determining their respective means and variances, except for the inverse gamma prior distributions for which the mode and the degrees of freedom are reported. The posterior distributions, which also take into account the system prior, are based on a Markov chain with 1,000,000 draws, with 500,000 draws being discarded as burn-in draws. See Appendix Figure A.2 for graphs of the marginal system prior and posterior densities of the parameters.

Table 5: Posterior distributions of estimated non-financial-sector parameters: Model structure

| Parameter                                 | Description                               | Posterior distribution of NAWM II   | Posterior distribution of NAWM II   | Posterior distribution of NAWM II   | Posterior mode of NAWM   | Posterior mode of NAWM   |
|-------------------------------------------|-------------------------------------------|-------------------------------------|-------------------------------------|-------------------------------------|--------------------------|--------------------------|
| Parameter                                 | Description                               | mode                                | 5%                                  | 95%                                 | CCW                      | updated                  |
| A. Preferences                            | A. Preferences                            |                                     |                                     |                                     |                          |                          |
| κ                                         | Habit formation                           | 0.62                                | 0.56                                | 0.66                                | 0.56                     | 0.65                     |
| B. Wage and price setting                 | B. Wage and price setting                 |                                     |                                     |                                     |                          |                          |
| ξ W                                       | Calvo scheme: wages                       | 0.78                                | 0.73                                | 0.82                                | 0.76                     | 0.72                     |
| χ W                                       | Indexation to inflation: wages            | 0.37                                | 0.24                                | 0.52                                | 0.63                     | 0.41                     |
| χ ˜ W                                     | Indexation to productivity: wages         | 0.66                                | 0.46                                | 0.83                                | [1.00]                   | [1.00]                   |
| ξ H                                       | Calvo scheme: domestic prices             | 0.82                                | 0.80                                | 0.84                                | 0.92                     | 0.89                     |
| χ H                                       | Indexation: domestic prices               | 0.23                                | 0.15                                | 0.32                                | 0.42                     | 0.48                     |
| ξ X                                       | Calvo scheme: export prices               | 0.75                                | 0.69                                | 0.80                                | 0.77                     | 0.73                     |
| χ X                                       | Indexation: export prices                 | 0.31                                | 0.21                                | 0.44                                | 0.49                     | 0.52                     |
| ξ ∗                                       | Calvo scheme: import prices               | 0.58                                | 0.51                                | 0.65                                | 0.53                     | 0.49                     |
| χ ∗                                       | Indexation: import prices                 | 0.38                                | 0.26                                | 0.54                                | 0.48                     | 0.35                     |
| o ∗                                       | Oil import share                          | 0.29                                | 0.23                                | 0.36                                | 0.16                     | 0.20                     |
| C. Final and intermediate-good production | C. Final and intermediate-good production |                                     |                                     |                                     |                          |                          |
| µ C                                       | Subst. elasticity: consumption            | 2.78                                | 2.33                                | 3.34                                | 1.94                     | 2.54                     |
| µ I                                       | Subst. elasticity: investment             | 1.38                                | 1.03                                | 1.92                                | 1.60                     | 1.94                     |
| µ X                                       | Subst. elasticity: exports                | 0.82                                | 0.64                                | 1.09                                | -                        | -                        |
| µ ∗                                       | Price elasticity: exports                 | 1.12                                | 0.82                                | 1.50                                | 1.03                     | 1.08                     |
| D. Adjustment costs                       | D. Adjustment costs                       |                                     |                                     |                                     |                          |                          |
| γ I                                       | Investment                                | 10.78                               | 8.63                                | 13.62                               | 5.17                     | 5.77                     |
| γ u, 2                                    | Capital utilisation                       | 0.91                                | 0.67                                | 1.36                                | [Inf]                    | [Inf]                    |
| γ IM C                                    | Import content: consumption               | 6.27                                | 4.71                                | 8.61                                | 5.60                     | 4.61                     |
| γ IM I                                    | Import content: investment                | 0.74                                | 0.39                                | 2.09                                | 0.40                     | 1.38                     |
| γ ∗                                       | Export market share                       | 2.03                                | 1.24                                | 4.76                                | 2.42                     | 2.23                     |
| E. Interest-rate rule                     | E. Interest-rate rule                     |                                     |                                     |                                     |                          |                          |
| φ R                                       | Interest-rate smoothing                   | 0.93                                | 0.91                                | 0.94                                | 0.86                     | 0.86                     |
| φ Π                                       | Response to inflation                     | 2.74                                | 2.38                                | 3.30                                | 1.90                     | 1.86                     |
| φ ∆Π                                      | Response to change in inflation           | 0.04                                | 0.02                                | 0.07                                | 0.18                     | 0.22                     |
| φ Y                                       | Response to output gap                    | 0.03                                | 0.02                                | 0.05                                | [0.00]                   | [0.00]                   |
| φ ∆ Y                                     | Response to change in output gap          | 0.10                                | 0.09                                | 0.12                                | 0.15                     | 0.09                     |
| F. Perception updating equations          | F. Perception updating equations          |                                     |                                     |                                     |                          |                          |
| /pi1 Π ¯ p                                | Sens. of perc. inflation objective        | 0.06                                | 0.04                                | 0.07                                | -                        | -                        |
| /pi1 g Y p                                | Sens. of perc. trend growth rate          | 0.06                                | 0.05                                | 0.08                                | -                        | -                        |
| G. Employment (bridge) equation           | G. Employment (bridge) equation           |                                     |                                     |                                     |                          |                          |
| ξ E                                       | Calvo-style weighing scheme               | 0.86                                | 0.84                                | 0.87                                | 0.85                     | 0.82                     |

Note: This table provides information on the marginal posterior distributions of the estimated parameters for the nonfinancial-sector model structure of NAWM II, along with posterior mode estimates of the corresponding parameters of the original NAWM as reported in CCW and when re-estimated using (per-capita) data until 2014Q4. The posterior distributions for NAWM II, which take into account the marginal prior distributions of the parameters and the system prior, are based on a Markov chain with 1,000,000 draws, with 500,000 draws being discarded as burn-in draws. See Appendix Figure A.2 for graphs of the marginal system prior and posterior densities of the parameters. For the NAWM, entries in brackets denote calibrated values.

Table 6: Posterior distributions of estimated non-financial-sector parameters: Shock processes

| Parameter                    | Description                      | Posterior distribution of NAWM II   | Posterior distribution of NAWM II   | Posterior distribution of NAWM II   | Posterior mode of NAWM       | Posterior mode of NAWM       |
|------------------------------|----------------------------------|-------------------------------------|-------------------------------------|-------------------------------------|------------------------------|------------------------------|
|                              |                                  | mode                                | 5%                                  | 95%                                 | CCW                          | updated                      |
| A. Autoregressive parameters | A. Autoregressive parameters     | A. Autoregressive parameters        | A. Autoregressive parameters        | A. Autoregressive parameters        | A. Autoregressive parameters | A. Autoregressive parameters |
| ρ RP                         | Domestic risk premium shock      | 0.97                                | 0.96                                | 0.97                                | 0.92                         | 0.97                         |
| ρ RP ∗                       | External risk premium shock      | 0.99                                | 0.97                                | 1.00                                | 0.88                         | 0.99                         |
| ρ g z                        | Permanent technology shock       | -                                   | -                                   | -                                   | 0.80                         | 0.70                         |
| ρ g p z                      | [. . .]: persistent component    | 0.94                                | 0.92                                | 0.96                                | -                            | -                            |
| ρ ε                          | Transitory technology shock      | 0.92                                | 0.87                                | 0.95                                | 0.90                         | 0.98                         |
| ρ I                          | Investment-specific techn. shock | 0.91                                | 0.86                                | 0.95                                | 0.71                         | 0.44                         |
| ρ ϕ W                        | Wage mark-up shock               | 0.68                                | 0.61                                | 0.73                                | 0.67                         | 0.68                         |
| ρ ϕ H                        | Domestic price mark-up shock     | 0.59                                | 0.51                                | 0.65                                | 0.40                         | 0.58                         |
| ρ ϕ X                        | Export price mark-up shock       | 0.41                                | 0.33                                | 0.48                                | 0.38                         | 0.34                         |
| ρ ϕ ∗                        | Import price mark-up shock       | 0.50                                | 0.42                                | 0.58                                | 0.55                         | 0.82                         |
| ρ IM                         | Import demand shock              | 0.85                                | 0.73                                | 0.89                                | 0.86                         | 0.82                         |
| ρ ν ∗                        | Export preference shock          | 0.88                                | 0.74                                | 0.92                                | 0.81                         | 0.87                         |
| B. Scaling parameters        | B. Scaling parameters            | B. Scaling parameters               | B. Scaling parameters               | B. Scaling parameters               | B. Scaling parameters        | B. Scaling parameters        |
| σ RP                         | Domestic risk premium shock      | 0.14                                | 0.12                                | 0.17                                | 0.16                         | 0.12                         |
| σ RP ∗                       | External risk premium shock      | 0.13                                | 0.11                                | 0.19                                | 0.43                         | 0.19                         |
| σ g z                        | Permanent technology shock       | -                                   | -                                   | -                                   | 0.12                         | 0.50                         |
| σ g p z                      | [. . .]: persistent component    | 0.07                                | 0.06                                | 0.09                                | -                            | -                            |
| σ ε                          | Transitory technology shock      | 1.00                                | 0.81                                | 1.37                                | 1.13                         | 0.81                         |
| σ I                          | Investment-specific techn. shock | 0.25                                | 0.22                                | 0.29                                | 0.41                         | 0.53                         |
| σ ϕ W                        | Wage mark-up shock               | 0.10                                | 0.08                                | 0.13                                | 0.11                         | 0.14                         |
| σ ϕ H                        | Domestic price mark-up shock     | 0.13                                | 0.11                                | 0.15                                | 0.12                         | 0.16                         |
| σ ϕ X                        | Export price mark-up shock       | 0.90                                | 0.78                                | 1.06                                | 1.06                         | 1.35                         |
| σ ϕ ∗                        | Import price mark-up shock       | 1.10                                | 0.94                                | 1.28                                | 0.97                         | 1.27                         |
| σ IM                         | Import demand shock              | 6.49                                | 5.26                                | 9.47                                | 4.60                         | 6.46                         |
| σ ν ∗                        | Export preference shock          | 7.97                                | 5.40                                | 14.82                               | 8.07                         | 8.67                         |
| σ R                          | Interest rate shock              | 0.11                                | 0.10                                | 0.12                                | 0.12                         | 0.11                         |
| σ ¯ Π p                      | Perc. inflation objective shock  | 0.03                                | 0.02                                | 0.03                                | -                            | -                            |
| σ g p Y                      | Perc. trend growth rate shock    | 0.03                                | 0.02                                | 0.03                                | -                            | -                            |
| C. Signal-to-noise ratio     | C. Signal-to-noise ratio         | C. Signal-to-noise ratio            | C. Signal-to-noise ratio            | C. Signal-to-noise ratio            | C. Signal-to-noise ratio     | C. Signal-to-noise ratio     |
| σ 2 g p z /σ 2 g tr z        | Permanent technology shock       | 0.06                                | 0.05                                | 0.06                                | -                            | -                            |

Note: This table provides information on the marginal posterior distributions of the estimated parameters of the nonfinancial-sector shock processes of NAWM II, along with posterior mode estimates of the corresponding parameters of the original NAWM as reported in CCW and when re-estimated using (per-capita) data until 2014Q4. The posterior distributions for NAWM II, which take into account the marginal prior distributions of the parameters and the system prior, are based on a Markov chain with 1,000,000 draws, with 500,000 draws being discarded as burn-in draws. See Appendix Figure A.2 for graphs of the marginal system prior and posterior densities of the parameters.

Table 7: Sample means and standard deviations of observed variables

|                               | Sample mean   | Sample mean   | Sample mean   | Sample mean   | Sample standard deviation   | Sample standard deviation   | Sample standard deviation   | Sample standard deviation   |
|-------------------------------|---------------|---------------|---------------|---------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|
| Variable                      | NAWM II       | NAWM II       | NAWM II       | Data          | NAWM II                     | NAWM II                     | NAWM II                     | Data                        |
|                               | Mean          | 5%            | 95%           | Data          | Mean                        | 5%                          | 95%                         | Data                        |
| Real GDP                      | 0 . 38        | 0 . 14        | 0 . 60        | 0 . 37        | 0 . 88                      | 0 . 72                      | 1 . 05                      | 0 . 59                      |
| Consumption                   | 0 . 38        | 0 . 13        | 0 . 62        | 0 . 35        | 0 . 92                      | 0 . 75                      | 1 . 10                      | 0 . 50                      |
| Investment                    | 0 . 37        | - 0 . 12      | 0 . 86        | 0 . 36        | 3 . 12                      | 2 . 39                      | 3 . 97                      | 1 . 49                      |
| Exports                       | 0 . 38        | - 0 . 02      | 0 . 78        | 0 . 37        | 3 . 92                      | 3 . 32                      | 4 . 62                      | 2 . 49                      |
| Imports                       | 0 . 37        | - 0 . 05      | 0 . 79        | 0 . 37        | 2 . 99                      | 2 . 43                      | 3 . 64                      | 2 . 05                      |
| GDP defl. inflation           | 0 . 47        | 0 . 16        | 0 . 78        | 0 . 58        | 0 . 61                      | 0 . 50                      | 0 . 74                      | 0 . 33                      |
| Consumption defl. inflation   | 0 . 47        | 0 . 16        | 0 . 77        | 0 . 56        | 0 . 55                      | 0 . 44                      | 0 . 68                      | 0 . 37                      |
| Import defl. inflation        | 0 . 47        | 0 . 09        | 0 . 86        | 0 . 05        | 2 . 92                      | 2 . 40                      | 3 . 51                      | 2 . 48                      |
| Employment                    | - 0 . 01      | - 2 . 00      | 2 . 00        | 0 . 00        | 2 . 08                      | 1 . 23                      | 3 . 16                      | 2 . 43                      |
| Wage inflation                | 0 . 77        | 0 . 40        | 1 . 13        | 0 . 82        | 0 . 77                      | 0 . 62                      | 0 . 94                      | 0 . 52                      |
| Short-term nom. interest rate | 3 . 89        | 0 . 91        | 6 . 82        | 5 . 01        | 2 . 52                      | 1 . 53                      | 3 . 83                      | 3 . 43                      |
| Real effective exchange rate  | 0 . 12        | - 11 . 13     | 11 . 39       | 0 . 00        | 13 . 15                     | 8 . 45                      | 19 . 31                     | 8 . 59                      |
| 10-year gov't bond yield      | 5 . 54        | 4 . 33        | 6 . 76        | 5 . 21        | 0 . 97                      | 0 . 61                      | 1 . 45                      | 1 . 90                      |
| Comp. long-term lending rate  | 6 . 06        | 4 . 38        | 7 . 71        | 4 . 09        | 0 . 74                      | 0 . 37                      | 1 . 24                      | 0 . 67                      |
| Long-term inflation expect's  | 1 . 88        | 0 . 42        | 3 . 36        | 1 . 91        | 0 . 69                      | 0 . 33                      | 1 . 23                      | 0 . 06                      |
| Long-term growth expect's     | 1 . 50        | 0 . 76        | 2 . 25        | 2 . 10        | 0 . 34                      | 0 . 17                      | 0 . 60                      | 0 . 29                      |
| Output gap                    | 0 . 09        | - 10 . 75     | 10 . 71       | - 0 . 01      | 4 . 94                      | 2 . 29                      | 8 . 75                      | 2 . 05                      |

Note: This table reports posterior mean estimates as well as the 5% and 95% percentiles for the sample means and standard deviations implied by NAWM II for the observed variables used in its Bayesian estimation, along with the corresponding sample moments based on the data. The model-based sample means and standard deviations are computed by simulating data with the model using 5,000 draws from the posterior distribution of its estimated parameters with one sample path per parameter draw.

Table 8: Forecast-error-variance decomposition for real GDP growth

| Shock group                     | 4-quarter horizon   | 4-quarter horizon   | 4-quarter horizon   | Long run   | Long run   | Long run   |
|---------------------------------|---------------------|---------------------|---------------------|------------|------------|------------|
|                                 | Mean                | 5%                  | 95%                 | Mean       | 5%         | 95%        |
| Technology                      | 28                  | 23                  | 33                  | 34         | 28         | 40         |
| Demand                          | 12                  | 9                   | 16                  | 13         | 9          | 16         |
| Mark-ups                        | 14                  | 11                  | 17                  | 15         | 12         | 18         |
| Policy                          | 7                   | 5                   | 9                   | 7          | 5          | 9          |
| Foreign                         | 7                   | 6                   | 8                   | 7          | 6          | 9          |
| Financial                       | 22                  | 18                  | 26                  | 23         | 19         | 27         |
| of which: dom. risk prem. shock | 18                  | 14                  | 22                  | 17         | 13         | 21         |
| ext. risk prem. shock           | 0                   | 0                   | 0                   | 1          | 0          | 1          |
| shock to survival rate          | 3                   | 2                   | 4                   | 4          | 3          | 6          |
| mark-down shock                 | 1                   | 0                   | 1                   | 1          | 1          | 2          |
| Perceptions                     | 0                   | 0                   | 0                   | 0          | 0          | 0          |
| All                             | 90                  | 88                  | 91                  | 99         | 99         | 99         |

Note: This table reports posterior mean estimates as well as the 5% and 95% percentiles for the contributions of the structural shocks of NAWM II to the forecast error variance of per-capita real GDP growth at the 4-quarter horizon and in the long run. The shock contributions are reported only for the share of the forecast errors attributable to the structural shocks (expressed in percent), while the shares of the forecast errors due to measurement errors and unobserved state variables are skipped. The shock contributions are computed using 5,000 draws from the posterior distribution of the model's estimated parameters. The structural shocks are bundled into seven groups: technology, demand, mark-up, foreign, financial and perception shocks, plus a monetary policy shock. For the financial shock group, the contributions of the individual financial shocks are reported as well.

Table 9: Effects of central bank asset purchases

| Simulation                      | Real GDP growth (pp)   | Real GDP growth (pp)   | Real GDP growth (pp)   | Consumption deflator inflation (pp)   | Consumption deflator inflation (pp)   | Consumption deflator inflation (pp)   |
|---------------------------------|------------------------|------------------------|------------------------|---------------------------------------|---------------------------------------|---------------------------------------|
|                                 | Year 1                 | Year 2                 | Year 3                 | Year 1                                | Year 2                                | Year 3                                |
| Benchmark w/o interest-rate peg | 0.06                   | 0.09                   | 0.02                   | 0.03                                  | 0.03                                  | 0.01                                  |
| Benchmark w interest-rate peg   | 0.34                   | 0.21                   | - 0.04                 | 0.19                                  | 0.18                                  | 0.12                                  |
| Variants w/o interest-rate peg: |                        |                        |                        |                                       |                                       |                                       |
| Higher riskiness of inv. bonds  | 0.07                   | 0.12                   | 0.05                   | 0.03                                  | 0.03                                  | 0.01                                  |
| No exchange-rate channel        | 0.04                   | 0.08                   | 0.05                   | 0.01                                  | - 0.00                                | - 0.01                                |

Note: This table reports the NAWM II-based responses of real GDP growth and consumption deflator inflation to a central bank asset purchase shock with a total size of 11% of GDP. The responses are expressed as average annual percentage-point (pp) deviations from the model's balanced-growth path. They are computed using the posterior mode estimates of the parameters of the standard model.

Figure 1: Data for original model

<!-- image -->

Note: This figure shows the updated and extended time series of the observed variables used in the estimation of the original version of the NAWM. Details on the variable transformations are provided in Section 3.1.1. Inflation and interest rates are reported in annualised percentage terms.

Figure 1: Data for original model (continued)

Note: See above.

<!-- image -->

Figure 1: Data for original model (continued)

Note: See above.

<!-- image -->

Figure 2: Additional financial, survey and output gap data

<!-- image -->

Note: This figure shows the time series of the additional observed variables used in the estimation of NAWM II. The euro area 10-year government bond yield series (AAA) is available from 2004Q3 onwards, while the earlier observations concern the German 10-year government bond yield. The euro area long-term lending rate is available from 2003Q1 onwards and covers (new business) lending with an original maturity of over 1 year to households for house purchases and to non-financial corporations.

Figure 3: Convergence of the posterior sampling algorithm

<!-- image -->

Note: The upper panel in this figure displays sequential estimates of the multivariate potential scale reduction factor (MPSRF) for NAWM II based on five Markov chains. Each chain has 1,000,000 posterior draws, with the first 500,000 being discarded as burn-in draws. The lower panel shows sequential estimates of the trace of the within covariance matrix used in the computation of the MPSRF and of the trace of the pooled covariance matrix.

Figure 4: Sample correlations with respect to real GDP growth

<!-- image -->

Note: This figure shows the mean (blue solid lines) and the 70% and 90% equal-tail credible intervals (grey-shaded areas) of the model-based sample correlations between real GDP growth and other observed variables for NAWM II, along with the sample correlations based on the data (red solid lines with plus markers). The model-based correlations are computed using 5,000 draws from the posterior distribution of the model's estimated parameters using one simulated sample path per parameter draw.

Figure 5: Propagation of an interest-rate shock

<!-- image -->

Note: This figure shows the mean (blue solid lines) and the 70% and 90% equal-tail credible intervals (grey-shaded areas) of the impulse responses to an interest-rate shock equal to one standard deviation for key variables of NAWM II, along with the mean (red dashed lines) of the corresponding impulse responses to a shock of equal size for the original NAWM. For each model, the impulse responses are reported as percentage deviations from its non-stochastic balanced-growth path, except for the impulse responses of the inflation and interest rates which are reported as annualised percentage-point deviations. The mean and the uncertainty bands of the impulse responses are computed for each model using 5,000 draws from the posterior distribution of its estimated parameters.

Figure 6: Propagation of an interest-rate shock: The role of financial frictions

<!-- image -->

Note: This figure shows the impulse responses to an interest-rate shock equal to one standard deviation for key variables of the standard version of NAWM II with financial frictions (blue solid lines) and for a version of the model without financial frictions (red dashed lines). The two sets of impulse responses are reported as percentage deviations from the model's non-stochastic balanced-growth path, except for the impulse responses of the inflation and interest rates which are reported as annualised percentage-point deviations. They are computed using the posterior mode estimates of the parameters of the standard model.

Figure 7: Propagation of a shock to the survival rate of wholesale banks

<!-- image -->

Note: This figure shows the mean (blue solid lines) and the 70% and 90% equal-tail credible intervals (grey-shaded areas) of the impulse responses to a shock to the survival rate of wholesale banks equal to one standard deviation for key variables of NAWM II. The impulse responses are reported as percentage deviations from the model's non-stochastic balanced-growth path, except for the impulse responses of the inflation and interest rates which are reported as annualised percentage-point deviations. The mean and the uncertainty bands of the impulse responses are computed using 5,000 draws from the posterior distribution of the model's estimated parameters.

Figure 8: Propagation of a shock to the mark-down parameter of retail banks

<!-- image -->

Note: This figure shows the mean (blue solid lines) and the 70% and 90% equal-tail credible intervals (grey-shaded areas) of the impulse responses to a shock to the mark-down parameter of retail banks equal to one standard deviation for key variables of NAWM II. The impulse responses are reported as percentage deviations from the model's non-stochastic balanced-growth path, except for the impulse responses of the inflation and interest rates which are reported as annualised percentage-point deviations. The mean and the uncertainty bands of the impulse responses are computed using 5,000 draws from the posterior distribution of the model's estimated parameters.

Figure 9: Propagation of a domestic risk premium shock

<!-- image -->

Note: This figure shows the mean (blue solid lines) and the 70% and 90% equal-tail credible intervals (grey-shaded areas) of the impulse responses to a domestic risk premium shock equal to one standard deviation for key variables of NAWM II, along with the mean (red dashed lines) of the corresponding impulse responses to a shock of equal size for the original NAWM. For each model, the impulse responses are reported as percentage deviations from its non-stochastic balanced-growth path, except for the impulse responses of the inflation and interest rates which are reported as annualised percentage-point deviations. The mean and the uncertainty bands of the impulse responses are computed for each model using 5,000 draws from the posterior distribution of its estimated parameters.

Figure 10: Propagation of a permanent technology shock: Persistent component

<!-- image -->

Note: This figure shows the mean (blue solid lines) and the 70% and 90% equal-tail credible intervals (grey-shaded areas) of the impulse responses to an increase in the persistent component of the permanent technology shock equal to one standard deviation for key variables of NAWM II, along with the mean (red dashed lines) of the corresponding impulse responses to a shock of equal long-run impact for the original NAWM. For each model, the impulse responses are reported as percentage deviations from its non-stochastic balanced-growth path, except for the impulse responses of the inflation and interest rates which are reported as annualised percentage-point deviations. The mean and the uncertainty bands of the impulse responses are computed for each model using 5,000 draws from the posterior distribution of its estimated parameters.

Figure 11: Historical decomposition of real GDP growth

<!-- image -->

<!-- image -->

Note: The upper panel in this figure depicts the decomposition of annual per-capita real GDP growth into the contributions of the structural shocks of NAWM II over the period 2000 to 2014. The shocks are bundled into seven groups: technology, demand, mark-up, foreign, financial and perception shocks, plus a monetary policy shock. The decomposition is computed in deviation from the model-implied mean real GDP growth rate using the posterior mode estimates of the model's estimated parameters. The lower panel shows the contributions of the individual financial sector shocks in the real GDP growth decomposition.

Figure 12: Mean paths of real GDP, inflation and interest-rate forecasts

## A. Real GDP growth

<!-- image -->

B. GDP deflator inflation

<!-- image -->

Note: For NAWM II, the original NAWM, and two na¨ ıve benchmarks (the no change/random walk and the sample mean), this figures shows the mean paths of unconditional one to eight quarter-ahead forecasts for annual per-capita real GDP growth, annual GDP deflator inflation and the annualised short-term nominal interest rate. The forecasts are computed out-of-sample over the period 2006Q1 to 2014Q4. For NAWM II and the NAWM, the forecasts are given by the means of the predictive densities conditional on the posterior mode estimates of the model's parameters.

Figure 12: Mean paths of real GDP, inflation and interest-rate forecasts (continued)

## C. Short-term nominal interest rate

<!-- image -->

Note: See above.

Figure 13: Root-mean-squared errors of real GDP, inflation and interest-rate forecasts

<!-- image -->

Note: For NAWM II, the original NAWM, and a na¨ ıve benchmark (the no change/random walk), this figure shows the root-mean-squared errors (RMSEs) of one to eight-quarter-ahead forecasts for annual per-capita real GDP growth, annual GDP deflator inflation and the annualised short-term nominal interest rate, relative to the RMSEs of their sample mean forecasts. RMSEs below (above) unity reflect a better (worse) point forecasting performance than the sample mean. The forecasts are computed out-of-sample over the period 2006Q1 to 2014Q4. For NAWM II and the NAWM, the forecasts are given by the means of the predictive densities conditional on the posterior mode estimates of the model's parameters.

Figure 14: Effects of central bank asset purchases

<!-- image -->

Note: This figure shows the impulse responses to a central bank asset purchase shock with a total size of 11% of GDP for key variables of NAWM II when using the benchmark simulation set-up with an endogenous short-term nominal interest rate reaction (blue solid lines) and when using the benchmark set-up with the nominal interest rate being kept unchanged for eight quarters and with imperfect credibility of the central bank's announcement thereof (blue dashed lines). The two sets of impulse responses are reported as percentage deviations from the model's non-stochastic balanced-growth path, except for the impulse responses of the inflation and interest rates which are reported as annualised percentage-point deviations. They are computed using the posterior mode estimates of the model's parameters.

Figure 15: Effects of asset purchases: The role of the riskiness of assets

<!-- image -->

Note: This figure shows the impulse responses to a central bank asset purchase shock with a total size of 11% of GDP for key variables of NAWM II when using the benchmark simulation set-up (blue solid lines) and when using an alternative simulation set-up with a higher riskiness of the long-term investment bonds (red dashed lines). The two sets of impulse responses are reported as percentage deviations from the model's non-stochastic balanced-growth path, except for the impulse responses of the inflation and interest rates which are reported as annualised percentage-point deviations. They are computed using the posterior mode estimates of the model's parameters.

Figure 16: Effects of asset purchases: The role of the exchange-rate channel

<!-- image -->

Note: This figure shows the impulse responses to a central bank asset purchase shock with a total size of 11% of GDP for key variables of the standard version of the NAWM II with a no-arbitrage condition for long-term government bonds determining the exchange-rate response (blue solid lines) and for a version of the model in which this no-arbitrage condition is replaced by the standard UIP condition (red dashed lines). The two sets of impulse responses are reported as percentage deviations from the model's nonstochastic balanced-growth path, except for the impulse responses of the inflation and interest rates which are reported as annualised percentage-point deviations. They are computed using the posterior mode estimates of the parameters of the standard model.

Figure 17: Consequences of a de-anchoring of longer-term inflation expectations

<!-- image -->

Note: This figure shows the results of the counterfactual simulation of a de-anchoring of longer-term inflation expectations within NAWM II. In the baseline (BL) simulation, it is assumed that the private sector's inflation anchor follows the movements in the SPF measure of longer-term inflation expectations until the end of 2014 and, hence, it falls by about 0.1 percentage point below the central bank's maintained inflation objective of 1.9% in the course of 2014. Thereafter, the inflation anchor is assumed to gradually recover, along a gently increasing path. In the counterfactual (CF) simulation, the inflation anchor is obtained endogenously from an adaptive expectations scheme, with the weight on lagged consumer price inflation set equal to the estimated value of 0.058. Consumer price inflation (measured in terms of the private consumption deflator) and per-capita real GDP growth are expressed as annual percentages, the output gap is measured in percent of potential output and the short-term nominal interest rate (corresponding to the EONIA) is expressed in annualised percentage terms. The effective lower bound on the short-term nominal interest rate is imposed at an interest-rate level of -14 basis points (set equal to the minimum of the EONIA forward curve over the extended baseline horizon). For conducting the simulations, the posterior mode estimates of the model's parameters are used.

Figure A.1: Marginal prior densities of estimated parameters with and without system prior

<!-- image -->

PSfrag replacements

Note: This figure compares the marginal prior densities of the estimated parameters of NAWM II using the system prior information (blue solid lines) and their marginal prior densities without the system prior (red dashed lines). The parameter densities are ordered according to the order of the parameters

in Tables 4 to 6.

Figure A.1: Marginal prior densities of estimated parameters with and without system prior (continued)

<!-- image -->

PSfrag replacements

Note: See above.

Figure A.1: Marginal prior densities of estimated parameters with and without system prior (continued)

<!-- image -->

PSfrag replacements

Note: See above.

Figure A.1: Marginal prior densities of estimated parameters with and without system prior (continued)

<!-- image -->

PSfrag replacements

Note: See above.

Figure A.2: Marginal system prior and posterior densities of estimated parameters

<!-- image -->

PSfrag replacements

Note: This figure plots the marginal posterior densities of the estimated parameters of NAWM II based on a Markov chain with 1,000,000 draws (blue solid lines) against their marginal system prior densities (red dashed lines), with 500,000 draws being discarded as burn-in draws. The parameter densities are ordered according to the order of the parameters in Tables 4 to 6.

Figure A.2: Marginal system prior and posterior densities of estimated parameters (continued)

<!-- image -->

PSfrag replacements

Note: See above.

Figure A.2: Marginal system prior and posterior densities of estimated parameters (continued)

<!-- image -->

PSfrag replacements

Note: See above.

Figure A.2: Marginal system prior and posterior densities of estimated parameters (continued)

<!-- image -->

PSfrag replacements

Note: See above.

Figure A.3: Smoothed estimates of shocks

<!-- image -->

Figure A.3: Smoothed estimates of shocks (continued)

<!-- image -->

Figure A.3: Smoothed estimates of shocks (continued)

Note: See above.

<!-- image -->

Figure A.3: Smoothed estimates of shocks (continued)

Note: See above.

<!-- image -->

Figure A.4: Smoothed estimates of innovation component of shocks

<!-- image -->

mode estimates of its parameters.

Figure A.4: Smoothed estimates of innovation component of shocks (continued)

Note: See above.

<!-- image -->

Figure A.4: Smoothed estimates of innovation component of shocks (continued)

Note: See above.

<!-- image -->

Figure A.4: Smoothed estimates of innovation component of shocks (continued)

Note: See above.

<!-- image -->

Figure A.5: Recursive posterior mode estimates of parameters

<!-- image -->

PSfrag replacements

Note: This figure graphs the recursively estimated posterior mode of the parameters of NAWM II, with the estimation sample being gradually extended by a full year from 2005Q4 to 2014Q4. The graphs of the recursive posterior mode estimates are ordered according to the order of the parameters in Tables 4 to 6.

Figure A.5: Recursive posterior mode estimates of parameters (continued)

<!-- image -->

PSfrag replacements

Note: See above.

Figure A.5: Recursive posterior mode estimates of parameters (continued)

<!-- image -->

PSfrag replacements

Note: See above.

Figure A.5: Recursive posterior mode estimates of parameters (continued)

<!-- image -->

PSfrag replacements

Note: See above.

## Acknowledgements

We would like to thank Kai Christoffel, Fiorella De Fiore, Jordi Galí, Andrea Gerali, Philipp Hartmann, Luc Laeven, Johannes Pfeifer, Giorgio Primiceri, Ricardo Reis, Chris Sims, Frank Smets, Mathias Trabandt and Oreste Tristani for helpful discussions and suggestions at the various stages of the model-development project. We are particularly grateful to Marco Onofri, who provided excellent research assistance, to José-Emilio Gumiel, who offered meticulous data support, and to Carlos Montes-Galdón, who provided us with the code for simulating the effects of interest-rate forward guidance under imperfect credibility. We have benefited from presentations of earlier drafts and of related material at the ECB, to the Monetary Policy Committee of the European System of Central Banks and to its Working Group on Econometric Modelling, at the 2016 GSE Summer Forum in Barcelona, at the 1st Research Conference of the CEPR Network on Macroeconomic Modelling and Model Comparison in Frankfurt, at the 7th BIS Research Network Meeting on 'Pushing the frontier of central banks' macro-modelling' in Basle, at the 20th Central Bank Macroeconomic Modelling Workshop in Paris, and at the 24th International Conference on Computing in Economics and Finance in Milan. The views expressed in this paper are those of the authors and do not necessarily reflect the views of the ECB.

## Günter Coenen (corresponding author)

European Central Bank, Frankfurt am Main, Germany; email: gunter.coenen@ecb.europa.eu

## Peter Karadi

European Central Bank, Frankfurt am Main, Germany; email: peter.karadi@ecb.europa.eu

## Sebastian Schmidt

European Central Bank, Frankfurt am Main, Germany; email: sebastian.schmidt@ecb.europa.eu

## Anders Warne

European Central Bank, Frankfurt am Main, Germany; email: anders.warne@ecb.europa.eu

## © European Central Bank, 2019

Postal address  60640 Frankfurt am Main, Germany Telephone +49 69 1344 0 Website www.ecb.europa.eu

All rights reserved. Any reproduction, publication and reprint in the form of a different publication, whether printed or produced electronically, in whole or in part, is permitted only with the explicit written authorisation of the ECB or the authors.

This paper can be downloaded without charge from www.ecb.europa.eu, from the Social Science Research Network electronic library or from RePEc: Research Papers in Economics. Information on all of the papers published in the ECB Working Paper Series can be found on the ECB's website.

PDF

ISBN 978-92-899-3305-6

QB-AR-18-080-EN-N