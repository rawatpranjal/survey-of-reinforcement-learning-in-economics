## Constraining DM properties with SPI

Alexey Boyarsky 1 /star , Denys Malyshev 2 , 3 , Andrey Neronov 4 /star , Oleg Ruchayskiy 5

1 CERN, PH/TH, CH-1211 Geneve 23, Switzerland

- 2 Bogolyubov Institute for Theoretical Physics, Kiev, 03780, Ukraine

3

Dublin Institute for Advanced Studies, 31 Fitzwilliam Place, Dublin 2, Ireland

4 INTEGRAL Science Data Center, Chemin d' ´ Ecogia 16, 1290 Versoix, Switzerland

Geneva Observatory, 51 ch. des Maillettes, CH-1290 Sauverny, Switzerland

- 5 ´

Ecole Polytechnique F´ ed´ erale de Lausanne, Institute of T heoretical Physics,

- FSB/ITP/LPPC, BSP 720, CH-1015, Lausanne, Switzerland

## ABSTRACT

Using the high-resolution spectrometer SPI on board the International Gamma-Ray Astrophysics Laboratory (INTEGRAL) , we search for a spectral line produced by a dark matter (DM) particle with a mass in the range 40 keV &lt; M DM &lt; 14 MeV , decaying in the DM halo of the Milky Way. To distinguish the DM decay line from numerous instrumental lines found in the SPI background spectrum, we study the dependence of the intensity of the line signal on the offset of the SPI pointing from the direction toward the Galactic Centre. After a critical analysis of the uncertainties of the DM density profile in the inner Galaxy, we find that the intensity of the DM decay line should decrease by at least a factor of 3 when the offset from the Galactic Centre increases from 0 ◦ to 180 ◦ . We find that such a pronounced variation of the line flux across the sky is not observed for any line, detected with a significance higher than 3 σ in the SPI background spectrum. Possible DM decay origin is not ruled out only for the unidentified spectral lines, having low ( ∼ 3 σ ) significance or coinciding in position with the instrumental ones. In the energy interval from 20 keV to 7 MeV, we derive restrictions on the DM decay line flux, implied by the (non-)detection of the DM decay line. For a particular DMcandidate, the sterile neutrino of mass M DM , we derive a bound on the mixing angle.

Key words: methods: data analysis-techniques: spectroscopic - Galaxy: halo - dark matter;

pact halo objects (MACHOs) - constitute dominant fraction of mass in the halo (Alcock et al. 2000; Lasserre et al. 2000; Alard 1999). The only non-baryonic DM candidate in the SM candidates -(left-handed) neutrino -is ruled out from the large scale structure (LSS) considerations (see e.g. Bond et al. 1980; Hannestad &amp; Raffelt 2004; Crotty et al. 2004).

What are the properties of a successful DM candidate? First of all, this particle should be massive. Many extensions of the SM present the DM candidates with the masses ranging from ∼ 10 -10 eV (massive gravitons, Dubovsky et al. 2005) and ∼ 10 -6 eV (axions) to hundreds of GeV (WIMPs) and even to 10 13 GeV (WIMPZILLA, Kuzmin &amp; Tkachev 1998, 1999; Chung et al. 1999). For a review of particle physics DM candidates see e.g. Bergstrom (2000); Bertone et al. (2005); Carr et al. (2006).

Secondly, there should exist mechanisms of DM production with the correct abundances. The production mechanism in particular determines the velocity distribution of particles in the early Universe. This velocity distribution can, in principle, be probed experimentally. Namely, if during the structure formation epoch the DM particles have velocities, comparable to the speed of sound

## 1 INTRODUCTION

## Dark matter in the Universe

There is a vast body of evidence, suggesting that the large fraction of matter in the Universe exists in the form of the Dark matter (DM) . However, while the total density of the DM is measured with a very high precision ( Ω DM h 2 = 0 . 105 +0 . 007 -0 . 009 , Spergel et al. 2007), little is known about its properties apart from this. The possibility that the DM is composed of the Standard Model (SM) particles has been ruled out for a long time already. Indeed, the DM cannot be made out of baryons, as producing such an amount of baryonic matter would require drastic modifications of the scenario of the Big Bang nucleosynthesis (BBN), which otherwise successfully describes the abundance of light elements (see for example Dar 1995). Recent microlensing experiments rule out the possibility that another type of baryonic DM - massive com-

/star On leave of absence from Bogolyubov Institute for Theoretical Physics, Kiev, Ukraine

CERN-PH-TH/2007-202

## 2 A. Boyarsky et al.

in the baryon-photon plasma, they 'erase' density fluctuations at scales, smaller than the distance, they have traveled (called the free-streaming length ). To differentiate various models in accordance with this property, the DM candidates with the negligible velocity dispersion (and, correspondingly, free-streaming) are called cold DM (CDM), while those with the free-streaming of the order of ∼ 1 Mpc are considered to be warm (WDM). 1 It is possible to constrain the free-streaming length of a particular DM candidate by probing the structure of the Universe at galaxy-size scales. This can be done through the analysis of the Lymanα forest data (Hui et al. 1997). Lymanα analysis puts an upper bound on the free-streaming of the DM particles (Hansen et al. 2002; Viel et al. 2005; Seljak et al. 2006; Viel et al. 2006; Viel et al. 2007). It should be noted however that currently existing interpretation of the Lymanα data is model-dependent, as, apart from a number of astrophysical assumptions (see Hui et al. 1997) and complicated hydrodynamic simulations, it relies on a priori assumptions about the velocity distribution of the DM particles.

A way to differentiate between CDM and WDM models would be to compare the numerical simulations of the DM distribution in the Milky Way-type galaxies with the actual observations. However, the resolution of the N-body simulations is not yet sufficient to answer the questions about e.g. the DM density profiles in dwarf satellite galaxies. Moreover, most of the simulations include only collisionless DM particles, and do not model the baryons and their feedback on the galaxy structure formation. These problems are not solved even for the CDM simulations, and WDMsimulations have additional serious difficulties. From an observational point of view, it has been argued for some time already that there is a discrepancy between CDM simulations and observations (see e.g. Moore 1994; Moore et al. 1999; Klypin et al. 1999; Bode et al. 2001; Avila-Reese et al. 2001; Goerdt et al. 2006) It has been claimed recently that a number of recent observations of dwarf satellite galaxies of the Milky way and Andromeda galaxy seem to indicate the existence of the smallest scale at which the DM exists (Gilmore et al. 2006, 2007; Gilmore 2007; Koposov et al. 2007). However, this statement and the interpretation of the observations are still subject to debate (Klimentowski et al. 2007; Penarrubia et al. 2007; Strigari et al. 2007; Simon &amp; Geha 2007). Therefore it is too early to say what kind of DM models is favoured by comparing simulations and observations.

Usually it is also necessary for the DM candidate to be stable. For the most popular DM candidate - weakly interacting massive particles (WIMPs), this is related to the fact that the particles of ∼ electroweak mass, having weak strength interaction with SM matter (required to produce the correct amount of DM), would decay too fast and would not be 'dark'. If, however, the DM particle interacts with the SM more weakly than WIMPs, it could well have a finite (although cosmologically long) life time.

There exist several unstable (decaying) DM candidates e.g. gravitino (Borgani et al. 1996; Baltz &amp; Murayama 2003; Roszkowski et al. 2005; Cerdeno et al. 2006; Cembranos et al. 2006; Lola et al. 2007). In this paper we will concentrate mainly on one candidate, the sterile neutrino (although our results will be applicable for any type of decaying DM). Constraints on the decaying DM were analyzed in de Rujula &amp; Glashow (1980); Berezhiani et al. (1987); Doroshkevich et al. (1989); Berezhiani et al. (1990); Berezhiani &amp; Khlopov (1990);

1 The left-handed neutrino would represent hot DM in this terminology, i.e. the DM with the free-streaming length /greatermuch 1 Mpc.

Bertone et al. (2007); Zhang et al. (2007) (see also the book by Khlopov 1997).

## Sterile neutrino DM

It was noticed long ago that the right-handed (or as it is often called sterile ) neutrino with the mass in the keV range would represent a viable DM candidate (Dodelson &amp; Widrow 1994). Such a neutrino would interact with the rest of the matter only via the quadratic mixing with left-handed ( active ) neutrinos and therefore (although not stable) could have cosmologically long life-time. At the same time, it could be produced in the early Universe with the correct abundances (Dodelson &amp; Widrow 1994; Shi &amp; Fuller 1999; Shaposhnikov &amp; Tkachev 2006). One of the decay channels of the unstable sterile neutrinos includes emission of photons of the energy equal to half of the sterile neutrino rest energy. This potentially provides a possibility to observe the decays of DM sterile neutrinos via detection of a characteristic spectral line in the spectra of astrophysical objects with large DM concentration.

Recently this DM candidate has attracted much attention (see e.g. Shaposhnikov (2007) and references therein). It was found that a very modest and natural extension of the SM by 3 right-handed neutrinos (making the SM more symmetric as all SM fermions, including neutrino, would have now their left and right handed counterparts) provided a viable extension of the theory, capable of solving several 'beyond the SM' problems. First of all, such an extension makes neutrinos massive and thus perhaps provides the simplest and the most natural explanation of the phenomenon of 'neutrino oscillations' (see e.g. Fogli et al. (2006); Strumia &amp; Vissani (2006); Giunti (2007) for reviews). The smallness of neutrino masses in this model (called ν MSM in Asaka &amp; Shaposhnikov 2005) is achieved by the usual see-saw mechanism with Majorana masses of right-handed neutrinos being below electroweak scale. 2

Secondly, if two heavier sterile neutrinos ( N 2 and N 3 ) are almost degenerate in mass and have their masses between O (100)MeV and O (20)GeV , the ν MSMprovides the mechanism of generating the baryon asymmetry of the Universe. Thirdly, the lightest sterile neutrino N 1 can have arbitrary mass and arbitrarily weak coupling with the (active) neutrino sector. At the same time, it can be produced in the early Universe in the correct amounts. It represents therefore the DM particle in the ν MSM.Thus, altogether the ν MSMrepresents (arguably) the simplest extension of the SM, capable of explaining three important questions: origin and smallness of neutrino masses, baryon asymmetry in the Universe and the existence of the DM.

## Existing restrictions on sterile neutrino DM parameters.

What are the current restrictions on parameters (mass and mixing ) of sterile neutrino DM? First of all sterile neutrino mass should

2 The fact that the ν MSM does not introduce any new scale above the electroweak one, makes this theory especially appealing from the point of view of its experimental verification/falsification.

satisfy the universal Tremaine-Gunn lower bound: 3 M DM /greaterorsimilar 300 -500 eV . 4

Next, as the sterile neutrino possesses the (two-body) radiative decay channel: N 1 → ν + γ , the emitted photon would carry the energy E γ = M DM / 2 . A large flux of such photons is expected from the large concentrations of the DM sterile neutrinos, like galaxies or galaxy clusters.

Recently an extensive search of the DM decay line in the region of masses M DM /lessorsimilar 20 keV was conducted, using the data of Chandra (Riemer-Sørensen et al. 2006; Boyarsky et al. 2006d; Abazajian et al. 2007) and XMM-Newton (Boyarsky et al. 2006a,b,c; Watson et al. 2006; Boyarsky et al. 2007). The region of soft X-ray (down to energies 0 . 2 keV) was explored by Boyarsky et al. (2007) with the use of the wide field of view spectrometer (McCammon et al. 2002). The non-observation of the DM decay line in X-ray, combined with the first principles calculation of DM production in the early Universe (Asaka et al. 2007), implies that the Dodelson &amp; Widrow (1994) (DW) scenario can work only if the sterile neutrino mass is below 4 keV (Boyarsky et al. 2008). If one takes into account recent lower bound on the mass of sterile neutrino DM in the DW scenario M DM /greaterorequalslant 5 . 6 keV (Viel et al. 2007), it seems that the possibility that all the DM is produced via DW scenario is ruled out (Boyarsky et al. 2008). The possibility that only fraction of the DM is produced via DW mechanism remains open (Palazzo et al. 2007).

There are other viable mechanisms of DM production, including e.g. resonant oscillation production in the presence of lepton asymmetries (Shi &amp; Fuller 1999). Sterile neutrino DM can be produced by the decay of light inflaton (Shaposhnikov &amp; Tkachev 2006) or in a similar model with the different choice of parameters (Kusenko 2006; Petraki &amp; Kusenko 2007). These mechanisms are currently not constrained and remain valid for DM particles with the masses in the keV range and above.

The search for the DM decay line signal produced by sterile neutrinos with masses above ∼ 20 keV is complicated by the absence of the focusing optics telescopes (similar to Chandra or XMM-Newton ) in the hard X-ray and γ -ray domain of the spectrum. For example, the existing restrictions in the 20 -100 keV mass range (Boyarsky et al. 2006a,c) are derived from the observations of diffuse X-ray background, with the help of non-imaging instruments, HEAO-I (Gruber et al. 1999). The current status of astrophysical observations in summarized in Ruchayskiy (2007).

In this paper we use the spectrometer SPI on board of INTEGRAL satellite to place restrictions on parameters of decaying DM in the mass range 40 keV -14 MeV . This range of masses is interesting, for example, the sterile neutrinos, produced in the early Universe in the presence of large lepton asymmetries (Shi &amp; Fuller 1999) or through the inflaton decay (Shaposhnikov &amp; Tkachev 2006). It is also relevant for the case of gravitino DM (Pagels &amp; Primack 1982; Bond et al. 1982).

3 In its simplest form the Tremaine-Gunn bound comes from the fact that for the fermions there is a maximal density in the phase space (Tremaine &amp; Gunn 1979; Dalcanton &amp; Hogan 2001) and therefore the observed phase-space density in various DM dominated systems should be less that this (mass dependent) bound.

4 A stronger lower bound from Lyα (Seljak et al. 2006; Viel et al. 2006; Viel et al. 2007) can be obtained in the case of the particular production mechanisms - the Dodelson-Widrow scenario (Dodelson &amp; Widrow 1994). For other possible production mechanisms (e.g. Shi &amp; Fuller 1999; Shaposhnikov &amp; Tkachev 2006) the Lyα constraints should be reanalyzed.

Figure 1. Comparison of sensitivity towards the search of the narrow DM decay line for different instruments with the wide FoV. Diagonal straight lines show the improvement of sensitivity (by a factor, marked on the line) as compared with the HEAO-I A4 low energy detector (LED), taken as a reference.

<!-- image -->

When the preparation of this paper was at its final stage, Y¨ uksel et al. (2008, hereafter Y07 ) published their work, which used the results of Teegarden &amp; Watanabe (2006, hereafter T06 ) to place restrictions on the parameters of sterile neutrino DM in the range 40 -700 keV . We discuss it in more details in Section 6.

## SPI spectrometer

The absence of the focusing optics significantly reduces the sensitivity of the telescopes operating in the hard X-ray/soft γ -ray energy band. Most of the instruments operating in this energy band use collimators and/or coded masks to distinguish signals from the sources on the sky from the instrumental background. Contrary to the focusing optics telescopes, both the source and background signals are collected from the entire detector, which significantly increases the irreducible background.

The focusing optics enables to significantly reduce the background only in the studies of point sources. If the source under investigation occupies a large fraction of the sky (e.g. the entire Milky Way galaxy), the performance of the focusing and non-focusing instruments with the same detector collection area are, in fact, comparable.

In the case of an extended source, emitting a narrow spectral line, an efficient way of reduction of instrumental background is via the improvement of the spectral resolution of the instrument (in the case of a broad continuum background spectrum, the number of background counts at the energy of the line is proportional to the spectral resolution ∆ E ). The best possible sensitivity is achieved when the spectral resolution reaches the intrinsic width of the spectral line (see Fig.1 for the case of wide FoV instruments and Boyarsky et al. (2007) for the case of narrow FoV instruments).

In the case of the line produced by the DM decaying in the Milky Way halo, the line width is determined by the Doppler broad-

Figure 2. The geometry of the SPI FoV.

<!-- image -->

ening by the random motion of the DM particles. The velocity dispersion of the DM motion in the halo is about the rotation velocity of the Galactic disk, v ∼ 200 km/s. This means that Doppler broadening of the DM decay line is about

<!-- formula-not-decoded -->

Thus, the optimal spectral resolution of an instrument searching for the DM decay line produced by the Milky Way DM halo should be ∆ E /similarequal 10 -3 E .

Such optimal spectral resolution is almost achieved with the spectrometer SPI on board of INTEGRAL satellite, which has the maximal spectral resolving power of E/ ∆ E /similarequal 500 and works in the energy range 20 keV - 8 MeV (Vedrenne et al. 2003). SPI is a 'coded mask' type instrument with an array of 19 hexagonal shaped Ge detectors (of which only 17 are operating at the moment).

The SPI telescope consists of a coded mask inscribed into a circle of the radius R mask = 39 cm, placed at the height H = 171 cm above the detector plane and of the detector, which has the shape of a hexagon inscribed into a circle of the radius R det /similarequal 15 . 3 cm (see Fig. 2). The portion of the sky visible from each point of the SPI detector (the so-called fully coded field of view , FCFOV) has therefore angular diameter

<!-- formula-not-decoded -->

while the portion of the sky visible by at least some of the detectors (the partially coded field of view , PCFOV) is

<!-- formula-not-decoded -->

The solid angle spanned by the cone with this opening angle is Ω PCFOV = 2 π ' 1 -cos(Θ PCFOV / 2) ' /similarequal 0 . 29 (see Fig. 2). Wide field of view makes the SPI telescope suitable for the study of the very extended sources, like the Milky Way DM halo.

Figure 3. The effective area of the SPI detector for an on-axis source, as a function of the photon energy. The plot is produced by collective the on-axis effective areas of the 17 SPI detectors from the instrumental characteristics files.

<!-- image -->

## 2 THE EXPECTED SIGNAL FROM THE DM DECAY IN THE HALO OF THE MILKY WAY.

The expected surface brightness of the DM decay line in a given direction on the sky is a function of the angular distance φ between the given direction on the sky and the direction towards the Galactic center (GC). It can be calculated by taking the integral of the DM density profile ρ DM ( r ) along the line of sight (' column density ')

<!-- formula-not-decoded -->

where r /circledot /similarequal 8 . 5 kpc is the distance from the Solar system to the GC . Angle φ is related to the galactic coordinates ( b, l ) via

<!-- formula-not-decoded -->

Thus, the galactic center corresponds to φ = 0 ◦ , the anti-center φ = 180 ◦ , and the direction perpendicular to the galactic plane to φ = 90 ◦ . The expected DM flux is given then by

<!-- formula-not-decoded -->

where Γ DM is the DM decay rate

In general, the surface brightness F DM ( φ ) is variable across the telescope FoV. This is especially true for a wide field of view (FoV) instruments (like SPI). In order to calculate the detector count rate, one has to integrate flux (6) over the FoV and over the (effective) detector area and then divide by the energy of the photons, E γ = M DM / 2 :

<!-- formula-not-decoded -->

where ( α, β ) are the angular coordinates in the FoV, A eff is the effective area at energy E γ for the photons, coming from the direction ( α, β ).

The effective area of the SPI detector (which is determined by the transparency of the mask and the quantum efficiency of the detector) changes with the photon energy. For an on-axis point source,

<!-- formula-not-decoded -->

the integral of Eq. (7) reduces to f 0 A eff , on , where A eff , on ( E γ ) is the detector effective area for an on-axis source . Its dependence on energy E γ is shown on Fig. 3. 5

5 The on-axis effective area is calculated by summing the energy-

Figure 4. Dependence of the effective area on the off-axis position of a (point) source.

<!-- image -->

In the general case of extended sources, evaluation of the detector count rate (7) analytically is not possible because of the complicated dependence of the effective area on the off-axis angle (shown on Fig. 4). In the simplest case of an extended source with a constant surface brightness dF DM ( φ ) /d Ω = f ext = const , the integral of Eq. (7) reduces to the multiplication by the solid angle Ω PCFOV /similarequal 0 . 29 and the effective area, averaged over the FoV:

<!-- formula-not-decoded -->

The numerical factor κ ( E γ ) depends on the energy and has to be calculated via a numerical integration over the energy dependent off-axis response map of the SPI detector. A reasonably accurate numerical approximation to κ ( E γ ) is given by

<!-- formula-not-decoded -->

One can see that κ /lessmuch 1 in all the energy interval. This is explained by the fact that the detector area visible from a given direction on the sky strongly decreases with the increase of the off-axis angle of this direction, so that the sky-averaged effective area is much smaller than the on-axis effective area of the detector. Substituting (9), (8) into (7) one finds that for an extended source of constant surface brightness the detector count rate is

<!-- formula-not-decoded -->

dependent on-axis effective areas of each of the 17 operating detectors of SPI, extracted from the instrument's characteristics files.

## 2.1 Modeling the DM halo of the Galaxy

The DM halo of the Galaxy has been extensively studied (see e.g. Kravtsov et al. 1998; Klypin et al. 2002; Battaglia et al. 2005). Various DM profiles, used to fit observed velocity distributions, differ the most in the GC region.

It was shown in Klypin et al. (2002); Battaglia et al. (2005) that the DM halo of the MW can be described by the NavarroFrenk-White (NFW) profile (Navarro et al. 1997)

<!-- formula-not-decoded -->

with parameters, given in Table 1.The relation between virial parameters and ρ s , r s can easily be found (see e.g. the Appendix A of Boyarsky et al. 2007).

To explore the uncertainty of the DM density profile in the inner part of the Galaxy, we also describe the DM distribution in the MW via an isothermal profile (Bahcall &amp; Soneira 1980):

<!-- formula-not-decoded -->

The following parameters of isothermal profile reproduce the DM contribution to the (outer parts of) Galaxy rotation curve v h = 170 km/sec and r c = 4 kpc (Boyarsky et al. 2006c, 2007) (i.e. ρ 0 = 1 . 2 × 10 6 keV cm 3 [ v h 170 km / s ] 2 [ 4 kpc r c ] 2 ). These parameters are consistent with those, from favored NFW models of Klypin et al. (2002); Battaglia et al. (2005), i.e. for φ /greaterorequalslant 90 ◦ the difference between isothermal model and NFW with preferred parameters was completely negligible (less than 5%) - c.f. FIG.5. Both types of models provide the local DM density at the position of the Sun to be ρ DM ( r /circledot ) /similarequal 0 . 22 GeV / cm 3 , which is close to the existing estimates (Kuijken &amp; Gilmore 1989c,a,b, 1991; Gilmore et al. 1989).

The DM flux from a given direction φ , measured by an observer on Earth (distance r /circledot /similarequal 8 . 5 kpc from the GC), is given by

<!-- formula-not-decoded -->

where R = √ r 2 c + r 2 /circledot sin 2 φ and ρ 0 r c /similarequal 1 . 5 × 10 28 keV / cm 2 .

The uncertainty of the DM radial density profile in the inner Galaxy stems from the difficulty of separation between visible and DM contributions to the inner Galaxy rotation curve. 6 In order to get the most conservative limit on the column density of the DM in the direction of the GC, one can assume the following 'rigid lower bound': while the DM outside the r /circledot is described by the 'maximal disk' model (model A 2 of Klypin et al. 2002), for r /lessorequalslant r /circledot DM density remains constant (so that the total DM mass within r /circledot is the same as in the model A 2 of Klypin et al. 2002). This gives

<!-- formula-not-decoded -->

The surface brightness profile on the 'constant density' model is shown in black dashed line on the Fig.5. One can see that the difference between the maximal ( φ = 0 ◦ ) and the minimal φ = 180 ◦ ) column densities is ∼ 3 . 4 (as compared to ∼ 6 for

6 When quoting results of Klypin et al. (2002), we do not take the effects of baryon compression on DM into account. While these effects make DM distribution in the core of the MW denser, any such computation is strongly model dependent.

Table 1. Best-fit parameters of NFW model of the MW DM halo. Max. disk model maximizes amount of baryonic matter in the inner 3 kpc of the MW halo ( M DM / ( M disk + M bulge ) = 0 . 4 for the model A 2 and M DM / ( M disk + M bulge ) = 0 . 14 in the model B 2 ).

| References                                          | M vir [ M /circledot ]       |   r vir [kpc] |   Concentration |   r s [kpc] | ρ s [ M /circledot / kpc 3 ]   |
|-----------------------------------------------------|------------------------------|---------------|-----------------|-------------|--------------------------------|
| Klypin et al. (2002), favored models ( A 1 or B 1 ) | 1 . 0 × 10 12                |           258 |              12 |        21.5 | 4 . 9 × 10 6                   |
| Klypin et al. (2002), Max. disk models A 2          | 0 . 71 × 10 12               |           230 |               5 |        46   | 0 . 6 × 10 6                   |
| Klypin et al. (2002), Max. disk models B 2          | 0 . 71 × 10 12               |           230 |              10 |        23   | 3 . 1 × 10 6                   |
| Battaglia et al. (2005)                             | 0 . 8 +1 . 2 - 0 . 2 × 10 12 |           255 |              18 |        14.2 | 11 . 2 × 10 6                  |

Figure 5. Expected column density for various DM profiles: favored NFW profile (red thick solid line); NFW profile with the maximal disk (model A 2 , see Table 1) - blue solid line; cored (isothermal) profile - green thick dashed line; constant density within r /circledot - black dashed line).

<!-- image -->

isothermal model). For comparison we show on Fig. 5 expected DM flux (6) for various profiles. The minimal column density is of course the one in the direction of anti-center: S ( φ = 180 ◦ ) /similarequal 0 . 33 × 10 28 keV / cm 2 . We see that even for the minimal profile S ( φ &lt; 30 ◦ ) /greaterorequalslant 10 28 keV / cm 2 .

## 2.2 DMdecay line count rate

In the case of the Majorana sterile neutrinos of mass M DM the DMdecay width is given by (Pal &amp; Wolfenstein 1982; Barger et al. 1995): 7

<!-- formula-not-decoded -->

Substituting (15) to (6) we find

<!-- formula-not-decoded -->

7 The quoted value of Γ DM is for the Majorana sterile neutrino. In case of Dirac particle this value is 2 times smaller (c.f. Pal &amp; Wolfenstein 1982; Barger et al. 1995).

The lower bound on the DM decay line rate in SPI pointings toward the inner Galaxy is calculated by substituting the column density S = 10 28 keV / cm 2 (see Fig. 5) into Eqs. (10), (16)

<!-- formula-not-decoded -->

The approximation of the constant surface brightness works well, if the extended source has a core of the angular diameter exceeding the size of the SPI partially coded FoV ( Θ PCFOV ≈ 17 ◦ maximal off-axis angle). Taking isothermal profile the angular size of the flat core of the extended source is

<!-- formula-not-decoded -->

which satisfies this constraint.

## 3 STRATEGY OF SEARCH FOR THE DM DECAY LINE WITH SPI

The MW halo contribution to the DM decay signal represents the all-sky source. Indeed, as the results of Section 2.1 show, the variability of the signal over the sky may be as low as the factor ∼ 3 . This makes the strategy of search of the DM decay signal different from any other types of astrophysical sources: the point sources, diffuse sources (e.g. ∼ 10 ◦ Gaussian profile for e + e -annihilation region, Kn¨ odlseder et al. (2005)) or even the search for DM a nnihilation signal (see e.g. Tasitsiomi et al. 2004; Boehm et al. 2004; Diemand et al. 2007; S´ anchez-Conde et al. 2006; Carr et al. 2006).

The problem gets exacerbated by the fact that during its motion, SPI is irradiated by the charged high-energy particles (particles from Earth radiation belt, Solar wind, cosmic TeV photons). As a result, the materials (even detectors themselves) used for SPI construction start to radiate in different energy regions (see subsection 3.2). As a result any SPI spectrum consists of a broad continuum, which is a combination of the sky and instrumental backgrounds, and of a set of the instrumental background lines (Atti´ e et al. 2003; Diehl et al. 2003; Jean et al. 2003; Weidenspointner et al. 2003). In order to detect a spectral line produced by an astrophysical source one has to be able to (a) separate the continuum and line contributions to the spectrum and (b) separate the instrumental and sky signal contributions to the lines found.

One can expect three a priori situations:

- (I) DM decay line is strong (its equivalent width much larger than the spectral resolution) and at its position there are no other strong lines (of either instrumental or astrophysical origin). Such a line, due to its presence in any SPI spectrum and its low variability over the sky can in principle be confused with some unknown instrumental line.
- (II) DM line is weak ( ∼ 3 -4 σ detection over the continuum) but its position also does not coincide with any instrumental line.

- (III) DM decay line coincides with some instrumental line. To be able find such a line we need to model SPI instrumental background.

To be able to work effectively with all these situations, we need to find the way to separate the source and background contributions.

## 3.1 Imaging

To distinguish source and background contribution to the signal, one often uses imaging capabilities of an instrument. If the size of a point or even an extended source on the sky is smaller than the size of the SPI FoV one can (at least, to some extent) use the imaging capabilities of the SPI instrument. In this case the coded mask, placed above the detector, partially screens the individual detectors from the source, so that the source at a given position on the sky produces different count rates in different detectors. One can find the source flux by comparing the ratios of the actual count rates in different modules of the detector to the ones predicted by the degree of screening of the modules by the mask (see Dubath et al. 2005; Skinner &amp; Connell 2003). It is a challenge, however, to use the imaging capabilities of the SPI to separate the astrophysical signal from the instrumental background if the size of the extended source is comparable to the size of the SPI FoV (see e.g. Kn¨ odlseder et al. 2005; Allain &amp; Roques 200 6; Weidenspointner et al. 2007, and refs. therein). Therefore for our analysis we did not use any imaging capabilities of SPI, and to produce spectra from some point in the sky we just collected all the photons, arriving in the SPI FoV.

## 3.2 SPI background modeling

In the absence of imaging, the separation of the instrumental and astrophysical contributions to the line spectrum requires some sort of background modeling (see e.g. Weidenspointner et al. 2003; Teegarden et al. 2004; Teegarden &amp; Watanabe 2006). Namely, for the background modeling we can use the fact that for any DM distribution model the intensity of the DM decay line changes by a factor /greaterorequalslant 3 between the pointings towards the Galactic center ( φ ∼ 0 ◦ ) and anti-center ( φ ∼ 180 ◦ , see Sec.2.1). On the other hand, if the line is of purely instrumental origin, there is no a-priori reason why the strength of the line in the background spectra of the pointings towards e.g. the Galactic Anti-center should be different from the strength of the line in the spectra of the pointings toward e.g the GC. Thus, one possible way to distinguish between the DM decay and instrumental origin of the line is to study the variations of the line's strength depending on its sky position (in the simplest case on the 'off-GC' angle φ , of the pointing, Eq. (5)).

The situation becomes more complicated due to the fact that the instrumental background (and thus the intensity of the instrumental lines) experiences great variability in time (depending on the position in orbit, solar flares and the solar activity period, degradation of the detectors, etc., c.f. Jean et al. 2003; Teegarden et al. 2004). As observations of different parts of the sky can be significantly separated in time, one needs to use 'background tracers' to find the correct spatial dependence of the line intensity (Jean et al. 2003; Teegarden &amp; Watanabe 2006). Without some sort of 'renormalization' procedure, which corrects the absolute value of the line flux using a measurement of a specific characteristics of the SPI instrument as a 'calibrator' of the flux, the φ dependence for any of the detected lines contains no useful information. There exist various 'background tracers' (Ge detectors saturation rates, anti-coincidence shield rates, rates of certain background lines, see Jean et al. (2003); Teegarden et al. (2004); Teegarden &amp; Watanabe (2006) and refs. therein).

## 3.3 Searching for the lines

To be able to detect strong DM line, which is not close in position to any instrumental line (case I above), we used the modification of the method of background subtraction, described in TW06. TW06 looked for γ -ray lines, assuming different types of sources, from the point sources to the very diffuse sources ( 10 ◦ Gaussian, 30 ◦ flat, etc.) TW06 showed that the strong background line at 198 keV can be used as a background tracer, if background observations are matched close in time to the corresponding 'source' ones. This allowed TW06 to cancel all strong instrumental lines with the precision better than 1%. TW06 detected no emission line in such background subtracted spectrum (apart from the 511 keV and 1809 keV) with the significance above 3 . 5 σ .

We adopt the following modification of the TW06 method:

-As the DM decay signal remains nearly constant within central 30 -50 ◦ , the method of TW06, if applied directly, could cancel most of the DM signal. 8 We therefore subtract the data (renormalized by the strength of 198 keV line) in the direction away from the GC (off-GC angle φ &gt; 120 ◦ ) from the ON-GC dataset (the angle φ /lessorequalslant 13 ◦ ).

-In the resulting 'ON-OFF' spectrum we perform the search for the line with the significance higher than 3 σ .

This procedure allows to eliminate strong instrumental lines with the precision better than few percents. At the same time any strong DM line would remain in the 'ON-OFF' spectrum. Indeed, even for the flattest profile (Section 2.1), the strength of the DM signal in the OFF dataset is at least 60% weaker that of the ON dataset. Therefore we see, that the modification, described above, is indeed well suited for searching of the strong DM decay line (case I ).

However, this method does not work well for the weak ( 3 -4 σ ) lines, or for the lines, whose position coincides with some instrumental line (cases II-III above). Indeed, in this case it is not possible to tell whether the remaining line is the residual of the instrumental one or has the astrophysical origin. Below we will use an alternative method of analysis of the detected lines, suitable for cases II and III .

## 3.4 Analyzing a candidate line

Having detected a number of lines with the significance of 3 σ and above, we should decide which of them can be considered as 'DM decay line candidates'. To this end we do the following.

a) We compare line flux for each of these lines with the flux of the same line in the 'ON' spectrum. We decide that the line is a 'DM line candidate' if the cancellation of the flux between ON and OFF datasets was worse than 10%. 9

8 For example for the most conservative DM distribution model, the difference of DM signals at φ = 0 ◦ and φ = 30 ◦ is mere 8%.

9 In principle, the DM line in ON-OFF spectrum should not cancel by more than ∼ 40% , while the background instrumental line should cancel better than 1%. Thus the choice of the threshold to be around 10% ensures that no DM decay line was thrown away while most of the instrumental lines disappeared.

## 8 A. Boyarsky et al.

b) For any 'DM candidate line' we construct its 'spatial profile' (as described in details in the next Section) to check for the possibility of it to be a DM decay line (we also construct distribution of the line flux over the sky for all the unidentified lines from Weidenspointner et al. 2003). Since the column density of the DMin the direction toward the GC should be higher than that of in the direction toward the Galactic anti-center, one should see a gradual decrease of the line strength with the increasing angle φ . We do not make any specific assumption about the DM density profile and do not try to fit the candidate line spatial profile to any particular model, but rather look if there is a general trend of decreasing intensity of the line with the increasing off-GC angle.

## 4 DATA REDUCTION

## 4.1 ON dataset

During its almost 5 years in orbit INTEGRAL has intensively observed the inner part of the Galaxy (Galactic Center, Galactic Bulge and the inner part of the Galactic Plane) and collected about T exp ∼ 10 Ms of exposure time in the GC region. In our analysis of the inner Galaxy we used the publicly available data (as of July 2007) from all INTEGRAL pointings at which the angle off the GC was at most 13 ◦ and for which the SPI exposure time was larger than 1 ksec. This criteria selects 5355 pointings (or 'Science Windows', ScW), with total exposure time of 12.2 Ms, spread over the period from February 2003, till April, 25, 2006. We call this dataset "ON" dataset.

For each of the analyzed ScWs, we have extracted photon (event) lists from spi-oper.fits files and applied additional energy correction to convert the channel number into photon energy, using spi gain cor tool from standard Offline Analysis Software (OSA). We have binned the events into narrow energy bins of the size ∆ E bin = 0 . 5 keV to generate the background counts spectra in each ScW, each revolution and, subsequently, in the entire data set.

We then applied the 'sliding spectral window' method to the line search (as described e.g. in TW06 to produce a continuum subtracted spectrum of the "ON" dataset. Namely, at each given energy E 0 , one defines an energy interval E 0 -2∆ E &lt; E &lt; E 0 +2∆ E , where ∆ E is the SPI spectral resolution at a given energy, as a 'line signal' energy band. For the (energy dependent) ∆ E we used approximate formula from SPI/INTEGRAL ground calibration of FWHM(Atti´ e et al. 2003):

<!-- formula-not-decoded -->

where F 1 = 1 . 54 , F 2 = 4 . 6 · 10 -3 , F 3 = 6 . 0 · 10 -4 and energy E is in keV . For E = 10 3 keV FWHM ≈ 2 . 3 keV .

For each energy bin centered at an energy E 0 we have defined the two adjacent energy intervals, E 0 -4∆ E &lt; E &lt; E 0 -2∆ E and E 0 +2∆ E &lt; E &lt; E 0 +4∆ E , and postulated that the sum of the count rates in these two adjacent energy bands gives the measure of the continuum count rate in the energy band around E 0 . Subtracting the sum of the count rates in the adjacent energy bands from the count rate in the 'line signal' energy band, we have calculated the continuum subtracted count rate at a given energy E 0 . Doing such procedure at all the energies 20 keV &lt; E 0 &lt; 8 MeV , we have produced a 'continuum subtracted' SPI background spectrum.In this spectrum we were able to identify most of the known instrumental lines (Weidenspointner et al. 2003).

Figure 6. Comparison of ON-OFF spectrum (thick solid line) with the 0 . 01 × the ON spectrum after the sliding window (thin dashed line). It can be seen that the instrumental lines are subtracted with the precision better than 1%.

<!-- image -->

## 4.2 ON-OFF dataset

Most of the lines found in the continuum subtracted background spectrum are of the instrumental origin. To remove them, we matched each ScW in the ON dataset with the pointing away from the GC (galactic coordinate φ &gt; 120 ◦ ) - OFF pointing. As described by TW06, the 198 keV line can serve as good background tracer if the time duration between ON and OFF observations is /lessorequalslant 20 days. We were able to match 3688 ON-OFF pairs. For each ON-OFF pair we introduced normalizing coefficient n for the OFF spectrum in such a way that the strong instrumental line at 198 keV cancels completely after subtraction of the OFF spectrum multiplied by the factor n from the ON spectrum. After that we subtracted (renormalized) OFF ScW from the corresponding ScW from the ON dataset. This allowed us to remove most prominent instrumental lines with the precision better than 1% (c.f. Fig. 6). To avoid contributions of strong astrophysical sources (such as e.g. Crab) we threw out all pairs with negative total flux at 2040 keV range after subtraction. Taking average over 2456 remaining 'good' pairs we received the spectrum almost free from background at energies above 200 keV. At low ( &lt; 200keV ) energies we found continuum component, which can be fitted with the simple power law:

<!-- formula-not-decoded -->

Parameters of this background were found to be

<!-- formula-not-decoded -->

This continuum represents the residual contribution from all the set of the astrophysical sources present in the Galactic Bulge.

## 4.3 Systematic error

To estimate the systematic error of our 'ON-minus-OFF' dataset, we computed background around the 'tracer line' of 198 keV. We found that it does not vanish. Thus, we estimated the systematic error as the error in the normalization coefficient n which would make the background zero within systematic uncertainty. This correction δn can be found as follows. Let n be the coefficient, needed

Table 2. Splitting revolutions into phases in correspondence with annealing phases and breakage of the detectors (revolutions, marked with the ∗ ).

| Phase   | Revolutions (start-stop)   |
|---------|----------------------------|
| 1       | 042-092                    |
|         | 096-140 ∗                  |
| 2       | 140-205                    |
|         | 209-215 ∗                  |
| 3       | 215-277                    |
| 4       | 282-326                    |
| 5       | 330-395                    |
| 6       | 400-446                    |

to cancel flux in the 198 keV line in ON and OFF spectra:

<!-- formula-not-decoded -->

where F ON , F OFF - fluxes in 198keV line in ON and OFF ScWs correspondingly. The remaining non-zero δF flux in the adjacent to line position in ON-OFF spectrum, determines the uncertainty of the coefficient:

<!-- formula-not-decoded -->

We found, that average value of 〈 δn 〉 is equal to 〈 δn 〉 = 1 . 1 · 10 -3 , 〈 n 〉 ∼ 1 . So, our systematic error of final ON-OFF spectra at energy E is 1 . 1 · 10 -3 F OFF ( E ) ≈ 1 . 1 · 10 -3 F ON . We add this systematic uncertainty to the flux of ON-OFF spectrum in every energy bin.

## 4.4 Obtaining 3 σ restrictions

At the energies at which no lines were detected (i.e. the 'continuum subtracted' count rate did not deviate by more than 3 σ from zero) we obtained the 3 σ upper limit on the possible flux from the DM decay. Above ∼ 200 keV the flux in the energy bin is zero within statistical errors, therefore 3 σ upper limit flux is given by statistical plus systematic errors. Below 200keV we put statistical restrictions above power law continuum flux (20), described in the Section 4.2. Using Eq. (17) one can derive the restriction on the sterile neutrino mixing angle, implied by this upper limit. One should also take into account that the subtraction of the OFF observations led to the reduction of the expected DM signal. Taking the most conservative 'minimal' model, described in Section 2.1, we see that the subtraction of the OFF signal leads to about 40% decrease of the expected DMsignal. 10 The resulting 3 σ bound is shown on Fig.10.

## 4.5 Possible DM candidates

When analyzing ON-OFF spectrum, we found that almost all lines, present in ON spectrum cancel with precision better than few percents. We found 21 lines (see Table 3) that did not cancel by at least 90%, (including known lines at 511 keV and 1809 keV). Apart from these 2 lines all other lines are detected with low significance 3 -4 σ .

As discussed in Sections 3.3-3.4, we took all these lines as possible DM candidates and analyzed the dependence of the line fluxes F ( φ ) on the off-GC angle φ of the pointing. If the DM distribution in the inner part of the Galaxy were known, it would be

10 To estimate this, we took the maximal column density for OFF observations at φ = 120 ◦ -17 ◦ .

Figure 7. Position on the sky as a function of revolution over 6 years of INTEGRAL observations. The periods of annealing phases are shown in solid vertical lines. Two dashed lines indicate the revolution, during which 2ND and 17th of 19 SPI detectors have failed.

<!-- image -->

Figure 8. Scatter of the flux data points for the line at E = 1068 keV as a function of the off-GC angle.

<!-- image -->

possible to distinguish between the instrumental and DM decay origin of a line by fitting F ( φ ) with a known profile calculated from the radial DM density profile. However, the details of the radial DM density profile in the inner Galaxy are highly uncertain, and this prevents us from directly fitting the model profile to the data. We adopted a simple criterion which selects a DM decay candidate line: the ratio of fluxes

<!-- formula-not-decoded -->

where R min is the ratio of the DM decay line fluxes from the GC and the Galactic anti-center in the 'minimal DM content' model of DMdistribution.

Since the observations at different off-GC angles are done during different time periods, to properly study the dependence of the line flux on the off-GC angle φ one should take into account the time variability of the response of the SPI detectors. Several factors have to be taken into account. First, the SPI instrument goes through a so-called 'annealing' phase - heating of the detectors to

Figure 9. Dependence of the intensity of the positron annihilation line at E = 511 keV on the off-GC angle. The solid line shows fit to the data in the form const + Ne -φ 2 / (2 σ 2 ) .

<!-- image -->

recover from a radiative damage. 11 Next, two of the 19 SPI detector have 'died'. 12 The failed detectors also affect the response of their neighbors. To marginalize the effects of the changing response of the SPI detector, we split the entire data set into 7 periods, as shown on Fig. 7. The intervals are summarized in the Table 2. As both detector failures occurred soon after the end of an annealing phase, we chose to ignore revolutions 136 through 140 and 209 through 215. The period 096-140 does not cover the essential part of the sky and therefore we skip it, leaving only 6 periods.

For each of the periods, shown in the Table 2, we plot the distribution of the line flux as a function of the off-GC angle φ . The results are summarized on Fig. 13, p. 16. One can see that none of these lines exhibits clear trend of decreasing from φ = 0 ◦ towards φ = 180 ◦ . For each line (and each phase) we also compute the average flux ¯ F , standard deviation σ F from the average, minimum ( F min) and maximum ( F max). Our analysis shows that (a) 95 -100% of all points lie within 3 σ F from the average (thus, the data is consistent with having flat spatial profile) and (b) the scatter of the data ( F max -F min) is much less than its mean value 〈 F 〉 . Therefore, none of them cannot originate entirely from a DM decay. The corresponding numbers for each line and each phase are summarized in Table 4, page 17.

The positron annihilation line at E = 511 keV illustrates a situation, when a line of astrophysical origin is superimposed on top of the strong instrumental line. In this case, the data can be fitted by the constant, plus some function, depending on assumed shape of the source. Fig. 9 shows the dependence of the flux of the 511 keV line on the off-GC angle of the SPI pointing. One can see that for the pointings with the off-GC angle less than 20 ◦ (about the size of the PCFOV of SPI) the 511 keV line flux contains a contribution from a sky source at the position of the GC, while for the pointing at larger off-GC angles the astrophysical source is not visible and the only contribution comes from the instrumental line, whose flux does not depend on the off-GC angle of the pointing.

11

For details see SPI User Manual: http://isdc.unige.ch/Instrument/spi/doc/spi um.

12 Detector # 2 at revolution 140 and detector #17 at revolutions 214-215.

Figure 11. Life-time of the radiatively decaying DM as a function of the emitted photon energy. Region below the curve is excluded.

<!-- image -->

## 5 RESULTS

We analyzed the spectrum of SPI and found that none of the strong (i.e. detected with significance above 5 σ ) lines can be interpreted as that of the decaying DM. This conclusion was based on the fact that variability of these lines over the sky is less than 10% (when moving from GC to the anti-center, see Fig.8). At the same time for any realistic DM model such a variability would be greater than at least 60%. Thus, we exclude the possibility that one of the spectral lines, detected in the SPI background spectrum is a DM decay line.

The non-detection of a DM decay line in the entire energy range of the SPI detector has enabled us to put an upper limit on the parameters of the DM particles. In particular, the 3 σ upper bound on the mixing angle of the sterile neutrino DM in the mass range 40 keV - 7 MeV is shown on Fig. 10, p. 12.

Our results are applicable to any decaying DM. To this end we also present the restrictions on the DM life-time (with respect to the radiative decay) as a function of the energy of emitted photon. The corresponding exclusion plot is shown on Fig. 11. For example, the gravitino can decay into the neutrino and photon (similarly to the case of sterile neutrino) in supersymmetric theories with broken Rparity. Such an interaction is generated via the loop effects (see e.g. Borgani et al. 1996; Lola et al. 2007). The restrictions on Fig. 11 improve existing bounds on the life-time of such a gravitino DM by several orders of magnitude (c.f. Borgani et al. 1996).

To present our results in the form less dependent on a particular model of DM distribution in the MW, we show the 3 σ sensitivity towards the line search on Fig. 12. Note, that these results should be used with care, as the sensitivity depends on the assumed spatial profile of the source (because the effective area decreases with the off-axis angle, see discussion in Section 2). The results, presented on Fig. 12 are valid for an extended source with the surface brightness which varies on the angular scales larger than (or comparable to) the size of the SPI field of view (black solid line). This plot is analogous to the Fig. 9 of Teegarden &amp; Watanabe (2006) (TW06). However, a direct comparison of the Fig. 9 of TW06 and Fig. 12 is not possible, since TW06 have assumed a different morphology of the extended source ( 10 ◦ Gaussian). Explicitly taking into account the dependence of the effective area of the SPI detector on the offaxis angle (see Section 2), one can find that in order to make a direct comparison between the two figures, one has to 're-scale' the results of Fig. 9 of TW06 by an (energy dependent) factor of ≈ 1 . 5 . This factor converts the sensitivity for the line, produced by

Table 3. Lines, detected in the ON-OFF spectrum with the significance /greaterorequalslant 3 σ . Lines, marked with † cancel worse than by 90% in the ON-OFF spectrum (as compared with their flux in ON spectrum) and thus represent a 'DM candidates'. The 'Identification' column indicates the probable identification of the line in Weidenspointner et al. (2003). Lines, marked with '?' are not identified in Weidenspointner et al. (2003), lines marked with '??' are not present in Weidenspointner et al. (2003).

| E [keV]   |   Sign., σ |   ∆ E [keV] | Identification     |
|-----------|------------|-------------|--------------------|
| 68.5      |       11.4 |        0.65 | 66.7 Ge complex    |
| 76.5      |       58.8 |        1.1  | 75 Bi K α          |
| 87        |       21.2 |        0.9  | 87 Bi K β          |
| 94        |        5.2 |        0.55 | 91-105 GaZn        |
| 134.5     |       15.5 |        0.9  | 132-140 Ge complex |
| 143       |       12.6 |        1.2  | 140-147 Ge complex |
| 177       |        3.9 |        0.95 | 175 AsGe           |
| 186.5     |        7.3 |        1.1  | 184.6 GaZn         |
| 193       |       24.8 |        0.75 | 190-198 Ge complex |
| 200       |       29.3 |        0.6  | 198-215 Ge complex |
| 205.5     |        5.4 |        0.5  | 198-215 Ge complex |
| 240       |        3.4 |        1.1  | 238 PbBi           |
| 302       |        3.9 |        0.7  | 301.5 GaZn         |
| 311.5     |        7.4 |        1.1  | 309.8 GaZn+K       |
| 330.5     |        3.2 |        0.55 | 328=? or 331=PbTl  |
| 385.5     |        3.2 |        1.45 | 383=PbTl or ??     |
| 404.5     |        5.5 |        0.95 | 403 Ga Zn+K        |
| 431.5     |        4.4 |        0.55 | ??                 |
| 440.5     |       12.4 |        0.8  | 438 ZnZn           |
| 465       |        4.3 |        0.95 | 470-485 NaNa       |
| 511 †     |       52.5 |        1.25 | 511 e + e -        |
| 576       |        5.2 |        0.95 | 574 GeGa           |
| 585.5     |        7.2 |        1.1  | 584.5 GeGa+K       |
| 597.5     |        5   |        1.45 | 596-610 Ge complex |
| 754       |        4.7 |        0.85 | 751 BiBi           |
| 803.5     |        3.7 |        0.75 | 803 BiPb           |
| 812       |       10.6 |        0.95 | 810 CoFe           |
| 819.5     |       11.6 |        0.85 | 817 CoFe+K         |
| 827.5     |        7.1 |        0.75 | 825 PbPb           |
| 836       |       12.1 |        0.95 | 834 MnCr           |
| 845       |        6.5 |        1.3  | 843 MgAl           |
| 874       |        8.8 |        1.05 | 872 GeGa           |
| 884       |       10   |        1.05 | 882 GeGa+K         |
| 913       |        5.2 |        1.05 | 911 AcTh           |
| 937 †     |        3.1 |        0.95 | ?? or 935=MnCr     |

| E [keV]              | Sign., σ      | ∆ E [keV]      | Identification                   |
|----------------------|---------------|----------------|----------------------------------|
| 990                  | 6.0           | 0.85           | 987 PbPb                         |
| 947 †                | 3.6           | 1.50           | ??                               |
| 1014.5               | 11.1          | 1.40           | 1014 MgAl                        |
| 1068.5               | 3.3           | 1.40           | ?? or 1063=PbPb                  |
| 1079                 | 3.3           | 1.00           | 1077=GaZn                        |
| 1098                 | 6.8           | 1.05           | 1095=?                           |
| 1108.5               | 19.9          | 1.05           | 1106 GeGa                        |
| 1118.5               | 23.9          | 1.05           | 1117 GeGa+K                      |
| 1127                 | 23.5          | 0.85           | 1124.5 ZnCu+K                    |
| 1234 †               | 4.5           | 1.40           | 1231 TaW                         |
| 1349.5               | 3.5           | 0.80           | 1347 GeGa+K                      |
| 1368.5               | 13.2          | 1.55           | 1368 NaMg                        |
| 1719 . 5 †           | 3.3           | 1.55           | 1719 BiPb                        |
| 1753.5 1767.5 1781.5 | 4.0 11.1 12.4 | 1.45 1.40 1.45 | ?? or 1758=? 1764 BiPb 1778 AlSi |
| 1809 † 1904          | 15.2 3.2      | 1.85 1.45      | 1808 Mg 1901=GeGa+K or ??        |
| 2212                 | 6.7           | 2.20           | 2195-2223 BiPo, Al               |
| 2225                 | 5.4 3.1       | 1.10           | 2223 HD                          |
| 2322                 | 3.5           | 1.55           | 2319=?                           |
| 2583.5 2616 2756     | 5.0 9.5       | 1.65 1.90      | 2599=? or ?? 2614 PbTl           |
| 3002.5               | 5.1           | 1.90 2.45      | 2754 NaMg 2993-3013 Al           |
| 3176 . 5 † †         | 3.4           | 1.95           | ??                               |
|                      |               |                | ??                               |
| 3331                 |               |                |                                  |
| 3802                 | 3.4           | 2.05 2.10      | 3800 GaZn+K                      |
| 4307.5               | 3.6           | 3.40           | 4304                             |
|                      | 3.6           |                |                                  |
| 4454                 |               |                | GaZn+K                           |
|                      | 4.2           | 8.45           | 4434 C                           |
| 4738 †               | 3.5           | 2.30           | ??                               |
| 5186 . 5 †           | 3.5           | 2.20           | ??                               |
| 5208 . 5 †           | 4.0           | 1.95           | ??                               |
| 5757 †               | 3.4           | 2.95           | ??                               |
| 6129                 | 20.5          | 3.25           | 6128.9 O                         |

a source with the Gaussian surface brightness profile, into the one produced by a source of approximately constant surface brightness (see red curve in Fig. 12).

We have found a number of weak (with the significance 3 -4 σ ) lines in the background-subtracted spectrum of SPI. These lines cancel by worse than 90% when subtracting OFF dataset (see Section 4). Apart from it we have found in the backgroundsubtracted spectrum two lines with high significance - known lines at 511 keV and at 1809 keV. Any of these lines can in principle be a DM decay line. We analyzed each of them, by considering the profile of their intensity over the sky. Our analysis shows that none of these lines could be pure DM line (as their dependence on the off-GC angle does not show any clear trend to decrease towards the anti-center). The possibility that some of these lines are the superposition of instrumental and DM lines remains open. Quantitative analysis of the amount of DM flux admissible in a given line depends strongly on the model of the DM distribution in the Milky Way halo. Therefore it was not conducted here.

## 6 DISCUSSION

The purpose of this work was to understand how to search for the DMdecay line with the SPI spectrometer and to check that none of the strong lines, present in the SPI background, was confused with the DM decay line. Our analysis shows that all the strong lines were, indeed, of instrumental origin and provides the upper bound on the flux of 'weak' ( 3 -4 σ above the background) lines, which leads to the corresponding restrictions (see Sec. 5). To further improve the results, one needs to work with the weak lines (or lines, coinciding in position with instrumental ones). To do this one needs more sophisticated procedures of subtraction of the instrumental background (e.g. imaging).

One of the most interesting cases of the coinciding instrumental and celestial line is the positronium annihilation line at 511 keV. An excess of positron annihilation emission on top of the strong instrumental line (related to positrons annihilating inside the detector) was noticed long ago (for an incomplete set of references see e.g. Prantzos 1993; Milne et al. 1999; Cheng et al. 1997; Purcell et al. 1997; Kn¨ odlseder et al. 2005; Weidenspointn er et al. 2006, 2007). There exist many attempts of explanation of this excess. In particular, it was attributed to the annihilating or decaying DM (see e.g. Boehm et al. 2004; Hooper et al. 2004; Boehm et al.

Figure 10. Upper bound on the mixing angle of the DM sterile neutrino as a function of the sterile neutrino mass, obtained from the analysis of the background spectrum of the pointings toward the inner 13 ◦ of the Galaxy. For masses /lessorequalslant 700keV the restrictions from Y¨ uksel et al. (2008) (divided by the f actor 2, due to the Majorana nature of the DM) are also shown in dashed line (see Discussion). For masses &lt; 100 keV previous restrictions from HEAO-1 (Boyarsky et al. 2006c) are also shown. The region above the curve is excluded.

<!-- image -->

Figure 12. 3 σ sensitivity towards the line search in case of the flat diffuse source (thick black line). The results of Teegarden &amp; Watanabe (2006, Fig. 9), (rescaled to account for sensitivity towards the flat diffuse source, rather than 10 ◦ Gaussian) are shown in thin red line.

<!-- image -->

2006; Fr` ere et al. 2007; Picciotto &amp; Pospelov 2005; Rasera et al. 2006). The sterile neutrino DM with the mass m s &gt; 1 MeV pos- sesses decay channel N s → e + e -ν , with positrons annihilating either in flight or at rest, by forming the positronium atom (see e.g. Beacom &amp; Y¨ uksel 2006; Sizun et al. 2006). Thus, it is possibl e that the decay of sterile neutrino DM contributes to such a line. The detailed analysis of this case will be reported separately.

It should be also mentioned, that the region of masses between 20 keV /lessorsimilar m DM /lessorsimilar 40 keV remains inaccessible for the existing X-ray missions. The strongest restrictions in this region were produced, using the data of HEAO-1 mission (Boyarsky et al. 2006c).

When the work on this paper was at its final stage, the work of Y07 was published. Y07 obtained the restrictions on parameters of sterile neutrino in the range 40 keV - 700 keV. To facilitate the comparison, we plot the restrictions of Y07 on Fig. 10, (divided by the factor of 2 to translate them into the restrictions for the Majorana, rather than Dirac sterile neutrino DM, see footnote 7, p 6). As the data, used in our work, has about 5 times longer exposure than the INTEGRAL first years data, on which the results of Y07 are based, we could have expected results stronger by a factor ≈ 2 in our case. However, the Fig. 10 shows the opposite. The reason for this is as follows. For the SPI, the sensitivity towards the line search from a particular source depends on the shape of the source. In particular, the results of TW06, on which the work of Y07 was based, were obtained under the assumption of a particular diffuse source

( 10 ◦ Gaussian). As any realistic DM profile is much flatter than the 10 ◦ Gaussian, the results of TW06 cannot be applied directly for the case of the DM line search. They should be rescaled to account for the diffuse nature of the DM source (c.f. Section 5). Apart from this, the estimated DM signal from the inner part of the Galaxy is about 2 times stronger in Y07 than in our work. As the DM signal in the direction of the GC is the most uncertain, we have adopted the conservative flat profile everywhere inside the solar radius, to minimize this uncertainty.

## Acknowledgments

We would like to thank B. Teegarden and K. Watanabe for useful discussion. D.M. is grateful to the Scientific and Educational Center 13 of the Bogolyubov Institute for Theoretical Physics in Kiev, Ukraine, and especially to V. Shadura, for creating wonderful atmosphere for young Ukrainian scientists, and to Ukrainian Virtual Roentgen and Gamma-Ray Observatory VIRGO.UA 14 and computing cluster of Bogolyubov Institute for Theoretical Physics 15 , for using their computing resources. The work of D.M. was supported by the Swiss National Science Foundation and the Swiss Agency for Development and Cooperation in the framework of the programme SCOPES - Scientific co-operation between Eastern Europe and Switzerland. The work of A.B. was (partially) supported by the EU 6th Framework Marie Curie Research and Training network 'UniverseNet' (MRTN- CT-2006-035863). O.R. would like to acknowledge support of the Swiss Science Foundation.

## REFERENCES

- Abazajian, K. N., Markevitch, M., Koushiappas, S. M., &amp; Hickox, R. C. 2007, Phys. Rev. D, 75, 063511, ADS, arXiv:astro-ph/0611144
- Alard, C. 1999, A&amp;A, 343, 10, ADS, arXiv:astro-ph/9808092
- Alcock, C., et al. 2000, ApJ, 541, 270, ADS
- Allain, M., &amp; Roques, J.-P. 2006, A&amp;A, 447, 1175, ADS
- Asaka, T., Laine, M., &amp; Shaposhnikov, M. 2007, JHEP, 01, 091, hep-ph/0612182
- Asaka, T., &amp; Shaposhnikov, M. 2005, Phys. Lett., B620, 17, hep-ph/0505013
- Atti´ e, D. et al. 2003, A&amp;A, 411, L71, ADS, arXiv:astro-ph/0308504
- Avila-Reese, V., Col´ ın, P., Valenzuela, O., D'Onghia, E., &amp; Firmani, C. 2001, ApJ, 559, 516, ADS, arXiv:astro-ph/0010525
- Bahcall, J. N., &amp; Soneira, R. M. 1980, ApJS, 44, 73, ADS
- Baltz, E. A., &amp; Murayama, H. 2003, JHEP, 5, 67, ADS, arXiv:astro-ph/0108172
- Barger, V. D., Phillips, R. J. N., &amp; Sarkar, S. 1995, Phys. Lett., B352, 365, hep-ph/9503295
- Battaglia, G., et al. 2005, Mon. Not. Roy. Astron. Soc., 364, 433, astro-ph/0506102
- Beacom, J. F., &amp; Y¨ uksel, H. 2006, Phys. Rev. Lett., 97, 07110 2, ADS, arXiv:astro-ph/0512411
- Berezhiani, Z. G., &amp; Khlopov, M. Y. 1990, Sov. J. Nucl. Phys., 52, 60
- Berezhiani, Z. G., Vysotsky, M. I., &amp; Khlopov, M. Y. 1987, Sov. J. Nucl. Phys., 45, 1065
- 13 http://sec.bitp.kiev.ua
- 14 http://virgo.bitp.kiev.ua
- 15 http://grid.bitp.kiev.ua
- Berezhiani, Z. G., Vysotsky, M. I., Yurov, V. P., Doroshkevich, A. G., &amp; Khlopov, M. Y. 1990, Sov. J. Nucl. Phys., 51, 1020
- Bergstrom, L. 2000, Rept.Prog.Phys., 63, 793, hep-ph/0002126
- Bertone, G., Buchmuller, W., Covi, L., &amp; Ibarra, A. 2007, arXiv:0709.2299 [astro-ph]
- Bertone, G., Hooper, D., &amp; Silk, J. 2005, Phys. Rept., 405, 279, hep-ph/0404175
- Bode, P., Ostriker, J. P., &amp; Turok, N. 2001, ApJ, 556, 93, astro-ph/0010389
- Boehm, C., Hooper, D., Silk, J., Casse, M., &amp; Paul, J. 2004, Phys. Rev. Lett., 92, 101301, astro-ph/0309686
- Boehm, C., Orloff, J., &amp; Salati, P. 2006, Physics Letters B, 641, 247, ADS, arXiv:astro-ph/0607437
- Bond, J. R., Efstathiou, G., &amp; Silk, J. 1980, Phys. Rev. Lett., 45, 1980, ADS
- Bond, J. R., Szalay, A. S., &amp; Turner, M. S. 1982, Phys. Rev. Lett., 48, 1636, ADS
- Borgani, S., Masiero, A., &amp; Yamaguchi, M. 1996, Phys. Lett., B386, 189, hep-ph/9605222
- Boyarsky, A., den Herder, J. W., Neronov, A., &amp; Ruchayskiy, O. 2007, Astropart. Phys., 28, 303, ADS, astro-ph/0612219
- Boyarsky, A., Iakubovskyi, D., Ruchayskiy, O., &amp; Savchenko, V. 2008, MNRAS, 387, 1361
- Boyarsky, A., Neronov, A., Ruchayskiy, O., &amp; Shaposhnikov, M. 2006a, MNRAS, 370, 213, ADS, astro-ph/0512509
- -. 2006b, Phys. Rev. D, 74, 103506, astro-ph/0603368
- Boyarsky, A., Neronov, A., Ruchayskiy, O., Shaposhnikov, M., &amp; Tkachev, I. 2006c, Phys. Rev. Lett., 97, 261302, astro-ph/0603660
- Boyarsky, A., Nevalainen, J., &amp; Ruchayskiy, O. 2007, A&amp;A, 471, 51, ADS, astro-ph/0610961
- Boyarsky, A., Ruchayskiy, O., &amp; Markevitch, M. 2006d, Submitted to ApJ., ADS, astro-ph/0611168
- Carr, J., Lamanna, G., &amp; Lavalle, J. 2006, Reports of Progress in Physics, 69, 2475, ADS
- Cembranos, J. A. R., Feng, J. L., Rajaraman, A., Smith, B. T., &amp; Takayama, F. 2006, ADS, hep-ph/0603067
- Cerdeno, D. G., Choi, K.-Y., Jedamzik, K., Roszkowski, L., &amp; Ruiz de Austri, R. 2006, JCAP, 0606, 005, hep-ph/0509275
- Cheng, L. X. et al. 1997, ApJ, 481, L43+, ADS
- Chung, D. J. H., Kolb, E. W., &amp; Riotto, A. 1999, Phys.Rev. D, 59, 023501, hep-ph/9802238
- Crotty, P., Lesgourgues, J., &amp; Pastor, S. 2004, Phys. Rev., D69, 123007, hep-ph/0402049
- Dalcanton, J. J., &amp; Hogan, C. J. 2001, ApJ, 561, 35, astro-ph/0004381
- Dar, A. 1995, ApJ, 449, 550, ADS, arXiv:astro-ph/9504082
- de Rujula, A., &amp; Glashow, S. L. 1980, Phys. Rev. Lett., 45, 942, ADS
- Diehl, R. et al. 2003, A&amp;A, 411, L117, ADS
- Diemand, J., Kuhlen, M., &amp; Madau, P. 2007, ApJ, 657, 262, ADS, astro-ph/0611370
- Dodelson, S., &amp; Widrow, L. M. 1994, Phys. Rev. Lett., 72, 17, hep-ph/9303287
- Doroshkevich, A. G., Khlopov, M. I., &amp; Klypin, A. A. 1989, MNRAS, 239, 923, ADS
- Dubath, P. et al. 2005, MNRAS, 357, 420, ADS
- Dubovsky, S. L., Tinyakov, P. G., &amp; Tkachev, I. I. 2005, Phys. Rev. Lett., 94, 181102, ADS, arXiv:hep-th/0411158
- Fogli, G. L., Lisi, E., Marrone, A., Palazzo, A., &amp; Rotunno, A. M. 2006, Prog. Part. Nucl. Phys., 57, 71, ADS, arXiv:hep-ph/0506083

- Fr` ere, J.-M., Ling, F.-S., Honorez, L. L., Nezri, E., Swillens, Q., &amp; Vertongen, G. 2007, Phys. Rev. D, 75, 085017, ADS, arXiv:hep-ph/0610240
- Gilmore, G. 2007, ADS, astro-ph/0703370
- Gilmore, G., Wilkinson, M., Kleyna, J., Koch, A., Wyn Evans, N., Wyse, R. F. G., &amp; Grebel, E. K. 2006, ADS, astro-ph/0608528
- Gilmore, G., Wilkinson, M., Kleyna, J., et al. 2007, Nuclear Physics B Proceedings Supplements, 173, 15
- Gilmore, G., Wyse, R. F. G., &amp; Kuijken, K. 1989, ARA&amp;A, 27, 555, ADS
- Giunti, C. 2007, Nucl. Phys. Proc. Suppl., 169, 309, hep-ph/0611125
- Goerdt, T., Moore, B., Read, J. I., Stadel, J., &amp; Zemp, M. 2006, MNRAS, 368, 1073, ADS, astro-ph/0601404
- Gruber, D. E., Matteson, J. L., Peterson, L. E., &amp; Jung, G. V . 1999, ApJ, 520, 124, astro-ph/9903492
- Hannestad, S., &amp; Raffelt, G. 2004, JCAP, 0404, 008, hep-ph/0312154
- Hansen, S. H., Lesgourgues, J., Pastor, S., &amp; Silk, J. 2002, MNRAS, 333, 544, astro-ph/0106108
- Hooper, D., Ferrer, F., Boehm, C., Silk, J., Paul, J., Evans, N. W., &amp; Casse, M. 2004, Phys. Rev. Lett., 93, 161302, ADS, arXiv:astro-ph/0311150
- Hui, L., Gnedin, N. Y., &amp; Zhang, Y. 1997, ApJ, 486, 599, ADS, astro-ph/9608157
- Jean, P. et al. 2003, A&amp;A, 411, L107, ADS
- Khlopov, M. Y. 1997, Cosmoparticle Physics (World Scientific Pub Co Inc)
- Klimentowski, J., Lokas, E. L., Kazantzidis, S., Prada, F., Mayer, L., &amp; Mamon, G. A. 2007, MNRAS, 378, 353, ADS, arXiv:astro-ph/0611296
- Klypin, A., Kravtsov, A. V., Valenzuela, O., &amp; Prada, F. 1999, ApJ, 522, 82, ADS, arXiv:astro-ph/9901240
- Klypin, A., Zhao, H., &amp; Somerville, R. S. 2002, ApJ, 573, 597, ADS, astro-ph/0110390
- Kn¨ odlseder, J. et al. 2005, A&amp;A, 441, 513, ADS, arXiv:astro-ph/0506026
- Koposov, S. et al. 2007, ApJ, 663, 948, ADS, 0706.2687
- Kravtsov, A. V., Klypin, A. A., Bullock, J. S., &amp; Primack, J. R. 1998, ApJ, 502, 48, ADS, arXiv:astro-ph/9708176
- Kuijken, K., &amp; Gilmore, G. 1989a, MNRAS, 239, 605, ADS
- -. 1989b, MNRAS, 239, 651, ADS
- -. 1989c, MNRAS, 239, 571, ADS
- -. 1991, ApJ, 367, L9, ADS
- Kusenko, A. 2006, Phys. Rev. Lett., 97, 241301, hep-ph/0609081
- Kuzmin, V. A., &amp; Tkachev, I. I. 1998, JETP Lett., 68, 271, hep-ph/9802304
- -. 1999, Phys.Rept., 320, 199, hep-ph/9903542
- Lasserre, T., et al. 2000, A&amp;A, 355, L39, ADS, arXiv:astro-ph/0002253
- Lola, S., Osland, P., &amp; Raklev, A. R. 2007, Phys. Lett., B656, 83, arXiv:0707.2510 [hep-ph]
- McCammon, D. et al. 2002, ApJ, 576, 188, ADS, astro-ph/0205012
- Milne, A. P. et al. 1999, Astrophysical Letters Communications, 38, 441, ADS
- Moore, B. 1994, Nature, 370, 629, ADS
- Moore, B., Quinn, T., Governato, F., Stadel, J., &amp; Lake, G. 1999, MNRAS, 310, 1147, ADS, astro-ph/9903164
- Navarro, J. F., Frenk, C. S., &amp; White, S. D. M. 1997, ApJ, 490, 493, astro-ph/9611107
- Pagels, H., &amp; Primack, J. R. 1982, Phys. Rev. Lett., 48, 223, ADS
- Pal, P. B., &amp; Wolfenstein, L. 1982, Phys. Rev., D25, 766
- Palazzo, A., Cumberbatch, D., Slosar, A., &amp; Silk, J. 2007, arXiv:0707.1495 [astro-ph]
- Penarrubia, J., McConnachie, A., &amp; Navarro, J. F. 2007, ADS, astro-ph/0701780
- Petraki, K., &amp; Kusenko, A. 2007, arXiv:0711.4646 [hep-ph]
- Picciotto, C., &amp; Pospelov, M. 2005, Physics Letters B, 605, 15, ADS, arXiv:hep-ph/0402178
- Prantzos, N. 1993, A&amp;AS, 97, 119, ADS
- Purcell, W. R. et al. 1997, ApJ, 491, 725, ADS
- Rasera, Y., Teyssier, R., Sizun, P., Cass´ e, M., Fayet, P., Cordier, B., &amp; Paul, J. 2006, Phys. Rev. D, 73, 103518, ADS, arXiv:astro-ph/0507707
- Riemer-Sørensen, S., Hansen, S. H., &amp; Pedersen, K. 2006, ApJ , 644, L33, ADS, astro-ph/0603661
- Roszkowski, L., Ruiz de Austri, R., &amp; Choi, K.-Y. 2005, JHEP, 08, 080, hep-ph/0408227
- Ruchayskiy, O. 2007, in Proceedings of the 11th Marcel Grossmann Meeting on General Relativity, ed. H. Kleinert, R. Jantzen, &amp;R. Ruffini (World Scientific), arXiv:0704.3215 [astro-ph]
- S´ anchez-Conde, M. A., Prada, F., &amp; Łokas, E. L. 2006, in American Institute of Physics Conference Series, 125-131, ADS
- Seljak, U., Makarov, A., McDonald, P., &amp; Trac, H. 2006, Phys. Rev. Lett., 97, 191303, astro-ph/0602430
- Shaposhnikov, M. 2007, in Proceedings of the 11th Marcel Grossmann Meeting on General Relativity, ed. H. Kleinert, R. Jantzen, &amp;R. Ruffini (World Scientific), astro-ph/0703673
- Shaposhnikov, M., &amp; Tkachev, I. 2006, Phys. Lett., B639, 414, hep-ph/0604236
- Shi, X.-d., &amp; Fuller, G. M. 1999, Phys. Rev. Lett., 82, 2832, astro-ph/9810076
- Simon, J. D., &amp; Geha, M. 2007, 706, ADS, 0706.0516
- Sizun, P., Cass´ e, M., &amp; Schanne, S. 2006, Phys. Rev. D, 74, 06314, ADS, arXiv:astro-ph/0607374
- Skinner, G., &amp; Connell, P. 2003, A&amp;A, 411, L123, ADS, arXiv:astro-ph/0308444
- Spergel, D. N., et al. 2007, Astrophys. J. Suppl., 170, 377, astro-ph/0603449
- Strigari, L. E., Bullock, J. S., Kaplinghat, M., Diemand, J., Kuhlen, M., &amp; Madau, P. 2007, 704, ADS, 0704.1817
- Strumia, A., &amp; Vissani, F. 2006, hep-ph/0606054
- Tasitsiomi, A., Gaskins, J., &amp; Olinto, A. V . 2004, Astropart. Phys., 21, 637, astro-ph/0307375
- Teegarden, B. J. et al. 2004, in ESA Special Publication, Vol. 552, 5th INTEGRAL Workshop on the INTEGRAL Universe, ed. V. Schoenfelder, G. Lichti, &amp; C. Winkler, 819-+, ADS
- Teegarden, B. J., &amp; Watanabe, K. 2006, ApJ, 646, 965, ADS, arXiv:astro-ph/0604277
- Tremaine, S., &amp; Gunn, J. E. 1979, Phys. Rev. Lett., 42, 407
- Vedrenne, G. et al. 2003, A&amp;A, 411, L63, ADS
- Viel, M., Becker, G. D., Bolton, J. S., Haehnelt, M. G., Rauch, M., &amp;Sargent, W. L. W. 2007, 709, ADS, 0709.0131
- Viel, M., Lesgourgues, J., Haehnelt, M. G., Matarrese, S., &amp; Riotto, A. 2005, Phys. Rev., D71, 063534, ADS, astro-ph/0501562
- -. 2006, Phys. Rev. Lett., 97, 071301, astro-ph/0605706
- Watson, C. R., Beacom, J. F., Yuksel, H., &amp; Walker, T. P. 2006, Phys. Rev., D74, 033009, astro-ph/0605424
- Weidenspointner, G. et al. 2003, A&amp;A, 411, L113, ADS
- -. 2007, ArXiv Astrophysics e-prints, ADS, astro-ph/0702621
- -. 2006, A&amp;A, 450, 1013, ADS, arXiv:astro-ph/0601673
- Y¨ uksel, H., Beacom, J. F., &amp; Watson, C. R. 2008, Physical Rev iew Letters, 101, 121301, ADS, 0706.4084

Zhang, L., Chen, X., Kamionkowski, M., Si, Z., &amp; Zheng, Z. 2007, Phys. Rev. D, 76, 0613013(R), 0704.2444

Figure 13. Line flux as a function of the off-GC angle φ for the 'candidate' lines from Table 3 (page 11) . For all lines the flux does not depend on the off-GC angle (with 95 -100% of all points lying within 3 standard deviations from the average). Different colors represent different phases (see Table 2, p. 9).

<!-- image -->

Table 4. Characteristics of the spatial profiles of the candidate lines from Table 3. For each line (and for each of 6 phases) we compute the average 〈 F 〉 , the standard deviation (average scatter of the points around its mean value) σ F , minimal and maximal values and the ratio of ( F max -F min ) /F min, which gives the upper bound on the share of DM, present in the given line.

|   E [keV] |   〈 F 〉 | σ F             | F min           | F max           | F max - F min   |
|-----------|-----------|-----------------|-----------------|-----------------|-----------------|
|       330 |    0.0096 | 5.8e-04         | 8.7e-03         | 1.1e-02         | F min 0.23      |
|       330 |    0.0096 | 4.0e-04         | 8.9e-03         | 1.0e-02         | 0.15            |
|       330 |    0.01   | 3.4e-04         | 9.3e-03         | 1.1e-02         | 0.16            |
|       330 |    0.01   | 5.2e-04         | 9.7e-03         | 1.2e-02         | 0.23            |
|       330 |    0.01   | 4.7e-04         | 9.6e-03         | 1.2e-02         | 0.24            |
|       330 |    0.01   | 6.1e-04         | 9.7e-03         | 1.3e-02         | 0.33            |
|       385 |    0.0066 | 7.3e-04         | 5.4e-03         | 8.0e-03         | 0.48            |
|       385 |    0.0061 | 4.8e-04         | 5.2e-03         | 6.8e-03         | 0.31            |
|       385 |    0.0063 | 3.9e-04         | 5.3e-03         | 7.1e-03         | 0.32            |
|       385 |    0.0062 | 5.4e-04         | 5.2e-03         | 7.4e-03         | 0.42            |
|       385 |    0.0062 | 5.0e-04         | 5.5e-03         | 7.6e-03         | 0.40            |
|       385 |    0.0067 | 5.5e-04         | 5.8e-03         | 8.4e-03         | 0.44            |
|       431 |    0.011  | 5.3e-04         | 9.2e-03         | 1.2e-02         | 0.26            |
|       431 |    0.011  | 5.7e-04         | 9.5e-03         | 1.2e-02         | 0.26            |
|       431 |    0.012  | 1.0e-03         | 1.0e-02         | 1.5e-02         | 0.43            |
|       431 |    0.013  | 8.1e-04         | 1.1e-02         | 1.4e-02         | 0.32            |
|       431 |    0.014  | 1.4e-03         | 1.2e-02         | 1.8e-02         | 0.45            |
|       431 |    0.018  | 1.5e-03         | 1.4e-02         | 2.0e-02         | 0.45            |
|       511 |    0.045  | 7.8e-04         | 4.4e-02         | 4.7e-02         | 0.07            |
|       511 |    0.059  | 5.8e-04         | 5.8e-02         | 6.0e-02         | 0.04            |
|       511 |    0.07   | 8.5e-04         | 6.8e-02         | 7.2e-02         | 0.05            |
|       511 |    0.071  | 6.7e-04         | 6.9e-02         | 7.2e-02         | 0.04            |
|       511 |    0.073  | 8.7e-04         | 7.1e-02         | 7.4e-02         | 0.05            |
|       511 |    0.075  | 7.4e-04         | 7.3e-02         | 7.6e-02         | 0.03            |
|       937 |    0.0063 | 7.2e-04         | 5.2e-03         | 7.7e-03         | 0.47            |
|       937 |    0.0058 | 5.4e-04         | 4.7e-03         | 6.6e-03         | 0.40            |
|       937 |    0.0057 | 4.3e-04         | 4.9e-03         | 6.5e-03         | 0.33            |
|       937 |    0.0056 | 5.4e-04         | 4.6e-03         | 6.6e-03         | 0.44            |
|       937 |    0.0053 | 5.5e-04         | 4.7e-03         | 7.5e-03         | 0.60            |
|       937 |    0.0055 | 5.0e-04         | 4.6e-03         | 6.6e-03         | 0.44            |
|       947 |    0.006  | 7.2e-04         | 4.9e-03         | 7.4e-03         | 0.51            |
|       947 |    0.0054 | 5.8e-04         | 4.4e-03         | 6.5e-03         | 0.49            |
|       947 |    0.0053 | 4.4e-04         | 4.4e-03         | 6.1e-03         | 0.40            |
|       947 |    0.0052 | 5.0e-04         | 4.3e-03         | 6.1e-03         | 0.43            |
|       947 |    0.005  | 6.0e-04         | 4.3e-03         | 7.0e-03         | 0.61            |
|       947 |    0.0053 | 5.2e-04         | 4.2e-03         | 6.5e-03         | 0.55            |
|      1068 |    0.0069 | 7.2e-04         | 5.8e-03         | 8.2e-03         | 0.40            |
|      1068 |    0.0063 | 5.5e-04         | 5.2e-03         | 7.3e-03         | 0.41            |
|      1068 |    0.0061 | 4.2e-04         | 5.3e-03         | 7.0e-03         | 0.33            |
|      1068 |    0.006  | 5.3e-04         | 5.1e-03         | 7.2e-03         | 0.41            |
|      1068 |    0.0056 | 4.5e-04         | 4.9e-03         | 6.8e-03         | 0.38            |
|      1068 |    0.0055 | 4.1e-04         | 4.6e-03         | 6.6e-03         | 0.43            |
|      1098 |    0.0037 | 3.9e-04         | 3.1e-03         | 4.6e-03         | 0.49            |
|      1098 |    0.0037 | 2.5e-04         | 3.1e-03         | 4.0e-03         | 0.30            |
|      1098 |    0.0045 | 6.6e-04         | 3.1e-03         | 6.1e-03         | 0.95            |
|      1098 |    0.0045 | 4.8e-04         | 3.6e-03         | 5.5e-03         | 0.56            |
|      1098 |    0.0054 | 1.0e-03         | 3.8e-03         | 7.6e-03         | 0.99            |
|      1098 |    0.0076 | 1.3e-03         | 5.1e-03         | 9.9e-03         | 0.94            |
|      1234 |    0.0064 | 5.9e-04         | 5.4e-03         | 7.4e-03         | 0.36            |
|      1234 |    0.0062 | 5.4e-04         | 5.2e-03         | 7.2e-03         | 0.38            |
|      1234 |    0.006  | 3.7e-04         | 5.2e-03         | 6.6e-03         | 0.27            |
|      1234 |    0.0061 | 4.6e-04         | 5.4e-03         | 6.9e-03         | 0.27            |
|      1234 |    0.0058 | 5.5e-04         | 5.1e-03         | 7.6e-03         | 0.50            |
|      1719 |    0.0049 | 5.0e-04         | 4.0e-03         | 5.7e-03         | 0.42            |
|      1719 |    0.0046 | 3.8e-04         | 3.8e-03         | 5.1e-03         | 0.35            |
|      1719 |    0.0047 | 3.1e-04         |                 |                 |                 |
|      1719 |    0.0046 |                 | 3.9e-03 4.1e-03 | 5.2e-03         | 0.34            |
|      1719 |    0.0045 | 3.8e-04 4.0e-04 | 3.9e-03         | 5.4e-03 5.5e-03 | 0.32 0.40       |
|      1719 |    0.0047 | 4.2e-04         | 3.9e-03         | 5.7e-03         | 0.46            |

|   E [keV] |   〈 F 〉 |     σ F |   F min | F max           | F max - F min   |
|-----------|-----------|---------|---------|-----------------|-----------------|
|      1753 |    0.0032 | 0.00041 |  0.0026 | 4.0e-03         | F min 0.53      |
|      1753 |    0.003  | 0.00027 |  0.0024 | 3.4e-03         | 0.41            |
|      1753 |    0.0031 | 0.00027 |  0.0024 | 3.7e-03         | 0.59            |
|      1753 |    0.0031 | 0.0003  |  0.0026 | 3.7e-03         | 0.43            |
|      1753 |    0.0031 | 0.00035 |  0.0026 | 3.9e-03         | 0.51            |
|      1753 |    0.0036 | 0.00045 |  0.0027 | 4.3e-03         | 0.61            |
|      1809 |    0.0055 | 0.00051 |  0.0046 | 6.6e-03         | 0.43            |
|      1809 |    0.0052 | 0.00033 |  0.0046 | 5.8e-03         | 0.28            |
|      1809 |    0.0057 | 0.00033 |  0.005  | 6.3e-03         | 0.27            |
|      1809 |    0.0057 | 0.00035 |  0.0052 | 6.5e-03         | 0.24            |
|      1809 |    0.0059 | 0.00047 |  0.005  | 7.3e-03         | 0.46            |
|      1809 |    0.0061 | 0.00039 |  0.0054 | 7.0e-03         | 0.30            |
|      1904 |    0.0048 | 0.00051 |  0.0041 | 5.6e-03         | 0.39            |
|      1904 |    0.0045 | 0.00039 |  0.0038 | 5.3e-03         | 0.40            |
|      1904 |    0.0044 | 0.00032 |  0.0037 | 5.2e-03         | 0.39            |
|      1904 |    0.0043 | 0.0004  |  0.0036 | 5.2e-03         | 0.44            |
|      1904 |    0.0041 | 0.00037 |  0.0036 | 5.2e-03         | 0.42            |
|      1904 |    0.0041 | 0.00033 |  0.0034 | 4.9e-03         | 0.43            |
|      2322 |    0.0042 | 0.0004  |  0.0035 | 4.9e-03         | 0.39            |
|      2322 |    0.0039 | 0.0003  |  0.0032 | 4.3e-03         | 0.34            |
|      2322 |    0.0039 | 0.00029 |  0.0033 | 4.7e-03         | 0.42            |
|      2322 |    0.0038 | 0.00032 |  0.0033 | 4.7e-03         | 0.43            |
|      2322 |    0.0036 | 0.00029 |  0.0031 | 4.2e-03         | 0.35            |
|      2583 |    0.0032 | 0.00041 |  0.0026 | 4.0e-03         | 0.53            |
|      2583 |    0.0029 | 0.00026 |  0.0024 | 3.3e-03         | 0.41            |
|      2583 |    0.0029 | 0.00022 |  0.0024 | 3.4e-03         | 0.42            |
|      2583 |    0.0028 | 0.00026 |  0.0024 | 3.4e-03         | 0.44            |
|      2583 |    0.0027 | 0.00026 |  0.0021 | 3.3e-03         | 0.52            |
|      2583 |    0.0028 | 0.00026 |  0.0023 | 3.4e-03         | 0.48            |
|      3176 |    0.0025 | 0.00025 |  0.0021 | 3.0e-03         | 0.41            |
|      3176 |    0.0023 | 0.00019 |  0.0019 | 2.7e-03         | 0.40            |
|      3176 |    0.0024 | 0.00016 |  0.0019 | 2.7e-03         | 0.41            |
|      3176 |    0.0024 | 0.00019 |  0.002  | 2.7e-03         | 0.36            |
|      3176 |    0.0023 | 0.00021 |  0.0019 | 2.7e-03         | 0.42            |
|      3176 |    0.0023 | 0.00022 |  0.0019 | 2.9e-03         | 0.54            |
|      3331 |    0.0025 | 0.00029 |  0.0022 | 3.2e-03         | 0.50            |
|      3331 |    0.0023 | 0.0002  |  0.0017 | 2.7e-03         | 0.55            |
|      3331 |    0.0024 | 0.00015 |  0.002  | 2.8e-03         | 0.35            |
|      3331 |    0.0023 | 0.00017 |  0.002  | 2.7e-03         | 0.33            |
|      3331 |    0.0023 | 0.00021 |  0.0021 | 3.1e-03         | 0.51            |
|      3331 |    0.0024 | 0.00023 |  0.002  | 2.9e-03         | 0.44            |
|      4738 |    0.0017 | 6.2e-05 |  0.0015 | 1.8e-03         | 0.22            |
|      4738 |    0.0017 | 6.5e-05 |  0.0015 | 1.9e-03         | 0.24            |
|      4738 |    0.0018 | 9.9e-05 |  0.0015 | 2.0e-03         | 0.31            |
|      4738 |    0.0018 | 0.00013 |  0.0015 | 2.0e-03         | 0.38            |
|      4738 |    0.0018 | 0.00021 |  0.0015 | 2.4e-03         | 0.64            |
|      4738 |    0.0018 | 0.00017 |  0.0014 | 2.1e-03         | 0.46            |
|      5186 |    0.0015 | 6.7e-05 |  0.0014 | 1.7e-03         | 0.26            |
|      5186 |    0.0016 | 5.5e-05 |  0.0015 | 1.8e-03         | 0.16            |
|      5186 |    0.0018 | 9.4e-05 |  0.0016 | 2.1e-03         | 0.35            |
|      5186 |    0.0018 | 6.4e-05 |  0.0016 | 2.0e-03         | 0.24            |
|      5186 |    0.0018 | 6.9e-05 |  0.0016 | 1.9e-03         | 0.18            |
|      5208 |    0.0015 | 8.8e-05 |  0.0012 | 1.8e-03         | 0.51            |
|      5208 |    0.0016 | 5.6e-05 |  0.0015 | 1.8e-03         | 0.21            |
|      5208 |    0.0017 | 5.8e-05 |  0.0016 | 1.9e-03         |                 |
|      5208 |    0.0017 | 5.3e-05 |  0.0016 |                 | 0.16 0.15       |
|      5208 |    0.0018 | 8e-05   |  0.0016 | 1.9e-03 2.1e-03 | 0.28            |
|      5208 |    0.0018 | 0.00013 |  0.0015 | 2.1e-03         | 0.39            |