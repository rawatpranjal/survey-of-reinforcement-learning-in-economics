# refs.bib sweep — 2026-05-22

Scope: master arxiv survey only. Live compilation = files \input'd by docs/main.tex (20 files, transitive depth 1). Excluded: thesis/, thesis_v2/, journals/, archive/, ORE_main/, backups/, .worktrees/, .claude/worktrees/, ch12 v1/v2/v3_archived/.

## Counts

| Set | Count |
|-----|-------|
| A — keys defined in refs.bib | 491 |
| B — keys cited in live main.tex inputs | 435 |
| A\B — orphans (in bib, not cited in live survey) | 54 |
| B\A — missing (cited but absent from bib) | **0** |

No undefined citation references in the compiled survey. Zero missing keys.

## Orphans (54)

All are in bib but not cited by any file currently \input'd in main.tex. High orphan count is expected (bib accumulates entries; some only appear in journal/thesis variants). Do NOT remove.

Notable subset (ch12 world-models pre-compile additions, likely queued for inclusion):
`HafnerDreamer2020`, `HafnerDreamerV2_2021`, `HafnerDreamerV3_2025`, `HafnerPlaNet2019`, `HaSchmidhuber2018`, `Schrittwieser2020MuZero`, `Schmidhuber1990`, `Janner2022diffuser`, `Ajay2023`, `Chua2018PETS`, `GrimmPVE2021`, `GrimmVALEQ2020`, `Hansen2024TDMPC2`, `Lambert2020`, `Sekar2020`, `Talvitie2017`, `Kurutach2018`, `Pathak2017`, `Asadi2018Lipschitz`, `AsadiWasserstein2018`, `FarahmandVAML2017`, `FarahmandIterVAML2018`, `VoelckerCalib2025`, `janner2019model`.

Macro/game-theory orphans (likely queued for ch06_macro/ch06_games):
`Azinovic2022den`, `Carmona2021mfg`, `Maliar2021dl`, `Kase2025gem`, `Kase2025hank`, `Fang2026dpi`, `Thoeni2026nmfg`, `HofbauerSandholm2002`, `EspondaPouzo2016`, `EvansHonkapohja2001`, `FudenbergKreps1993`, `FudenbergLevine1998`, `BrockHommes1997`, `MarimonMcGrattanSargent1990`, `Arifovic1994`, `Arifovic1995`, `Young2004`, `Antonoglou2022StochMuZero`.

Other orphans: `Badanidiyuru2013`, `Eimer2023`, `HallHoffarth2023constraints`, `Towers2024`, `Mueller2019`, `Patterson2024`, `Madeka2022DeepInventory`, `idaIshiharaItoEtAl2024energyrebate`, `farahmand2010`, `luckett2020vlearning`, `AyoubVTR2020`, `DontiAmosKolter2017`.

## Draft-Only Files with Missing Keys (not flagged as B\A — these files are not compiled)

Two tex files exist in the repo but are NOT \input'd in main.tex and contain citations to keys absent from bib. These will become errors if the files are ever included.

**ch01_history/tex/structural_econometrics_history.tex** (15 missing keys):
`AckerbergCavesFrazer2015`, `AguirregabiriaMira2007`, `AtashbarShi2023`, `BajariBenkardLevin2007`, `BerryLevinsohnPakes1995`, `CilibertoTamer2009`, `DubeFoxSu2012`, `Haavelmo1944`, `Heckman1979`, `HoodKoopmans1953`, `Miller1984`, `PesendorferSchmidtDengler2008`, `Petrin2002`, `WeintraubBenkardVanRoy2008`, `Wolpin1984`.

**ch03_theory/tex/curse_of_dimensionality.tex** (16 missing keys):
`ayoub2020`, `bray2022comment`, `chow1989complexity`, `du2021`, `fan2020dqn`, `hotz1993`, `jin2020`, `jin2021`, `kearns1999`, `liu2022deep`, `lu2025`, `moerland2022unifying`, `papadimitriou1987`, `rust1997`, `traub1988`, `zanette2020`.

Action required before either file is \input'd in main.tex: add the 31 missing bib entries.

## Fabrication Flags

None. Spot-checked suspicious keys (`Fang2026dpi`, `Kase2025gem`, `Kase2025hank`, `Thoeni2026nmfg`, `VoelckerCalib2025`). All have plausible author names, real institutions (BIS working papers, ICML proceedings), and verifiable arXiv IDs. Minor: `Thoeni2026nmfg` key says 2026 but bib entry says `year={2025}` — key-year mismatch, not fabrication.
