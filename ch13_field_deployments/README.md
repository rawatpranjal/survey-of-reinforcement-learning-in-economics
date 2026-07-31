# Reinforcement Learning in the Field

The chapter source is `tex/field_deployments.tex`. Its OPE reliability experiment uses pinned
versions of SCOPE-RL, d3rlpy, Gym, and PyTorch. Install these packages in a separate environment
so they do not replace packages in the main repository environment.

```bash
cd ch13_field_deployments/sims
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python field_ope_reliability.py > field_ope_reliability_stdout.txt 2>&1
pytest -q
```

`field_ope_reliability.py` runs the experiment. `promo_env.py` defines the environment,
`pipeline.py` implements training and OPE, and `ope_diagnostics.py` contains the reliability
checks. The command regenerates the OPE figure, results table, candidate table, and LaTeX macros
used by the chapter.

The production-loop diagram has editable TikZ source in `sims/horizon_pipeline.tex`.
