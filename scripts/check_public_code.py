#!/usr/bin/env python3
"""Check that the repository includes the paper's programs and generated results."""

from pathlib import Path
import re
import runpy
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SCRIPT = ROOT / "scripts" / "package_arxiv.sh"
RUNNER = ROOT / "scripts" / "run_all_sims.py"

SPECIAL_SCRIPTS = [
    "ch13_field_deployments/sims/field_ope_reliability.py",
]

PRODUCER_OVERRIDES = {
    "ch03_theory/sims/wind_farm_curse_study_times.png": "ch03_theory/sims/wind_farm_curse_study.py",
    "ch03_theory/sims/wind_farm_curse_study_results.tex": "ch03_theory/sims/wind_farm_curse_study.py",
    "ch06_macro/sims/lq_mfg_results.tex": "ch06_macro/sims/lq_mfg.py",
    "ch11_dist_robust_constrained/sims/carbon_constrained_production_convergence.png": "ch11_dist_robust_constrained/sims/carbon_constrained_production.py",
    "ch11_dist_robust_constrained/sims/carbon_constrained_production_table.tex": "ch11_dist_robust_constrained/sims/carbon_constrained_production.py",
    "ch11_dist_robust_constrained/sims/engine_occupancy_kl_table.tex": "ch11_dist_robust_constrained/sims/engine_occupancy_kl.py",
    "ch13_field_deployments/sims/field_ope_reliability_mechanism.png": "ch13_field_deployments/sims/field_ope_reliability.py",
    "ch13_field_deployments/sims/field_ope_reliability_macros.tex": "ch13_field_deployments/sims/field_ope_reliability.py",
    "ch13_field_deployments/sims/field_ope_reliability_table.tex": "ch13_field_deployments/sims/field_ope_reliability.py",
    "ch13_field_deployments/sims/field_ope_reliability_candidates.tex": "ch13_field_deployments/sims/field_ope_reliability.py",
}

ROOT_REQUIREMENTS = {
    "numpy",
    "scipy",
    "matplotlib",
    "tqdm",
    "torch",
    "gymnasium",
    "pandas",
    "scikit-learn",
}

CH13_REQUIREMENTS = {
    "d3rlpy",
    "gym",
    "gymnasium",
    "matplotlib",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "scope-rl",
    "torch",
    "pytest",
}


def package_paths(array_name):
    source = PACKAGE_SCRIPT.read_text(encoding="utf-8")
    match = re.search(rf"{array_name}=\(\n(.*?)\n\)", source, re.DOTALL)
    if not match:
        raise ValueError(f"could not find {array_name} in {PACKAGE_SCRIPT}")
    return re.findall(r'"([^"]+)"', match.group(1))


def requirement_names(path):
    names = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        names.add(re.split(r"[<>=!~\[]", line, maxsplit=1)[0].lower())
    return names


def tracked_paths():
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return set(result.stdout.splitlines())


def find_producer(artifact):
    override = PRODUCER_OVERRIDES.get(artifact)
    if override:
        return override if (ROOT / override).is_file() else None

    path = ROOT / artifact
    same_stem_tex = path.with_suffix(".tex")
    if path.suffix != ".tex" and same_stem_tex.is_file():
        return str(same_stem_tex.relative_to(ROOT))

    for script in sorted(path.parent.glob("*.py")):
        source = script.read_text(encoding="utf-8", errors="ignore")
        if path.name in source or path.stem in source:
            return str(script.relative_to(ROOT))
    return None


def check_repository():
    errors = []
    tracked = tracked_paths()
    registry = runpy.run_path(str(RUNNER))["REGISTRY"]
    standard_scripts = [path for _, path, _ in registry]
    if len(standard_scripts) != len(set(standard_scripts)):
        errors.append("the standard runner contains duplicate script paths")

    for relative in standard_scripts + SPECIAL_SCRIPTS:
        if not (ROOT / relative).is_file():
            errors.append(f"missing simulation script: {relative}")
        elif relative not in tracked:
            errors.append(f"untracked simulation script: {relative}")

    figures = package_paths("FIGURES")
    tables = package_paths("TABLES")
    for relative in figures + tables:
        if not (ROOT / relative).is_file():
            errors.append(f"missing packaged result: {relative}")
            continue
        if relative not in tracked:
            errors.append(f"untracked packaged result: {relative}")
        producer = find_producer(relative)
        if producer is None:
            errors.append(f"no source found for: {relative}")
        elif producer not in tracked:
            errors.append(f"source is not tracked: {producer}")

    root_found = requirement_names(ROOT / "requirements.txt")
    for name in sorted(ROOT_REQUIREMENTS - root_found):
        errors.append(f"root requirements omit: {name}")

    ch13_path = ROOT / "ch13_field_deployments" / "sims" / "requirements.txt"
    ch13_found = requirement_names(ch13_path)
    for name in sorted(CH13_REQUIREMENTS - ch13_found):
        errors.append(f"Chapter 13 requirements omit: {name}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for public_link in ("REPRODUCIBILITY.md", "scripts/check_public_code.py"):
        if public_link not in readme:
            errors.append(f"README does not link to: {public_link}")

    python_files = sorted(path for path in tracked if path.endswith(".py"))
    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1

    print("PASS repository code check")
    print(f"  tracked Python files: {len(python_files)}")
    print(f"  standard simulation scripts: {len(standard_scripts)}")
    print(f"  separately installed simulation scripts: {len(SPECIAL_SCRIPTS)}")
    print(f"  packaged figures with source: {len(figures)}")
    print(f"  packaged tables with source: {len(tables)}")
    return 0


if __name__ == "__main__":
    sys.exit(check_repository())
