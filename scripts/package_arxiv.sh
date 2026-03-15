#!/bin/bash
# Package the RL survey for arXiv submission
# Creates a tarball with all necessary files, main.tex at the root

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/arxiv_submission"
TARBALL="$REPO_ROOT/arxiv_submission.tar.gz"

echo "=== Packaging arXiv submission ==="
echo "Repo root: $REPO_ROOT"

# Clean previous build
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# --- 1. Copy and rewrite main.tex ---
# Change ../chXX/ to chXX/ since main.tex will be at the root
sed 's|\.\./ch|ch|g' "$REPO_ROOT/docs/main.tex" > "$BUILD_DIR/main.tex"
echo "  Copied main.tex (rewrote paths)"

# --- 2. Copy bibliography (.bbl for arXiv, .bib as backup) ---
cp "$REPO_ROOT/docs/refs.bib" "$BUILD_DIR/"
cp "$REPO_ROOT/docs/main.bbl" "$BUILD_DIR/"
echo "  Copied refs.bib and main.bbl"

# --- 3. Copy chapter tex files (rewrite ../chXX/ paths) ---
CHAPTERS=(
    "ch00_introduction/tex/abstract.tex"
    "ch00_introduction/tex/intro.tex"
    "ch00_introduction/tex/language.tex"
    "ch01_history/tex/history.tex"
    "ch02_rl_algorithms/tex/rl_algorithms.tex"
    "ch03_theory/tex/planning_learning_v3.tex"
    "ch03b_deeprl_practice/tex/deeprl_practice.tex"
    "ch03a/tex/illustrated_example.tex"
    "ch04_control_problems/tex/applications.tex"
    "ch05_econ_models/tex/rl_in_se.tex"
    "ch06_games/tex/rl_in_games.tex"
    "ch07_bandits/tex/dynamic_pricing.tex"
    "ch08_rlhf/tex/rlhf.tex"
    "ch09_causal/tex/causal_rl.tex"
    "ch10_conclusion/tex/conclusion.tex"
)

for f in "${CHAPTERS[@]}"; do
    mkdir -p "$BUILD_DIR/$(dirname "$f")"
    sed 's|\.\./ch|ch|g' "$REPO_ROOT/$f" > "$BUILD_DIR/$f"
done
echo "  Copied ${#CHAPTERS[@]} chapter tex files (rewrote paths)"

# --- 4. Copy figures (PNG + PDF) ---
FIGURES=(
    "ch02_rl_algorithms/sims/algorithm_architectures.png"
    "ch03_theory/sims/brock_mirman_convergence.png"
    "ch03_theory/sims/lqc_fvi_fqi.png"
    "ch03_theory/sims/td_lambda_corridor.png"
    "ch03_theory/sims/deadly_triad_geometry.png"
    "ch03_theory/sims/info_geometry_npg.png"
    "ch03_theory/sims/mm_surrogate_trpo.png"
    "ch03_theory/sims/trust_region_lqc.png"
    "ch03a_bm/sims/bm_fvi_fqi.png"
    "ch03b_deeprl_practice/sims/overestimation_bias.png"
    "ch03b_deeprl_practice/sims/brock_mirman_bellman.png"
    "ch03a/sims/gridworld_value_heatmaps.png"
    "ch03a/sims/gridworld_policy_heatmaps.png"
    "ch04_control_problems/sims/bus_engine_scaling.png"
    "ch05_econ_models/sims/estimation_flowcharts.pdf"
    "ch06_games/sims/cournot_bertrand_marl.png"
    "ch07_bandits/sims/uninformative_price.png"
    "ch07_bandits/sims/regret_rates.png"
    "ch07_bandits/sims/knowledge_ladder_regret.png"
    "ch08_rlhf/sims/rlhf_dpo_pipeline.png"
    "ch08_rlhf/sims/job_search_env.png"
    "ch08_rlhf/sims/job_search_sample_complexity.png"
    "ch08_rlhf/sims/job_search_horizon.png"
    "ch09_causal/sims/identification_dags.png"
    "ch09_causal/sims/simulation_dag.png"
    "ch09_causal/sims/confounded_ope_bias.png"
)

for f in "${FIGURES[@]}"; do
    mkdir -p "$BUILD_DIR/$(dirname "$f")"
    cp "$REPO_ROOT/$f" "$BUILD_DIR/$f"
done
echo "  Copied ${#FIGURES[@]} figure files"

# --- 5. Copy table fragments (.tex in sims/) ---
TABLES=(
    "ch03_theory/sims/brock_mirman_results.tex"
    "ch03_theory/sims/td_lambda_corridor.tex"
    "ch03_theory/sims/lqc_fvi_fqi_weights.tex"
    "ch03a_bm/sims/bm_fvi_fqi_results.tex"
    "ch03a/sims/gridworld_study_results.tex"
    "ch03a/sims/gridworld_value_convergence.tex"
    "ch03a/sims/gridworld_policy_convergence.tex"
    "ch04_control_problems/sims/bus_engine_results.tex"
    "ch06_games/sims/cournot_bertrand_results.tex"
    "ch06_games/sims/durable_goods_results.tex"
    "ch07_bandits/sims/knowledge_ladder_results.tex"
    "ch08_rlhf/sims/job_search_results.tex"
    "ch08_rlhf/sims/job_search_diagnostics.tex"
    "ch08_rlhf/sims/job_search_horizon.tex"
    "ch09_causal/sims/confounded_ope_results.tex"
)

for f in "${TABLES[@]}"; do
    mkdir -p "$BUILD_DIR/$(dirname "$f")"
    cp "$REPO_ROOT/$f" "$BUILD_DIR/$f"
done
echo "  Copied ${#TABLES[@]} table fragment files"

# --- 6. Verify compilation ---
echo ""
echo "=== Verifying compilation in submission directory ==="
cd "$BUILD_DIR"
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
bibtex main > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

if [ -f main.pdf ]; then
    PAGES=$(pdfinfo main.pdf 2>/dev/null | grep Pages | awk '{print $2}')
    SIZE=$(du -h main.pdf | awk '{print $1}')
    echo "  Compilation successful: $PAGES pages, $SIZE"
else
    echo "  ERROR: Compilation failed! Check log:"
    cat main.log | grep "^!" | head -10
    exit 1
fi

# Remove build artifacts (keep .bbl for arXiv)
rm -f main.aux main.log main.out main.blg main.pdf main.luabridge.lua

# --- 7. Create tarball ---
cd "$REPO_ROOT"
tar czf "$TARBALL" -C "$BUILD_DIR" .

TARBALL_SIZE=$(du -h "$TARBALL" | awk '{print $1}')
FILE_COUNT=$(tar tzf "$TARBALL" | wc -l | tr -d ' ')
echo ""
echo "=== arXiv submission package created ==="
echo "  File: $TARBALL"
echo "  Size: $TARBALL_SIZE"
echo "  Files: $FILE_COUNT"
echo ""
echo "Upload $TARBALL to https://arxiv.org/submit"
