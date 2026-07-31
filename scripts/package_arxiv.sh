#!/bin/bash
# Package the RL survey for arXiv submission
# Creates a tarball with all necessary files, main.tex at the root

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FINAL_BUILD_DIR="$REPO_ROOT/arxiv_submission"
FINAL_TARBALL="$REPO_ROOT/arxiv_submission.tar.gz"
FINAL_VERIFY_FILE="$REPO_ROOT/arxiv_submission.verify.txt"
SOURCE_PDF="$REPO_ROOT/docs/main.pdf"

if [ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no)" ]; then
    echo "ERROR: tracked worktree changes remain; package only a clean committed HEAD"
    git -C "$REPO_ROOT" status --short --untracked-files=no
    exit 1
fi

if [ ! -f "$SOURCE_PDF" ]; then
    echo "ERROR: build docs/main.pdf before packaging"
    exit 1
fi

SOURCE_PAGES="$(pdfinfo "$SOURCE_PDF" | awk '/^Pages:/{print $2}')"
SOURCE_TEXT_HASH="$(pdftotext "$SOURCE_PDF" - | shasum -a 256 | awk '{print $1}')"
SOURCE_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"

echo "=== Checking bibliography ==="
python3 "$REPO_ROOT/scripts/check_bib.py" --main "$REPO_ROOT/docs/main.tex"
echo ""

echo "=== Packaging arXiv submission ==="
echo "Repo root: $REPO_ROOT"

# Build and verify the complete release under temporary sibling paths. Canonical
# outputs are promoted only after every compilation and extraction check passes.
TRANSACTION_DIR="$(mktemp -d "$REPO_ROOT/.arxiv-package.XXXXXX")"
BUILD_DIR="$TRANSACTION_DIR/arxiv_submission"
TARBALL="$TRANSACTION_DIR/arxiv_submission.tar.gz"
VERIFY_FILE="$TRANSACTION_DIR/arxiv_submission.verify.txt"
SMOKE_DIR="$TRANSACTION_DIR/smoke"
trap 'rm -rf "$TRANSACTION_DIR"' EXIT

mkdir -p "$BUILD_DIR"

# --- 1. Copy and rewrite main.tex ---
# Change ../chXX/ to chXX/ since main.tex will be at the root
sed 's|\.\./ch|ch|g; s|\.\./appA|appA|g' "$REPO_ROOT/docs/main.tex" > "$BUILD_DIR/main.tex"
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
    "ch03_theory/tex/curse_of_dimensionality.tex"
    "ch03b_deeprl_practice/tex/deeprl_practice.tex"
    "ch04_control_problems/tex/applications.tex"
    "ch05_econ_models/tex/rl_in_se.tex"
    "ch06_macro/tex/macro_rl.tex"
    "ch06_games/tex/rl_in_games.tex"
    "ch07_bandits/tex/dynamic_pricing.tex"
    "ch08_offline_rl/tex/offline_rl.tex"
    "ch09_rlhf/tex/rlhf.tex"
    "ch10_causal/tex/causal_rl.tex"
    "ch10b_rl_for_ci/tex/rl_for_ci.tex"
    "ch10c_adaptive_experiments/tex/adaptive_experiments.tex"
    "ch11_dist_robust_constrained/tex/dist_robust_constrained.tex"
    "ch12_world_models/tex/world_models.tex"
    "ch12_world_models/tex/s01_intro.tex"
    "ch12_world_models/tex/s03_dyna_q.tex"
    "ch12_world_models/tex/s04_deep_mbrl.tex"
    "ch12_world_models/tex/s06_objectives_convergence.tex"
    "ch12_world_models/tex/s09_dual_sim.tex"
    "ch12_world_models/tex/s09_engine_model_learning.tex"
    "ch12_world_models/tex/s11_high_dim_sims.tex"
    "ch12_world_models/tex/s10_synthesis.tex"
    "ch13_field_deployments/tex/field_deployments.tex"
    "ch99_conclusion/tex/conclusion.tex"
    "appA_preliminaries/tex/preliminaries.tex"
)

for f in "${CHAPTERS[@]}"; do
    mkdir -p "$BUILD_DIR/$(dirname "$f")"
    sed 's|\.\./ch|ch|g; s|\.\./appA|appA|g' "$REPO_ROOT/$f" > "$BUILD_DIR/$f"
done
echo "  Copied ${#CHAPTERS[@]} chapter tex files (rewrote paths)"

# Copy glossary
cp "$REPO_ROOT/docs/glossary.tex" "$BUILD_DIR/"
echo "  Copied glossary.tex"

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
    "ch03_theory/sims/wind_farm_curse_study_times.png"
    "ch03_theory/sims/curse_arithmetic.png"
    "ch03_theory/sims/engine_value_polytope.png"
    "ch03_theory/sims/engine_policy_square.png"
    "ch03_theory/sims/engine_value_learning.png"
    "ch03a_bm/sims/bm_fvi_fqi.png"
    "ch03b_deeprl_practice/sims/brock_mirman_bellman.png"
    "ch04_control_problems/sims/bus_engine_scaling.png"
    "ch05_econ_models/sims/estimation_flowcharts.png"
    "ch07_bandits/sims/curve_learning_pricing_pct_oracle.png"
    "ch05_econ_models/sims/nfxp_ccp_td_scaling_time.png"
    "ch08_offline_rl/sims/offline_rl_pricing_coverage.png"
    "ch08_offline_rl/sims/engine_coverage.png"
    "ch06_games/sims/cournot_bertrand_marl.png"
    "ch06_games/sims/kuhn_poker_exploitability.png"
    "ch07_bandits/sims/uninformative_price.png"
    "ch07_bandits/sims/regret_rates.png"
    "ch07_bandits/sims/knowledge_ladder_regret.png"
    "ch09_rlhf/sims/rlhf_dpo_pipeline.png"
    "ch09_rlhf/sims/job_search_env.png"
    "ch09_rlhf/sims/job_search_sample_complexity.png"
    "ch09_rlhf/sims/job_search_horizon.png"
    "ch10_causal/sims/identification_dags.png"
    "ch10_causal/sims/simulation_dag.png"
    "ch10_causal/sims/confounded_ope_bias.png"
    "ch10_causal/sims/counterfactual_ope.png"
    "ch06_macro/sims/lq_mfg.png"
    "ch06_macro/sims/rbc_dp_vs_drl_learning_curves.png"
    "ch10c_adaptive_experiments/sims/causal_bandit_combined.png"
    "ch10b_rl_for_ci/sims/dtr_dags.png"
    "ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.png"
    "ch10b_rl_for_ci/sims/ope_estimators.png"
    "ch10b_rl_for_ci/sims/dynamic_dml_snmm_coverage.png"
    "ch10b_rl_for_ci/sims/dtr_policy_learning.png"
    "ch10b_rl_for_ci/sims/engine_ope.png"
    "ch11_dist_robust_constrained/sims/carbon_constrained_production_convergence.png"
    "ch11_dist_robust_constrained/sims/risk_sensitive_inventory_policy.png"
    "ch11_dist_robust_constrained/sims/robust_consumption_savings_policy.png"
    "ch11_dist_robust_constrained/sims/engine_occupancy_kl.png"
    "ch12_world_models/sims/dyna_maze.png"
    "ch12_world_models/sims/dyna_maze_layout.png"
    "ch12_world_models/sims/cobweb_paradigms.png"
    "ch12_world_models/sims/cobweb_paradigms_param_recovery.png"
    "ch12_world_models/sims/cobweb_paradigms_policy_distance.png"
    "ch12_world_models/sims/fishery_paradigms.png"
    "ch12_world_models/sims/multi_echelon_paradigms.png"
    "ch12_world_models/sims/engine_model_learning.png"
    "ch06_games/sims/durable_goods_coase_collapse.png"
    "ch06_games/sims/durable_goods_coase_price_paths.png"
    "ch09_rlhf/sims/axiom_aware_aggregation.png"
    "ch13_field_deployments/sims/horizon_pipeline.png"
    "ch13_field_deployments/sims/field_ope_reliability_mechanism.png"
    "appA_preliminaries/sims/coverage_geometry.png"
    "appA_preliminaries/sims/curvature_geometry.png"
    "appA_preliminaries/sims/discount_cost.png"
    "appA_preliminaries/sims/elementary_concepts.png"
    "appA_preliminaries/sims/norm_balls.png"
    "appA_preliminaries/sims/jensen_gap.png"
    "appA_preliminaries/sims/projection_geometry.png"
    "appA_preliminaries/sims/robbins_monro.png"
    "appA_preliminaries/sims/spectral_radius.png"
)

for f in "${FIGURES[@]}"; do
    mkdir -p "$BUILD_DIR/$(dirname "$f")"
    cp "$REPO_ROOT/$f" "$BUILD_DIR/$f"
done
echo "  Copied ${#FIGURES[@]} figure files"

# --- 5. Copy table fragments (.tex in sims/) ---
TABLES=(
    "ch02_rl_algorithms/sims/engine_algorithms.tex"
    "ch02_rl_algorithms/sims/engine_dictionary.tex"
    "ch02_rl_algorithms/sims/engine_recap.tex"
    "ch03_theory/sims/brock_mirman_results.tex"
    "ch03_theory/sims/td_lambda_corridor.tex"
    "ch03_theory/sims/lqc_fvi_fqi_weights.tex"
    "ch03_theory/sims/wind_farm_curse_study_results.tex"
    "ch03_theory/sims/curse_grid_dp.tex"
    "ch03_theory/sims/curse_chow_tsitsiklis.tex"
    "ch03_theory/sims/curse_sample_complexity.tex"
    "ch03_theory/sims/curse_smoothness.tex"
    "ch03_theory/sims/curse_factored.tex"
    "ch03_theory/sims/engine_value_polytope.tex"
    "ch03_theory/sims/engine_policy_square.tex"
    "ch03_theory/sims/engine_value_learning.tex"
    "ch03_theory/sims/engine_hybrid_search.tex"
    "ch03b_deeprl_practice/sims/engine_bellman_error_results.tex"
    "ch03a_bm/sims/bm_fvi_fqi_results.tex"
    "ch04_control_problems/sims/bus_engine_results.tex"
    "ch05_econ_models/sims/nfxp_ccp_td_results.tex"
    "ch05_econ_models/sims/engine_estimation_results.tex"
    "ch06_games/sims/cournot_bertrand_results.tex"
    "ch06_games/sims/kuhn_poker_results.tex"
    "ch06_games/sims/durable_goods_results.tex"
    "ch07_bandits/sims/knowledge_ladder_results.tex"
    "ch07_bandits/sims/curve_learning_pricing_summary.tex"
    "ch08_offline_rl/sims/offline_rl_pricing_results.tex"
    "ch08_offline_rl/sims/engine_coverage.tex"
    "ch09_rlhf/sims/job_search_results.tex"
    "ch09_rlhf/sims/job_search_diagnostics.tex"
    "ch09_rlhf/sims/job_search_horizon.tex"
    "ch09_rlhf/sims/engine_preferences.tex"
    "ch10_causal/sims/confounded_ope_results.tex"
    "ch10_causal/sims/counterfactual_ope_table.tex"
    "ch10_causal/sims/engine_confounding.tex"
    "ch06_macro/sims/lq_mfg_results.tex"
    "ch06_macro/sims/rbc_dp_vs_drl_results.tex"
    "ch10c_adaptive_experiments/sims/causal_bandit_results.tex"
    "ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy_results.tex"
    "ch10b_rl_for_ci/sims/ope_estimators_results.tex"
    "ch10b_rl_for_ci/sims/ope_estimators_dr_ablation.tex"
    "ch10b_rl_for_ci/sims/ope_estimators_inference.tex"
    "ch10b_rl_for_ci/sims/dynamic_dml_snmm_results.tex"
    "ch10b_rl_for_ci/sims/dynamic_dml_snmm_joint_inference.tex"
    "ch10b_rl_for_ci/sims/dtr_policy_learning_results.tex"
    "ch10b_rl_for_ci/sims/engine_ope.tex"
    "ch11_dist_robust_constrained/sims/carbon_constrained_production_table.tex"
    "ch11_dist_robust_constrained/sims/risk_sensitive_inventory_table.tex"
    "ch11_dist_robust_constrained/sims/robust_consumption_savings_table.tex"
    "ch11_dist_robust_constrained/sims/engine_occupancy_kl_table.tex"
    "ch12_world_models/sims/cobweb_paradigms_final_recovery.tex"
    "ch12_world_models/sims/cobweb_paradigms_results.tex"
    "ch12_world_models/sims/dyna_maze_results.tex"
    "ch12_world_models/sims/fishery_paradigms_results.tex"
    "ch12_world_models/sims/fishery_paradigms_recovery.tex"
    "ch12_world_models/sims/multi_echelon_paradigms_results.tex"
    "ch12_world_models/sims/engine_model_learning.tex"
    "ch06_games/sims/durable_goods_coase_results.tex"
    "ch09_rlhf/sims/axiom_aware_aggregation.tex"
    "ch13_field_deployments/sims/field_ope_reliability_macros.tex"
    "ch13_field_deployments/sims/field_ope_reliability_table.tex"
    "ch13_field_deployments/sims/field_ope_reliability_candidates.tex"
    "ch10c_adaptive_experiments/sims/causal_bandit_mabuc_results.tex"
    "appA_preliminaries/sims/discount_cost.tex"
    "appA_preliminaries/sims/jensen_gap.tex"
    "appA_preliminaries/sims/projection_geometry.tex"
    "appA_preliminaries/sims/robbins_monro.tex"
    "appA_preliminaries/sims/running_example.tex"
    "appA_preliminaries/sims/running_example_td.tex"
    "appA_preliminaries/sims/spectral_radius.tex"
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
pdflatex -interaction=nonstopmode -halt-on-error main.tex > main_pass1.log 2>&1
bibtex main > main_bibtex.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main.tex > main_pass2.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main.tex > main_pass3.log 2>&1

if [ -f main.pdf ]; then
    PAGES=$(pdfinfo main.pdf 2>/dev/null | awk '/^Pages:/{print $2}')
    SIZE=$(du -h main.pdf | awk '{print $1}')
    echo "  Compilation successful: $PAGES pages, $SIZE"
else
    echo "  ERROR: Compilation failed! Check log:"
    cat main.log | grep "^!" | head -10
    exit 1
fi

if rg -i "undefined references|undefined citations|citation .* undefined|reference .* undefined" main.log; then
    echo "ERROR: staged source has unresolved references or citations"
    exit 1
fi

STAGED_TEXT_HASH="$(pdftotext main.pdf - | shasum -a 256 | awk '{print $1}')"
if [ "$PAGES" != "$SOURCE_PAGES" ] || [ "$STAGED_TEXT_HASH" != "$SOURCE_TEXT_HASH" ]; then
    echo "ERROR: staged build differs from docs/main.pdf"
    echo "  source pages/hash: $SOURCE_PAGES $SOURCE_TEXT_HASH"
    echo "  staged pages/hash: $PAGES $STAGED_TEXT_HASH"
    exit 1
fi

# Remove build artifacts (keep .bbl for arXiv)
rm -f main.aux main.log main.out main.blg main.pdf main.luabridge.lua main.toc \
    main_pass1.log main_bibtex.log main_pass2.log main_pass3.log

# --- Create 00README.XXX to prevent arXiv from deleting input'ed files ---
# Generated from the TABLES manifest rather than hardcoded, so the retention list
# cannot drift out of step with what is actually shipped. arXiv prunes files it
# believes are unreferenced; a `noop` line tells it to keep one untouched.
{
    echo "noop main.bbl"
    for t in "${TABLES[@]}"; do
        echo "noop $t"
    done
} > 00README.XXX
echo "  Created 00README.XXX (prevents arXiv file deletion)"

# --- 7. Create tarball ---
cd "$REPO_ROOT"
COPYFILE_DISABLE=1 tar czf "$TARBALL" -C "$BUILD_DIR" .

TARBALL_SIZE=$(du -h "$TARBALL" | awk '{print $1}')
FILE_COUNT=$(tar tzf "$TARBALL" | wc -l | tr -d ' ')
TARBALL_SHA256=$(shasum -a 256 "$TARBALL" | awk '{print $1}')

BAD_ENTRIES="$(
    tar tzf "$TARBALL" |
        awk '/^\// || /(^|\/)\.\.($|\/)/ || /(^|\/)\._/ || /(^|\/)(cache|papers)(\/|$)/'
)"
if [ -n "$BAD_ENTRIES" ]; then
    echo "ERROR: unsafe or excluded archive entries found"
    echo "$BAD_ENTRIES"
    exit 1
fi

if tar tvzf "$TARBALL" | awk '$1 ~ /^[lh]/{found=1} END{exit !found}'; then
    echo "ERROR: archive contains a symbolic or hard link"
    exit 1
fi

mkdir -p "$SMOKE_DIR"
tar xzf "$TARBALL" -C "$SMOKE_DIR"
cd "$SMOKE_DIR"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > main_pass1.log 2>&1
bibtex main > main_bibtex.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main.tex > main_pass2.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main.tex > main_pass3.log 2>&1
if rg -i "undefined references|undefined citations|citation .* undefined|reference .* undefined" main.log; then
    echo "ERROR: fresh archive extraction has unresolved references or citations"
    exit 1
fi
SMOKE_PAGES="$(pdfinfo main.pdf | awk '/^Pages:/{print $2}')"
SMOKE_TEXT_HASH="$(pdftotext main.pdf - | shasum -a 256 | awk '{print $1}')"
if [ "$SMOKE_PAGES" != "$SOURCE_PAGES" ] || [ "$SMOKE_TEXT_HASH" != "$SOURCE_TEXT_HASH" ]; then
    echo "ERROR: fresh archive build differs from docs/main.pdf"
    exit 1
fi

cat > "$VERIFY_FILE" <<EOF
commit $SOURCE_COMMIT
source_pages $SOURCE_PAGES
source_text_sha256 $SOURCE_TEXT_HASH
archive_sha256 $TARBALL_SHA256
archive_size $TARBALL_SIZE
archive_files $FILE_COUNT
fresh_extract_pages $SMOKE_PAGES
fresh_extract_text_sha256 $SMOKE_TEXT_HASH
EOF

# Promote the verified trio together. If any same-filesystem rename fails,
# restore the previous canonical outputs before returning an error.
PREVIOUS_BUILD_DIR="$TRANSACTION_DIR/previous-arxiv_submission"
PREVIOUS_TARBALL="$TRANSACTION_DIR/previous-arxiv_submission.tar.gz"
PREVIOUS_VERIFY_FILE="$TRANSACTION_DIR/previous-arxiv_submission.verify.txt"

promote_outputs() {
    local failed=0

    if [ -e "$FINAL_BUILD_DIR" ]; then
        mv "$FINAL_BUILD_DIR" "$PREVIOUS_BUILD_DIR"
    fi
    if [ -e "$FINAL_TARBALL" ]; then
        mv "$FINAL_TARBALL" "$PREVIOUS_TARBALL"
    fi
    if [ -e "$FINAL_VERIFY_FILE" ]; then
        mv "$FINAL_VERIFY_FILE" "$PREVIOUS_VERIFY_FILE"
    fi

    mv "$BUILD_DIR" "$FINAL_BUILD_DIR" || failed=1
    if [ "$failed" -eq 0 ]; then
        mv "$TARBALL" "$FINAL_TARBALL" || failed=1
    fi
    if [ "$failed" -eq 0 ]; then
        # The verification record is the commit marker and moves last.
        mv "$VERIFY_FILE" "$FINAL_VERIFY_FILE" || failed=1
    fi

    if [ "$failed" -ne 0 ]; then
        rm -rf "$FINAL_BUILD_DIR"
        rm -f "$FINAL_TARBALL" "$FINAL_VERIFY_FILE"
        if [ -e "$PREVIOUS_BUILD_DIR" ]; then
            mv "$PREVIOUS_BUILD_DIR" "$FINAL_BUILD_DIR"
        fi
        if [ -e "$PREVIOUS_TARBALL" ]; then
            mv "$PREVIOUS_TARBALL" "$FINAL_TARBALL"
        fi
        if [ -e "$PREVIOUS_VERIFY_FILE" ]; then
            mv "$PREVIOUS_VERIFY_FILE" "$FINAL_VERIFY_FILE"
        fi
        echo "ERROR: failed to promote verified arXiv package; previous outputs restored"
        return 1
    fi
}

promote_outputs

echo ""
echo "=== arXiv submission package created ==="
echo "  File: $FINAL_TARBALL"
echo "  Size: $TARBALL_SIZE"
echo "  Archive members: $FILE_COUNT"
echo "  SHA-256: $TARBALL_SHA256"
echo "  Evidence: $FINAL_VERIFY_FILE"
echo ""
echo "Upload $FINAL_TARBALL to https://arxiv.org/submit"
