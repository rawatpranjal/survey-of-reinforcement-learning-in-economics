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
    "ch03a_bm/sims/bm_fvi_fqi.png"
    "ch03b_deeprl_practice/sims/overestimation_bias.png"
    "ch03b_deeprl_practice/sims/brock_mirman_bellman.png"
    "ch04_control_problems/sims/bus_engine_scaling.png"
    "ch05_econ_models/sims/estimation_flowcharts.png"
    "ch07_bandits/sims/curve_learning_pricing_pct_oracle.png"
    "ch05_econ_models/sims/nfxp_ccp_td_scaling_time.png"
    "ch08_offline_rl/sims/offline_rl_pricing_coverage.png"
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
    "ch11_dist_robust_constrained/sims/carbon_constrained_production_convergence.png"
    "ch11_dist_robust_constrained/sims/risk_sensitive_inventory_policy.png"
    "ch11_dist_robust_constrained/sims/robust_consumption_savings_policy.png"
    "ch12_world_models/sims/dyna_maze.png"
    "ch12_world_models/sims/dyna_maze_layout.png"
    "ch12_world_models/sims/cobweb_paradigms.png"
    "ch12_world_models/sims/cobweb_paradigms_param_recovery.png"
    "ch12_world_models/sims/cobweb_paradigms_policy_distance.png"
    "ch12_world_models/sims/fishery_paradigms.png"
    "ch12_world_models/sims/multi_echelon_paradigms.png"
    "ch06_games/sims/durable_goods_coase_collapse.png"
    "ch06_games/sims/durable_goods_coase_price_paths.png"
    "ch09_rlhf/sims/axiom_aware_aggregation.png"
    "ch13_field_deployments/sims/horizon_pipeline.png"
    "ch13_field_deployments/sims/field_ope_reliability_mechanism.png"
    "appA_preliminaries/sims/banach_contraction.png"
    "appA_preliminaries/sims/envelope_theorem.png"
    "appA_preliminaries/sims/gradient_descent.png"
    "appA_preliminaries/sims/hilbert_projection.png"
    "appA_preliminaries/sims/jensen_gap.png"
    "appA_preliminaries/sims/lagrangian_duality.png"
    "appA_preliminaries/sims/lipschitz_continuity.png"
    "appA_preliminaries/sims/lln_clt.png"
    "appA_preliminaries/sims/markov_stationary.png"
    "appA_preliminaries/sims/martingale_convergence.png"
    "appA_preliminaries/sims/neumann_series.png"
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
    "ch03_theory/sims/brock_mirman_results.tex"
    "ch03_theory/sims/td_lambda_corridor.tex"
    "ch03_theory/sims/lqc_fvi_fqi_weights.tex"
    "ch03_theory/sims/wind_farm_curse_study_results.tex"
    "ch03a_bm/sims/bm_fvi_fqi_results.tex"
    "ch04_control_problems/sims/bus_engine_results.tex"
    "ch05_econ_models/sims/nfxp_ccp_td_results.tex"
    "ch06_games/sims/cournot_bertrand_results.tex"
    "ch06_games/sims/kuhn_poker_results.tex"
    "ch06_games/sims/durable_goods_results.tex"
    "ch07_bandits/sims/knowledge_ladder_results.tex"
    "ch07_bandits/sims/curve_learning_pricing_summary.tex"
    "ch08_offline_rl/sims/offline_rl_pricing_results.tex"
    "ch09_rlhf/sims/job_search_results.tex"
    "ch09_rlhf/sims/job_search_diagnostics.tex"
    "ch09_rlhf/sims/job_search_horizon.tex"
    "ch10_causal/sims/confounded_ope_results.tex"
    "ch10_causal/sims/counterfactual_ope_table.tex"
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
    "ch11_dist_robust_constrained/sims/carbon_constrained_production_table.tex"
    "ch11_dist_robust_constrained/sims/risk_sensitive_inventory_table.tex"
    "ch11_dist_robust_constrained/sims/robust_consumption_savings_table.tex"
    "ch12_world_models/sims/cobweb_paradigms_final_recovery.tex"
    "ch12_world_models/sims/cobweb_paradigms_results.tex"
    "ch12_world_models/sims/dyna_maze_results.tex"
    "ch12_world_models/sims/fishery_paradigms_results.tex"
    "ch12_world_models/sims/fishery_paradigms_recovery.tex"
    "ch12_world_models/sims/multi_echelon_paradigms_results.tex"
    "ch06_games/sims/durable_goods_coase_results.tex"
    "ch09_rlhf/sims/axiom_aware_aggregation.tex"
    "ch13_field_deployments/sims/field_ope_reliability_macros.tex"
    "ch13_field_deployments/sims/field_ope_reliability_table.tex"
    "ch13_field_deployments/sims/field_ope_reliability_candidates.tex"
    "ch10c_adaptive_experiments/sims/causal_bandit_mabuc_results.tex"
    "appA_preliminaries/sims/banach_contraction.tex"
    "appA_preliminaries/sims/envelope_theorem.tex"
    "appA_preliminaries/sims/gradient_descent.tex"
    "appA_preliminaries/sims/hilbert_projection.tex"
    "appA_preliminaries/sims/jensen_gap.tex"
    "appA_preliminaries/sims/lagrangian_duality.tex"
    "appA_preliminaries/sims/lipschitz_continuity.tex"
    "appA_preliminaries/sims/lln_clt.tex"
    "appA_preliminaries/sims/markov_stationary.tex"
    "appA_preliminaries/sims/martingale_convergence.tex"
    "appA_preliminaries/sims/neumann_series.tex"
    "appA_preliminaries/sims/robbins_monro.tex"
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
rm -f main.aux main.log main.out main.blg main.pdf main.luabridge.lua main.toc

# --- Create 00README.XXX to prevent arXiv from deleting input'ed files ---
cat > 00README.XXX <<'READMEEOF'
noop main.bbl
noop ch04_control_problems/sims/bus_engine_results.tex
noop ch05_econ_models/sims/nfxp_ccp_td_results.tex
noop ch06_games/sims/cournot_bertrand_results.tex
noop ch06_games/sims/kuhn_poker_results.tex
noop ch06_games/sims/durable_goods_results.tex
noop ch07_bandits/sims/knowledge_ladder_results.tex
noop ch08_offline_rl/sims/offline_rl_pricing_results.tex
noop ch09_rlhf/sims/job_search_results.tex
noop ch09_rlhf/sims/job_search_diagnostics.tex
noop ch09_rlhf/sims/job_search_horizon.tex
noop ch10_causal/sims/confounded_ope_results.tex
noop ch10_causal/sims/counterfactual_ope_table.tex
noop ch03_theory/sims/brock_mirman_results.tex
noop ch03_theory/sims/td_lambda_corridor.tex
noop ch03_theory/sims/lqc_fvi_fqi_weights.tex
noop ch03_theory/sims/wind_farm_curse_study_results.tex
noop ch03a_bm/sims/bm_fvi_fqi_results.tex
noop ch06_macro/sims/lq_mfg_results.tex
noop ch06_macro/sims/rbc_dp_vs_drl_results.tex
noop ch10c_adaptive_experiments/sims/causal_bandit_results.tex
noop ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy_results.tex
noop ch10b_rl_for_ci/sims/ope_estimators_results.tex
noop ch10b_rl_for_ci/sims/ope_estimators_dr_ablation.tex
noop ch10b_rl_for_ci/sims/ope_estimators_inference.tex
noop ch10b_rl_for_ci/sims/dynamic_dml_snmm_results.tex
noop ch10b_rl_for_ci/sims/dynamic_dml_snmm_joint_inference.tex
noop ch10b_rl_for_ci/sims/dtr_policy_learning_results.tex
noop ch11_dist_robust_constrained/sims/carbon_constrained_production_table.tex
noop ch11_dist_robust_constrained/sims/risk_sensitive_inventory_table.tex
noop ch11_dist_robust_constrained/sims/robust_consumption_savings_table.tex
noop ch12_world_models/sims/cobweb_paradigms_final_recovery.tex
noop ch12_world_models/sims/cobweb_paradigms_results.tex
noop ch12_world_models/sims/dyna_maze_results.tex
noop ch12_world_models/sims/fishery_paradigms_results.tex
noop ch12_world_models/sims/multi_echelon_paradigms_results.tex
noop ch07_bandits/sims/curve_learning_pricing_summary.tex
noop ch06_games/sims/durable_goods_coase_results.tex
noop ch09_rlhf/sims/axiom_aware_aggregation.tex
noop ch13_field_deployments/sims/field_ope_reliability_macros.tex
noop ch13_field_deployments/sims/field_ope_reliability_table.tex
noop ch13_field_deployments/sims/field_ope_reliability_candidates.tex
noop ch10c_adaptive_experiments/sims/causal_bandit_mabuc_results.tex
noop ch12_world_models/sims/fishery_paradigms_recovery.tex
noop appA_preliminaries/sims/banach_contraction.tex
noop appA_preliminaries/sims/envelope_theorem.tex
noop appA_preliminaries/sims/gradient_descent.tex
noop appA_preliminaries/sims/hilbert_projection.tex
noop appA_preliminaries/sims/jensen_gap.tex
noop appA_preliminaries/sims/lagrangian_duality.tex
noop appA_preliminaries/sims/lipschitz_continuity.tex
noop appA_preliminaries/sims/lln_clt.tex
noop appA_preliminaries/sims/markov_stationary.tex
noop appA_preliminaries/sims/martingale_convergence.tex
noop appA_preliminaries/sims/neumann_series.tex
noop appA_preliminaries/sims/robbins_monro.tex
noop appA_preliminaries/sims/spectral_radius.tex
READMEEOF
echo "  Created 00README.XXX (prevents arXiv file deletion)"

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
