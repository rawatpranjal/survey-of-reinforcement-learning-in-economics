# docs/ Wiki Index

`docs/` is the project wiki: research notes and reference docs accumulate here and get
organized under this index. Roadmap and long-term memory live at the repo root
(`../roadmap.md`, `../memory.md`). LaTeX source and the assembled `main.pdf` also live in
`docs/` (they are the build, not wiki entries).

## Active operation (2026-07)

- [theory-rigor-rl.md](theory-rigor-rl.md) - Workstream 1 dossier: give `ch03_theory`
  Theorem + Proof rigor (Enoch Kang style). Sources to fetch, proof inventory, the open
  sup-norm problem.
- [sim-automation-audit.md](sim-automation-audit.md) - Workstream 2 audit and manifest:
  every number traced to generated output, the honesty gaps, the build-gate design.

## Theory notes

- [../ch03_theory/notes/theoretical_foundations_dp_rl.md](../ch03_theory/notes/theoretical_foundations_dp_rl.md)
- [../ch03_theory/notes/dp_rl_policy_gradient_findings.md](../ch03_theory/notes/dp_rl_policy_gradient_findings.md)
- [../ch03_theory/notes/deadly_triad_breakdown.md](../ch03_theory/notes/deadly_triad_breakdown.md)
- [../ch03_theory/notes/rl_economics_operations_research_survey.md](../ch03_theory/notes/rl_economics_operations_research_survey.md)

## Chapter research notes

- [../ch01_history/notes/history_notes.md](../ch01_history/notes/history_notes.md)
- [../ch10_causal/notes/concrete_examples.md](../ch10_causal/notes/concrete_examples.md)
- [../ch10b_rl_for_ci/notes/claim_source_ledger_2026-07-24.md](../ch10b_rl_for_ci/notes/claim_source_ledger_2026-07-24.md)
- [../ch10b_rl_for_ci/notes/translation_table_verification.md](../ch10b_rl_for_ci/notes/translation_table_verification.md)
- [../ch10c_adaptive_experiments/notes/claim_source_ledger_2026-07-24.md](../ch10c_adaptive_experiments/notes/claim_source_ledger_2026-07-24.md)

## Reference and process docs

- [journal_target.md](journal_target.md) - venue analysis for journal submissions.
- [bloat.md](bloat.md) - the bloat-detection and trimming guide for the editing pass.
- [note_macro.md](note_macro.md) - notes on the LaTeX macro approach (relevant to Workstream 2 S3).
- [humanizer_edits_report.md](humanizer_edits_report.md) - the de-AI prose pass report.
- [simulation_audit_2026-03-07.md](simulation_audit_2026-03-07.md) - earlier sim audit snapshot.
- [ch10bc_reconciliation_verification_2026-07-24.md](ch10bc_reconciliation_verification_2026-07-24.md) -
  final proof, source, cold-run, build, and independent-gate record for chapters 10b and 10c.

## Draft content (superseded by chapters, kept for provenance)

- [CI_RL.md](CI_RL.md) and [CI_RL_proofread_2026-05-16.md](CI_RL_proofread_2026-05-16.md) -
  early causal-inference-RL draft, now covered by `ch10_causal/`, `ch10b_rl_for_ci/`, and
  `ch10c_adaptive_experiments/`.

## Per-simulation audits

The `../audits/` directory holds dated per-simulation audit reports (one per sim, plus polish
and re-audit passes). See [../audits/_INDEX.md](../audits/_INDEX.md).

## Planned additions (roadmap W4, W5)

Forthcoming, not yet written: `proof-library.md` (key RL results indexed by theorem, each with
the cleanest source proof and citation; roadmap W5) and per-topic synthesis articles
(deadly-triad, offline-RL, bandits-pricing, MFG-macro, RLHF-alignment, causal-OPE,
world-models; roadmap W4). A primary-source RAG retrieval tool (roadmap W3) will back both.
