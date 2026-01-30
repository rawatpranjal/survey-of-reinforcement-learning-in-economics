#!/usr/bin/env python3
"""
Split Bertsekas RL Course 2nd Edition into searchable sections.

Uses manually extracted TOC from the book to create semantically meaningful splits.
Page numbers from actual TOC on pages 6-9 of the PDF.
"""

import fitz  # PyMuPDF
import json
from pathlib import Path


# Configuration
INPUT_PDF = Path("/Users/pranjal/Code/rl/ch02_planning_learning/papers/RLCOURSECOMPLETE 2ndEDITION.pdf")
OUTPUT_DIR = Path("/Users/pranjal/Code/rl/ch02_planning_learning/papers/bertsekas_rl_2ed")

# Manually extracted TOC from the PDF (pages 6-9)
# Format: (id, title, start_page, end_page, topics)
# Page numbers are from the actual TOC
SECTIONS = [
    # Frontmatter (pages 1-9)
    ("00_frontmatter", "Frontmatter and Contents", 1, 9,
     ["title page", "about author", "contents", "table of contents"]),

    # Chapter 1: Exact and Approximate Dynamic Programming (p. 4-165)
    ("01_01_alphazero_det_dp", "1.1-1.2: AlphaZero & Deterministic DP", 10, 27,
     ["AlphaZero", "offline training", "online play", "deterministic DP", "finite horizon",
      "DP algorithm", "approximation in value space", "rollout"]),

    ("01_02_stochastic_dp", "1.3: Stochastic Exact and Approximate DP", 28, 41,
     ["stochastic DP", "finite horizon stochastic", "approximation in value space",
      "approximation in policy space", "training cost function", "policy approximation"]),

    ("01_03_infinite_horizon", "1.4: Infinite Horizon Problems Overview", 42, 55,
     ["infinite horizon", "infinite horizon methodology", "approximation in value space",
      "understanding approximation"]),

    ("01_04_newton_lq", "1.5: Newton's Method & LQ Problems", 56, 78,
     ["Newton's method", "linear quadratic", "LQ problems", "region of stability",
      "rollout", "policy iteration", "error bounds"]),

    ("01_05_examples_modeling", "1.6.1-1.6.4: Examples & Modeling", 79, 91,
     ["modeling", "termination state", "discrete optimization", "finite to infinite horizon",
      "reformulations"]),

    ("01_06_state_augmentation", "1.6.5-1.6.6: State Augmentation & POMDP", 92, 101,
     ["state augmentation", "time delays", "forecasts", "partial state information",
      "belief states", "POMDP"]),

    ("01_07_multiagent_adaptive", "1.6.7-1.6.8: Multiagent & Adaptive Control", 102, 117,
     ["multiagent problems", "multiagent rollout", "unknown model", "adaptive control"]),

    ("01_08_mpc", "1.6.9: Model Predictive Control", 118, 130,
     ["model predictive control", "MPC", "receding horizon", "certainty equivalence"]),

    ("01_09_rl_dp_relations", "1.7: Reinforcement Learning and Decision/Control", 131, 140,
     ["RL terminology", "RL notation", "DP vs RL", "LLM synergy", "machine learning",
      "optimization"]),

    ("01_10_notes_exercises", "1.8: Notes, Sources, and Exercises Ch1", 141, 165,
     ["exercises", "notes", "sources", "chapter 1"]),

    # Chapter 2: Approximation in Value Space - Rollout Algorithms (p. 166-348)
    ("02_01_det_finite_horizon", "2.1-2.2: Deterministic Finite Horizon & Approximation", 166, 176,
     ["deterministic finite horizon", "approximation in value space"]),

    ("02_02_rollout_discrete_opt", "2.3: Rollout for Discrete Optimization", 177, 207,
     ["rollout algorithms", "discrete optimization", "sequential consistency",
      "sequential improvement", "fortified rollout", "parallel rollout", "truncated rollout"]),

    ("02_03_model_free_rollout", "2.3.7-2.3.10: Model-Free Rollout & Inference", 208, 220,
     ["expert rollout", "model-free rollout", "world model", "local search",
      "n-grams", "transformers", "HMM", "Markov chains"]),

    ("02_04_multistep_lookahead", "2.4-2.5: Multistep Lookahead & Constrained Rollout", 221, 256,
     ["multistep lookahead", "iterative deepening", "forward DP", "incremental rollout",
      "constrained rollout", "integer programming"]),

    ("02_05_continuous_time", "2.6-2.7.2: Continuous-Time & Stochastic Rollout", 257, 270,
     ["continuous-time rollout", "small stage costs", "long horizon",
      "stochastic problems", "simplified rollout", "certainty equivalence"]),

    ("02_06_mcts", "2.7.3-2.7.6: Simulation-Based & MCTS", 271, 282,
     ["simulation-based rollout", "variance reduction", "Monte Carlo tree search",
      "MCTS", "randomized policy improvement"]),

    ("02_07_infinite_spaces", "2.8: Rollout for Infinite-Spaces Problems", 283, 296,
     ["infinite spaces", "optimization heuristics", "deterministic problems",
      "stochastic programming", "certainty equivalence"]),

    ("02_08_multiagent_rollout", "2.9: Multiagent Rollout", 297, 308,
     ["multiagent rollout", "asynchronous rollout", "autonomous rollout"]),

    ("02_09_bayesian_opt", "2.10: Rollout for Bayesian Optimization", 309, 316,
     ["Bayesian optimization", "sequential estimation", "rollout"]),

    ("02_10_adaptive_pomdp", "2.11: Adaptive Control by Rollout with POMDP", 317, 326,
     ["adaptive control", "POMDP formulation", "rollout"]),

    ("02_11_minimax", "2.12: Minimax Control and RL", 327, 340,
     ["minimax control", "minimax DP", "minimax approximation", "computer chess",
      "sequential games", "noncooperative games"]),

    ("02_12_notes_exercises_ch2", "2.13: Notes, Sources, and Exercises Ch2", 341, 348,
     ["exercises", "notes", "sources", "chapter 2"]),

    # Chapter 3: Learning Values and Policies (p. 349-454)
    ("03_01_parametric_approx", "3.1: Parametric Approximation Architectures", 349, 369,
     ["parametric approximation", "cost function approximation", "feature-based",
      "linear architecture", "nonlinear architecture", "training"]),

    ("03_02_neural_networks", "3.2: Neural Networks", 370, 376,
     ["neural networks", "training neural networks", "multilayer", "deep neural networks"]),

    ("03_03_learning_cost", "3.3: Learning Cost Functions in Approximate DP", 377, 396,
     ["fitted value iteration", "Q-factor approximation", "model-free",
      "approximate policy iteration", "SARSA", "DQN", "advantage updating"]),

    ("03_04_learning_policy", "3.4: Learning a Policy in Approximate DP", 397, 405,
     ["policy learning", "classifiers", "policy network", "value network",
      "lookahead minimization"]),

    ("03_05_policy_gradient", "3.5: Policy Gradient and Related Methods", 406, 426,
     ["policy gradient", "gradient methods", "proximal policy optimization", "PPO",
      "random direction", "random search", "cross-entropy"]),

    ("03_06_aggregation", "3.6: Aggregation", 427, 448,
     ["aggregation", "representative states", "discretization", "POMDP discretization",
      "error bounds", "feature aggregation", "biased aggregation", "distributed aggregation"]),

    ("03_07_notes_exercises_ch3", "3.7: Notes, Sources, and Exercises Ch3", 449, 454,
     ["exercises", "notes", "sources", "chapter 3"]),

    # References
    ("04_references", "References", 455, 521,
     ["references", "bibliography", "citations"]),
]


def split_pdf(input_path: Path, output_dir: Path, sections: list) -> list[dict]:
    """Split PDF according to predefined sections."""
    doc = fitz.open(input_path)
    total_pages = len(doc)
    print(f"PDF has {total_pages} pages")

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for section_id, title, start_page, end_page, topics in sections:
        # Clamp to actual page count
        start_page = max(1, start_page)
        end_page = min(end_page, total_pages)

        if start_page > total_pages:
            print(f"  Skipping {section_id}: start page {start_page} > total pages {total_pages}")
            continue

        filename = f"{section_id}.pdf"
        filepath = output_dir / filename

        # Extract pages (fitz uses 0-indexed pages)
        start_idx = start_page - 1
        end_idx = end_page  # will be exclusive in insert_pdf

        # Create new PDF with selected pages
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=start_idx, to_page=end_idx-1)
        new_doc.save(filepath)
        new_doc.close()

        page_count = end_page - start_page + 1
        print(f"  Created: {filename} ({page_count} pages, p.{start_page}-{end_page})")

        results.append({
            'id': section_id,
            'file': filename,
            'title': title,
            'pages': f"{start_page}-{end_page}",
            'page_count': page_count,
            'topics': topics,
            'summary': ''  # To be filled manually or by AI
        })

    doc.close()
    return results


def create_index(sections: list[dict], output_dir: Path):
    """Create searchable index.json."""
    index = {
        'book': 'A Course in Reinforcement Learning, 2nd Edition',
        'author': 'Dimitri P. Bertsekas',
        'year': 2025,
        'publisher': 'Athena Scientific',
        'source_pdf': 'RLCOURSECOMPLETE 2ndEDITION.pdf',
        'total_sections': len(sections),
        'chapters': [
            {'num': 0, 'title': 'Frontmatter', 'sections': '00_*'},
            {'num': 1, 'title': 'Exact and Approximate Dynamic Programming', 'sections': '01_*'},
            {'num': 2, 'title': 'Approximation in Value Space - Rollout Algorithms', 'sections': '02_*'},
            {'num': 3, 'title': 'Learning Values and Policies', 'sections': '03_*'},
            {'num': 4, 'title': 'References', 'sections': '04_*'},
        ],
        'sections': sections,
        'usage': 'Search topics array to find relevant sections, then read the corresponding PDF file.'
    }

    index_path = output_dir / 'index.json'
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"\nCreated index: {index_path}")


def cleanup_old_splits(output_dir: Path):
    """Remove old arbitrary splits."""
    for f in output_dir.glob("*.pdf"):
        if "pages_" in f.name or f.name.startswith("0") and "_00_" in f.name:
            print(f"  Removing old split: {f.name}")
            f.unlink()


def main():
    print(f"Input PDF: {INPUT_PDF}")
    print(f"Output dir: {OUTPUT_DIR}\n")

    # Clean up old splits
    print("Cleaning up old splits...")
    cleanup_old_splits(OUTPUT_DIR)

    # Remove old index
    old_index = OUTPUT_DIR / 'index.json'
    if old_index.exists():
        old_index.unlink()

    # Split PDF
    print("\nSplitting PDF by sections...")
    sections = split_pdf(INPUT_PDF, OUTPUT_DIR, SECTIONS)

    # Create index
    create_index(sections, OUTPUT_DIR)

    print(f"\nDone! Created {len(sections)} section PDFs + index.json")


if __name__ == '__main__':
    main()
