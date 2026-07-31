import unittest
from pathlib import Path

from scripts.run_all_sims import REGISTRY


REPO_ROOT = Path(__file__).resolve().parents[1]


class SimulationRegistryTests(unittest.TestCase):
    def test_registered_paths_are_unique_and_exist(self):
        paths = [path for _, path, _ in REGISTRY]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(
            [path for path in paths if not (REPO_ROOT / path).is_file()],
            [],
        )

    def test_stochastic_engine_studies_are_compute_category(self):
        categories = {path: category for _, path, category in REGISTRY}
        expected = {
            "ch03_theory/sims/engine_value_learning.py": "A",
            "ch10_causal/sims/engine_confounding.py": "A",
            "ch12_world_models/sims/engine_model_learning.py": "A",
        }
        for path, category in expected.items():
            self.assertEqual(categories[path], category)

    def test_engine_value_learning_plots_only_requires_cache(self):
        source = (REPO_ROOT / "ch03_theory/sims/engine_value_learning.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("add_cache_args(parser)", source)
        self.assertIn("if args.plots_only:", source)
        self.assertIn("raise FileNotFoundError(", source)
        self.assertIn("No cache found. Run without --plots-only first.", source)

    def test_runner_limits_algo_flags_to_component_scripts(self):
        source = (REPO_ROOT / "scripts/run_all_sims.py").read_text(encoding="utf-8")
        self.assertIn('if "add_component_args" in source:', source)

    def test_engine_later_uses_table_is_retired(self):
        source = (REPO_ROOT / "ch02_rl_algorithms/sims/engine_algorithms.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("engine_pointer", source)
        self.assertFalse(
            (REPO_ROOT / "ch02_rl_algorithms/sims/engine_pointer.tex").exists()
        )


if __name__ == "__main__":
    unittest.main()
