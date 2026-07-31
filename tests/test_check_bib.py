import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.check_bib import check_project  # noqa: E402


ARTICLE = """\
@article{{{key},
  author = {{Ada Author}},
  title = {{{title}}},
  journal = {{Journal of Tests}},
  year = {{2024}}{extra}
}}
"""


class BibliographyCheckTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "docs").mkdir()

    def tearDown(self):
        self.tempdir.cleanup()

    def write(self, relative_path, content):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def make_main(self, body, bib="refs"):
        return self.write(
            "docs/main.tex",
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            f"{body}\n"
            f"\\bibliography{{{bib}}}\n"
            "\\end{document}\n",
        )

    def issue_codes(self, issues):
        return [issue.code for issue in issues]

    def test_scans_nested_inputs_relative_to_compile_directory(self):
        main = self.make_main("\\citep{Alpha}\n\\input{../chapter/part}")
        self.write(
            "chapter/part.tex",
            "\\citet[see][p.~2]{Beta}\n\\input{../tables/fragment}",
        )
        self.write("tables/fragment.tex", "No additional citation.")
        self.write(
            "docs/refs.bib",
            ARTICLE.format(key="Alpha", title="Alpha", extra="")
            + ARTICLE.format(key="Beta", title="Beta", extra=""),
        )

        report = check_project(main)

        self.assertEqual(report.cited_keys, {"Alpha", "Beta"})
        self.assertEqual(report.tex_file_count, 3)
        self.assertEqual(report.errors, [])

    def test_ignores_comments_and_code_environments(self):
        main = self.make_main(
            "% \\input{missing}\n"
            "\\begin{minted}{tex}\n"
            "\\cite{Ghost}\n"
            "\\end{minted}\n"
            "\\cite{Real}"
        )
        self.write(
            "docs/refs.bib",
            ARTICLE.format(key="Real", title="Real", extra=""),
        )

        report = check_project(main)

        self.assertEqual(report.cited_keys, {"Real"})
        self.assertNotIn("unresolved-input", self.issue_codes(report.errors))

    def test_reports_cited_but_missing_key_as_error(self):
        main = self.make_main("\\cite{Missing}")
        self.write("docs/refs.bib", ARTICLE.format(key="Present", title="Present", extra=""))

        report = check_project(main)

        self.assertIn("missing-citation", self.issue_codes(report.errors))

    def test_reports_unresolved_input_as_error(self):
        main = self.make_main("\\input{../chapter/missing}")
        self.write("docs/refs.bib", "")

        report = check_project(main)

        self.assertIn("unresolved-input", self.issue_codes(report.errors))

    def test_reports_duplicate_bibtex_key_as_error(self):
        main = self.make_main("\\cite{Repeated}")
        self.write(
            "docs/refs.bib",
            ARTICLE.format(key="Repeated", title="First", extra="")
            + ARTICLE.format(key="Repeated", title="Second", extra=""),
        )

        report = check_project(main)

        self.assertIn("duplicate-key", self.issue_codes(report.errors))

    def test_reports_malformed_bibtex_as_error(self):
        main = self.make_main("\\cite{Broken}")
        self.write(
            "docs/refs.bib",
            "@article{Broken,\n  author = {Ada Author},\n  title = {Never closed}\n",
        )

        report = check_project(main)

        self.assertIn("malformed-bibtex", self.issue_codes(report.errors))

    def test_reports_missing_required_field_as_error(self):
        main = self.make_main("\\cite{NoYear}")
        self.write(
            "docs/refs.bib",
            "@article{NoYear,\n"
            "  author = {Ada Author},\n"
            "  title = {No Year},\n"
            "  journal = {Journal of Tests}\n"
            "}\n",
        )

        report = check_project(main)

        self.assertIn("missing-field", self.issue_codes(report.errors))

    def test_reports_invalid_doi_as_error(self):
        main = self.make_main("\\cite{BadDoi}")
        self.write(
            "docs/refs.bib",
            ARTICLE.format(
                key="BadDoi",
                title="Bad DOI",
                extra=",\n  doi = {not-a-doi}",
            ),
        )

        report = check_project(main)

        self.assertIn("invalid-doi", self.issue_codes(report.errors))

    def test_hygiene_findings_warn_without_failing(self):
        main = self.make_main("\\cite{Canonical}")
        self.write(
            "docs/refs.bib",
            ARTICLE.format(key="Canonical", title="Same Work", extra="")
            + ARTICLE.format(key="DuplicateWork", title="Same Work", extra="")
            + "@inproceedings{ProceedingsFields,\n"
            "  author = {Ada Author},\n"
            "  title = {Proceedings Fields},\n"
            "  booktitle = {Proceedings of Tests},\n"
            "  volume = {1},\n"
            "  number = {2},\n"
            "  year = {2024}\n"
            "}\n",
        )

        report = check_project(main)

        codes = self.issue_codes(report.warnings)
        self.assertEqual(report.errors, [])
        self.assertIn("orphan-entry", codes)
        self.assertIn("duplicate-work", codes)
        self.assertIn("incompatible-fields", codes)
        self.assertFalse(report.failed(strict=False))
        self.assertTrue(report.failed(strict=True))

    def test_nocite_star_marks_every_entry_as_used(self):
        main = self.make_main("\\nocite{*}")
        self.write(
            "docs/refs.bib",
            ARTICLE.format(key="Alpha", title="Alpha", extra="")
            + ARTICLE.format(key="Beta", title="Beta", extra=""),
        )

        report = check_project(main)

        self.assertEqual(report.cited_keys, {"Alpha", "Beta"})
        self.assertNotIn("orphan-entry", self.issue_codes(report.warnings))

    def test_cli_returns_zero_for_warnings_and_one_in_strict_mode(self):
        main = self.make_main("\\cite{Used}")
        self.write(
            "docs/refs.bib",
            ARTICLE.format(key="Used", title="Used", extra="")
            + ARTICLE.format(key="Unused", title="Unused", extra=""),
        )
        command = [sys.executable, str(REPO_ROOT / "scripts/check_bib.py"), "--main", str(main)]

        normal = subprocess.run(command, capture_output=True, text=True)
        strict = subprocess.run(command + ["--strict"], capture_output=True, text=True)

        self.assertEqual(normal.returncode, 0, normal.stdout + normal.stderr)
        self.assertEqual(strict.returncode, 1, strict.stdout + strict.stderr)

    def test_cli_returns_two_for_missing_main_file(self):
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/check_bib.py"),
            "--main",
            str(self.root / "absent.tex"),
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
