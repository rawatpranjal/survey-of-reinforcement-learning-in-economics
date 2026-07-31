#!/usr/bin/env python3
r"""Validate citations and the active BibTeX database for docs/main.tex.

The checker is intentionally dependency-free so it can run in a fresh CI
runner before LaTeX is installed. All ``\input`` paths are resolved relative
to the main document's directory, matching this repository's compile rules.

Usage:
    python3 scripts/check_bib.py
    python3 scripts/check_bib.py --main docs/main.tex
    python3 scripts/check_bib.py --strict
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


CODE_ENVIRONMENTS = ("verbatim", "Verbatim", "minted", "lstlisting", "comment")
DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$", re.IGNORECASE)
ENTRY_START_RE = re.compile(r"@([A-Za-z]+)\s*([({])")
INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")
BIBLIOGRAPHY_RE = re.compile(r"\\bibliography\s*\{([^{}]+)\}")
CITATION_RE = re.compile(
    r"\\(?P<command>nocite|cite[A-Za-z]*)\*?\s*"
    r"(?:\[[^\]]*\]\s*){0,2}\{(?P<keys>[^{}]*)\}",
    re.IGNORECASE,
)

REQUIRED_FIELDS = {
    "article": ("author", "title", "journal", "year"),
    "inproceedings": ("author", "title", "booktitle", "year"),
    "conference": ("author", "title", "booktitle", "year"),
    "book": ("author", "title", "publisher", "year"),
    "incollection": ("author", "title", "booktitle", "publisher", "year"),
    "techreport": ("author", "title", "institution", "year"),
    "phdthesis": ("author", "title", "school", "year"),
    "mastersthesis": ("author", "title", "school", "year"),
    "unpublished": ("author", "title", "note"),
    "misc": ("title",),
}


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    path: Path | None = None
    line: int = 0

    def sort_key(self):
        return (self.code, str(self.path or ""), self.line, self.message)

    def render(self, severity: str, root: Path) -> str:
        location = ""
        if self.path:
            try:
                display_path = self.path.relative_to(root)
            except ValueError:
                display_path = self.path
            location = f" {display_path}"
            if self.line:
                location += f":{self.line}"
        return f"{severity} [{self.code}]{location} {self.message}"


@dataclass
class BibEntry:
    entry_type: str
    key: str
    fields: dict[str, str]
    path: Path
    line: int


@dataclass
class CheckReport:
    main_path: Path
    tex_files: set[Path] = field(default_factory=set)
    bib_files: set[Path] = field(default_factory=set)
    cited_keys: set[str] = field(default_factory=set)
    entries: list[BibEntry] = field(default_factory=list)
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)

    @property
    def tex_file_count(self) -> int:
        return len(self.tex_files)

    @property
    def unique_bib_keys(self) -> set[str]:
        return {entry.key for entry in self.entries}

    def failed(self, strict: bool = False) -> bool:
        return bool(self.errors or (strict and self.warnings))


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _strip_latex_comments(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines(keepends=True):
        comment_at = None
        for index, char in enumerate(line):
            if char != "%":
                continue
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                comment_at = index
                break
        if comment_at is None:
            cleaned_lines.append(line)
        elif line.endswith("\n"):
            cleaned_lines.append(line[:comment_at] + "\n")
        else:
            cleaned_lines.append(line[:comment_at])
    return "".join(cleaned_lines)


def _sanitize_latex(text: str) -> str:
    for environment in CODE_ENVIRONMENTS:
        pattern = re.compile(
            rf"\\begin\{{{re.escape(environment)}\}}.*?"
            rf"\\end\{{{re.escape(environment)}\}}",
            re.DOTALL,
        )
        text = pattern.sub(lambda match: "\n" * match.group(0).count("\n"), text)
    return _strip_latex_comments(text)


def _resolve_compile_path(compile_dir: Path, raw_path: str, suffix: str) -> Path:
    candidate = Path(raw_path.strip())
    if not candidate.suffix:
        candidate = candidate.with_suffix(suffix)
    if not candidate.is_absolute():
        candidate = compile_dir / candidate
    return candidate.resolve()


def _split_top_level(text: str) -> list[str]:
    parts = []
    start = 0
    braces = 0
    parentheses = 0
    quoted = False
    escaped = False
    for index, char in enumerate(text):
        if char == '"' and not escaped:
            quoted = not quoted
        elif not quoted:
            if char == "{":
                braces += 1
            elif char == "}":
                braces -= 1
            elif char == "(":
                parentheses += 1
            elif char == ")":
                parentheses -= 1
            elif char == "," and braces == 0 and parentheses == 0:
                parts.append(text[start:index])
                start = index + 1
        escaped = char == "\\" and not escaped
        if char != "\\":
            escaped = False
    parts.append(text[start:])
    return parts


def _strip_outer_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2:
        if value[0] == "{" and value[-1] == "}":
            return value[1:-1].strip()
        if value[0] == '"' and value[-1] == '"':
            return value[1:-1].strip()
    return value


def _parse_bib_file(path: Path, report: CheckReport) -> list[BibEntry]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        report.errors.append(
            Issue("bib-read-error", f"Could not read bibliography: {exc}", path)
        )
        return []

    entries = []
    cursor = 0
    while True:
        match = ENTRY_START_RE.search(text, cursor)
        if not match:
            break
        entry_type = match.group(1).lower()
        opening = match.group(2)
        closing = "}" if opening == "{" else ")"
        body_start = match.end()
        depth = 1
        quoted = False
        escaped = False
        index = body_start
        while index < len(text) and depth:
            char = text[index]
            if char == '"' and not escaped:
                quoted = not quoted
            elif not quoted:
                if char == opening:
                    depth += 1
                elif char == closing:
                    depth -= 1
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
            index += 1

        line = _line_number(text, match.start())
        if depth:
            report.errors.append(
                Issue(
                    "malformed-bibtex",
                    f"Unclosed @{entry_type} record.",
                    path,
                    line,
                )
            )
            break

        body = text[body_start : index - 1]
        cursor = index
        if entry_type in {"comment", "string", "preamble"}:
            continue

        parts = _split_top_level(body)
        key = parts[0].strip() if parts else ""
        if not key or len(parts) < 2:
            report.errors.append(
                Issue(
                    "malformed-bibtex",
                    f"@{entry_type} record lacks a key or fields.",
                    path,
                    line,
                )
            )
            continue

        fields = {}
        malformed_field = False
        for part in parts[1:]:
            if not part.strip():
                continue
            if "=" not in part:
                report.errors.append(
                    Issue(
                        "malformed-bibtex",
                        f"Entry '{key}' contains a field without '='.",
                        path,
                        line,
                    )
                )
                malformed_field = True
                continue
            name, value = part.split("=", 1)
            name = name.strip().lower()
            if not name or not value.strip():
                report.errors.append(
                    Issue(
                        "malformed-bibtex",
                        f"Entry '{key}' contains an empty field name or value.",
                        path,
                        line,
                    )
                )
                malformed_field = True
                continue
            fields[name] = _strip_outer_value(value)
        if not malformed_field or fields:
            entries.append(BibEntry(entry_type, key, fields, path, line))
    return entries


def _normalize_tex_text(value: str) -> str:
    value = re.sub(
        r"\\[\'\"`^~=.uvHckbdtr]\s*\{?([A-Za-z])\}?",
        r"\1",
        value,
    )
    value = re.sub(r"\\[A-Za-z]+\*?", "", value)
    value = value.replace("\\&", "and")
    value = value.replace("{", "").replace("}", "")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _first_author(author: str) -> str:
    return re.split(r"\s+and\s+", author, maxsplit=1, flags=re.IGNORECASE)[0]


def _scan_tex_tree(main_path: Path, report: CheckReport):
    compile_dir = main_path.parent
    pending = [main_path]
    cite_locations: dict[str, tuple[Path, int]] = {}
    nocite_all = False

    while pending:
        path = pending.pop()
        if path in report.tex_files:
            continue
        report.tex_files.add(path)
        try:
            raw_text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            report.errors.append(
                Issue("tex-read-error", f"Could not read TeX input: {exc}", path)
            )
            continue
        text = _sanitize_latex(raw_text)

        for match in CITATION_RE.finditer(text):
            keys = [key.strip() for key in match.group("keys").split(",") if key.strip()]
            if match.group("command").lower() == "nocite" and "*" in keys:
                nocite_all = True
                continue
            for key in keys:
                report.cited_keys.add(key)
                cite_locations.setdefault(
                    key, (path, _line_number(text, match.start()))
                )

        for match in INPUT_RE.finditer(text):
            raw_input = match.group(1).strip()
            input_path = _resolve_compile_path(compile_dir, raw_input, ".tex")
            if not input_path.is_file():
                report.errors.append(
                    Issue(
                        "unresolved-input",
                        f"Input '{raw_input}' does not resolve from {compile_dir}.",
                        path,
                        _line_number(text, match.start()),
                    )
                )
                continue
            pending.append(input_path)

        for match in BIBLIOGRAPHY_RE.finditer(text):
            for raw_bib in match.group(1).split(","):
                bib_path = _resolve_compile_path(compile_dir, raw_bib, ".bib")
                if not bib_path.is_file():
                    report.errors.append(
                        Issue(
                            "unresolved-bibliography",
                            f"Bibliography '{raw_bib.strip()}' does not resolve.",
                            path,
                            _line_number(text, match.start()),
                        )
                    )
                    continue
                report.bib_files.add(bib_path)

    return cite_locations, nocite_all


def _validate_entries(
    report: CheckReport,
    cite_locations: dict[str, tuple[Path, int]],
    nocite_all: bool,
):
    key_counts = Counter(entry.key for entry in report.entries)
    entries_by_key = defaultdict(list)
    for entry in report.entries:
        entries_by_key[entry.key].append(entry)

    for key, count in sorted(key_counts.items()):
        if count > 1:
            entry = entries_by_key[key][0]
            report.errors.append(
                Issue(
                    "duplicate-key",
                    f"BibTeX key '{key}' is defined {count} times.",
                    entry.path,
                    entry.line,
                )
            )

    unique_keys = set(key_counts)
    if nocite_all:
        report.cited_keys.update(unique_keys)

    for key in sorted(report.cited_keys - unique_keys):
        path, line = cite_locations.get(key, (report.main_path, 0))
        report.errors.append(
            Issue(
                "missing-citation",
                f"Cited key '{key}' is absent from the active bibliography.",
                path,
                line,
            )
        )

    for key in sorted(unique_keys - report.cited_keys):
        entry = entries_by_key[key][0]
        report.warnings.append(
            Issue(
                "orphan-entry",
                f"BibTeX key '{key}' is defined but never cited.",
                entry.path,
                entry.line,
            )
        )

    duplicate_signatures = defaultdict(set)
    for entry in report.entries:
        signature = (
            _normalize_tex_text(entry.fields.get("title", "")),
            _normalize_tex_text(_first_author(entry.fields.get("author", ""))),
            _normalize_tex_text(entry.fields.get("year", "")),
        )
        if all(signature):
            duplicate_signatures[signature].add(entry.key)
    for keys in sorted(
        (sorted(keys) for keys in duplicate_signatures.values() if len(keys) > 1),
        key=lambda group: tuple(group),
    ):
        first = entries_by_key[keys[0]][0]
        report.warnings.append(
            Issue(
                "duplicate-work",
                f"Likely duplicate work appears under keys: {', '.join(keys)}.",
                first.path,
                first.line,
            )
        )

    for entry in report.entries:
        missing = [
            name
            for name in REQUIRED_FIELDS.get(entry.entry_type, ())
            if not entry.fields.get(name, "").strip()
        ]
        if missing:
            report.errors.append(
                Issue(
                    "missing-field",
                    f"Entry '{entry.key}' is missing required field(s): "
                    f"{', '.join(missing)}.",
                    entry.path,
                    entry.line,
                )
            )

        doi = entry.fields.get("doi", "").strip()
        if doi and not DOI_RE.fullmatch(doi):
            report.errors.append(
                Issue(
                    "invalid-doi",
                    f"Entry '{entry.key}' has malformed DOI '{doi}'.",
                    entry.path,
                    entry.line,
                )
            )

        if (
            entry.entry_type in {"inproceedings", "conference"}
            and entry.fields.get("volume")
            and entry.fields.get("number")
        ):
            report.warnings.append(
                Issue(
                    "incompatible-fields",
                    f"Entry '{entry.key}' defines both volume and number; "
                    "plainnat ignores one of them.",
                    entry.path,
                    entry.line,
                )
            )


def check_project(main_path: str | Path = "docs/main.tex") -> CheckReport:
    main_path = Path(main_path).expanduser().resolve()
    if not main_path.is_file():
        raise FileNotFoundError(f"main TeX file not found: {main_path}")

    report = CheckReport(main_path=main_path)
    cite_locations, nocite_all = _scan_tex_tree(main_path, report)
    if not report.bib_files:
        report.errors.append(
            Issue(
                "missing-bibliography",
                "No active \\bibliography{...} command was found.",
                main_path,
            )
        )
    for bib_path in sorted(report.bib_files):
        report.entries.extend(_parse_bib_file(bib_path, report))
    _validate_entries(report, cite_locations, nocite_all)
    report.errors.sort(key=Issue.sort_key)
    report.warnings.sort(key=Issue.sort_key)
    return report


def _print_report(report: CheckReport, strict: bool):
    root = report.main_path.parent.parent
    print(f"Bibliography check: {report.main_path}")
    print(f"  TeX files: {report.tex_file_count}")
    print(f"  Bibliographies: {len(report.bib_files)}")
    print(f"  Cited keys: {len(report.cited_keys)}")
    print(
        f"  BibTeX records: {len(report.entries)} "
        f"({len(report.unique_bib_keys)} unique keys)"
    )
    print()
    for issue in report.errors:
        print(issue.render("ERROR", root))
    for issue in report.warnings:
        print(issue.render("WARNING", root))
    if report.errors or report.warnings:
        print()
    if report.failed(strict):
        reason = "strict warnings" if strict and not report.errors else "validation errors"
        print(
            f"Result: FAIL ({len(report.errors)} error(s), "
            f"{len(report.warnings)} warning(s); {reason})"
        )
    elif report.warnings:
        print(f"Result: PASS with {len(report.warnings)} warning(s)")
    else:
        print("Result: PASS")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main",
        default="docs/main.tex",
        help="main LaTeX document (default: docs/main.tex)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat hygiene warnings as failures",
    )
    args = parser.parse_args(argv)

    try:
        report = check_project(args.main)
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_report(report, args.strict)
    return 1 if report.failed(args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
