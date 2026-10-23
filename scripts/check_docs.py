#!/usr/bin/env python3
"""Mechanical checks for the documentation site.

Hard failures (exit 1):
  - a relative Markdown link pointing at a file that does not exist
  - a navigation entry pointing at a missing page
  - a page with no ``quadrant:`` front matter, or an unknown one
  - a page whose quadrant disagrees with the navigation section it sits in
  - a page that no page in the navigation links to
  - a table cell ending in an ellipsis
  - an em dash

Usage:
    python scripts/check_docs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
ROOT_MD = [
    "README.md", "CONTRIBUTING.md", "CHANGELOG.md", "ROADMAP.md",
    "ISSUES.md", "AI-USAGE.md", "CODE_OF_CONDUCT.md", "DATABASE.md",
]
EXTRA = ["examples/README.md"]

LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
FENCE = re.compile(r"```.*?```", re.S)
TRUNCATED_CELL = re.compile(r"…\.|\.\.\.\s*\|?\s*$")
QUADRANT = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)
EM_DASH = re.compile("—")

# Navigation section title -> the quadrant its pages must declare.
SECTIONS = {
    "Home": "home",
    "Tutorials": "tutorial",
    "How-to guides": "how-to",
    "Reference": "reference",
    "Explanation": "explanation",
}
QUADRANTS = set(SECTIONS.values())


def md_files() -> list[Path]:
    """Every reader-facing Markdown file. Generated files are left out."""
    files = sorted(DOCS.rglob("*.md"))
    files += [ROOT / f for f in ROOT_MD + EXTRA if (ROOT / f).exists()]
    notes = ROOT / "notes"
    if notes.is_dir():
        files += sorted(notes.glob("*.md"))
    return files


def prose(text: str) -> str:
    """Drop fenced code so code samples are never style-policed."""
    return FENCE.sub("", text)


def quadrant_of(path: Path) -> str:
    m = QUADRANT.match(path.read_text(encoding="utf-8"))
    if not m:
        return ""
    q = re.search(r"^quadrant:\s*(\S+)", m.group(1), re.M)
    return q.group(1) if q else ""


def nav_pages() -> dict[str, str]:
    """page path (relative to docs/) -> navigation section."""
    data = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    found: dict[str, str] = {}

    def walk(items, section):
        for item in items:
            if isinstance(item, str):
                found[item] = section
                continue
            for key, value in item.items():
                if isinstance(value, str):
                    found[value] = section or key
                else:
                    walk(value, section or key)

    walk(data.get("nav", []), None)
    return found


def check_links(files: list[Path]) -> list[str]:
    errors = []
    for f in files:
        for target in LINK.findall(prose(f.read_text(encoding="utf-8"))):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path = target.split("#")[0]
            if path and not (f.parent / path).resolve().exists():
                errors.append(f"{f.relative_to(ROOT)}: dead link -> {target}")
    return errors


def check_nav_targets(nav: dict[str, str]) -> list[str]:
    return [
        f"mkdocs.yml: nav points at missing docs/{page}"
        for page in nav
        if not (DOCS / page).exists()
    ]


def check_quadrants(files: list[Path], nav: dict[str, str]) -> list[str]:
    errors = []
    for f in files:
        if not f.is_relative_to(DOCS):
            continue
        rel = str(f.relative_to(DOCS))
        q = quadrant_of(f)
        if not q:
            errors.append(f"{rel}: no quadrant front matter")
            continue
        if q not in QUADRANTS:
            errors.append(f"{rel}: unknown quadrant {q!r}")
            continue
        expected = SECTIONS.get(nav.get(rel, ""))
        if expected and q != expected:
            errors.append(
                f"{rel}: quadrant {q!r} but nav section expects {expected!r}"
            )
    return errors


def check_reachable(nav: dict[str, str]) -> list[str]:
    """Every docs page must be in the nav or linked from a page that is."""
    seen: set[str] = set()
    queue = [DOCS / page for page in nav if (DOCS / page).exists()]
    while queue:
        page = queue.pop()
        rel = str(page.relative_to(DOCS))
        if rel in seen:
            continue
        seen.add(rel)
        for target in LINK.findall(prose(page.read_text(encoding="utf-8"))):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path = (page.parent / target.split("#")[0]).resolve()
            if path.suffix == ".md" and path.exists() and path.is_relative_to(DOCS):
                queue.append(path)
    return [
        f"{f.relative_to(DOCS)}: not reachable from the navigation"
        for f in sorted(DOCS.rglob("*.md"))
        if str(f.relative_to(DOCS)) not in seen
    ]


def check_truncated_cells(files: list[Path]) -> list[str]:
    errors = []
    for f in files:
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("|") and TRUNCATED_CELL.search(line):
                errors.append(f"{f.relative_to(ROOT)}:{n}: table cell ends in an ellipsis")
    return errors


def check_em_dashes(files: list[Path]) -> list[str]:
    """Prose only: a dash inside a fenced code block is code, not documentation."""
    errors = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        stripped = prose(text)
        if EM_DASH.search(stripped):
            for n, line in enumerate(stripped.splitlines(), 1):
                if EM_DASH.search(line):
                    errors.append(f"{f.relative_to(ROOT)}:{n}: em dash")
    return errors


def main() -> int:
    files = md_files()
    nav = nav_pages()
    errors = (
        check_links(files)
        + check_nav_targets(nav)
        + check_quadrants(files, nav)
        + check_reachable(nav)
        + check_truncated_cells(files)
        + check_em_dashes(files)
    )

    for e in errors:
        print(f"FAIL: {e}")

    if errors:
        print(f"\n{len(errors)} error(s)")
        return 1

    print(f"\nOK: {len(files)} files, {len(nav)} nav entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
