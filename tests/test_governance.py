"""Governance artefacts: stability policy, contribution guide, citation, paper.

These are part of the stability contract — a foundation project must publish
what it guarantees and how to cite/contribute to it — so their presence and
required content are checked like any other contract.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MKDOCS = ROOT / "mkdocs.yml"


# ─── Stability policy ─────────────────────────────────────────────────────


def test_stability_policy_exists_and_covers_the_contract() -> None:
    text = (DOCS / "stability.md").read_text()
    for heading in (
        "## The stable surface",
        "## Versioning",
        "## Deprecation lifecycle",
        "## Support window",
        "## Database schema",
    ):
        assert heading in text, f"stability policy is missing {heading!r}"
    assert "Provisional" in text and "Stable" in text


def test_docs_nav_links_the_contract_pages() -> None:
    nav = MKDOCS.read_text()
    for page in (
        "core.md",
        "stability.md",
        "building-on-core.md",
        "data-provenance.md",
        "data-schema.md",
    ):
        assert page in nav, f"{page} is not in the docs navigation"


def test_building_on_core_guide_exists() -> None:
    text = (DOCS / "building-on-core.md").read_text()
    assert "OpticalMaterial" in text
    assert "python examples/33_build_on_core.py" in text


# ─── Contribution guide ───────────────────────────────────────────────────


def test_contributing_guide_covers_setup_and_gates() -> None:
    text = (ROOT / "CONTRIBUTING.md").read_text()
    for required in ("ruff check", "mypy photonics_helper", "pytest", "update_api_snapshot.py"):
        assert required in text, f"CONTRIBUTING.md does not mention {required!r}"
    assert "seed_db.py" in text  # data regeneration procedure


# ─── Citation metadata ────────────────────────────────────────────────────


def test_citation_cff_parses_with_required_fields() -> None:
    data = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert data["cff-version"]
    assert data["title"]
    assert data["type"] == "software"
    assert data["repository-code"].startswith("https://github.com/")
    assert data["version"]
    assert data["authors"], "CITATION.cff has no authors"
    assert data["authors"][0].get("given-names") and data["authors"][0].get("family-names")


def test_citation_version_matches_package_version() -> None:
    import tomllib

    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert citation["version"] == pyproject["project"]["version"]


# ─── Software paper ───────────────────────────────────────────────────────


def test_paper_draft_exists_with_statement_of_need() -> None:
    paper = (ROOT / "paper" / "paper.md").read_text()
    assert "# Summary" in paper
    assert "# Statement of need" in paper
    assert "# References" in paper
    assert (ROOT / "paper" / "paper.bib").is_file()


def test_paper_front_matter_references_the_bibliography() -> None:
    paper = (ROOT / "paper" / "paper.md").read_text()
    front_matter = yaml.safe_load(paper.split("---")[1])
    assert front_matter["bibliography"] == "paper.bib"
    assert front_matter["title"]
    assert front_matter["authors"]
