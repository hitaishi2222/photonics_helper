"""Shared provenance helpers.

Kept dependency-free so both the SQLite layer (`raman/db.py`) and the material
API (`materials.py`) can use it without importing each other.
"""

from __future__ import annotations

import re

__all__ = ["extract_doi"]

# A permissive DOI pattern: registrant prefix (10.<4-9 digits>) + suffix.
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\)\]\"'<>]+", re.IGNORECASE)


def extract_doi(text: str | None) -> str | None:
    """Return the first DOI found in ``text``, or ``None``.

    Citations in the database often embed ``https://doi.org/...`` links; this
    pulls out the identifier so consumers do not have to parse the string.
    Trailing sentence punctuation is stripped.
    """
    if not text:
        return None
    match = _DOI_RE.search(text)
    if match is None:
        return None
    return match.group(0).rstrip(".,;:")
