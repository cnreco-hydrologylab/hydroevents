#!/usr/bin/env python3
"""Fail if pyproject.toml and CITATION.cff report different versions."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def extract(path, pattern):
    text = path.read_text()
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        sys.exit(f"Could not find version in {path}")
    return match.group(1)


def main():
    pyproject_version = extract(ROOT / "pyproject.toml", r'^version = "([^"]+)"')
    citation_version = extract(ROOT / "CITATION.cff", r'^version: "([^"]+)"')

    if pyproject_version != citation_version:
        sys.exit(
            f"Version mismatch: pyproject.toml has {pyproject_version!r}, "
            f"CITATION.cff has {citation_version!r}. Use `bump-my-version` "
            "to bump both together."
        )

    # The README's Zenodo badge links to the concept DOI (stable across all
    # releases, resolves to whichever version is latest). CITATION.cff must
    # reference that same concept DOI, not a version-specific one -- it
    # should never need to change when bumping versions.
    badge_doi = extract(ROOT / "README.md", r"doi\.org/(10\.5281/zenodo\.\d+)")
    citation_doi = extract(ROOT / "CITATION.cff", r'value:\s*(10\.5281/zenodo\.\d+)')

    if badge_doi != citation_doi:
        sys.exit(
            f"DOI mismatch: README badge points to {badge_doi!r}, "
            f"CITATION.cff has {citation_doi!r}. CITATION.cff should "
            "reference the stable Zenodo concept DOI from the README badge "
            "-- do not edit it per release."
        )

    print(f"OK: pyproject.toml and CITATION.cff agree on version {pyproject_version}")
    print(f"OK: CITATION.cff DOI matches the README's concept DOI ({citation_doi})")


if __name__ == "__main__":
    main()
