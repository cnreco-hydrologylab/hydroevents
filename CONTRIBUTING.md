# Contributing

## Releasing

Cutting a release does three things, in order: bump the version, publish a GitHub release, publish to PyPI. The first two are one click; the third is automatic.

1. **Bump the version.** Go to the **Actions** tab → **"Bump version"** → **Run workflow** → choose `patch`/`minor`/`major`. This:
   - updates `version` in `pyproject.toml` and `CITATION.cff` (and `CITATION.cff`'s `date-released`) together, via [bump-my-version](https://github.com/callowayproject/bump-my-version)
   - commits and tags (`vX.Y.Z`) on `main`
   - creates a **GitHub release** from that tag
2. **Zenodo archives it automatically.** The GitHub release triggers Zenodo's existing integration for this repo, which mints a new version-specific DOI under the same concept DOI shown in the README badge.
3. **PyPI publishing is automatic.** The release also triggers the `publish-pypi.yml` workflow, which builds the package and uploads it to PyPI via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) — no stored token needed.

You can also bump locally instead of step 1: `pip install bump-my-version && bump-my-version bump patch && git push && git push --tags`, then create the GitHub release by hand (`gh release create vX.Y.Z --generate-notes`) to trigger steps 2–3.

**Important — do not hand-edit the DOI in `CITATION.cff`.** It must always match the concept DOI in the README's Zenodo badge (`10.5281/zenodo.21650373`), which stays constant across all versions and always resolves to the latest release. It is not a per-version DOI.

A CI check (`version-consistency` job in `.github/workflows/tests.yml`) fails the build if `pyproject.toml`/`CITATION.cff` versions disagree, or if `CITATION.cff`'s DOI drifts from the README badge's.
