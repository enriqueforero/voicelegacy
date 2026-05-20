# Release and PyPI checklist

This repository should go to PyPI only after GitHub use is stable.

## Before tagging

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
python notebooks/build_notebook.py
python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"
python -m pip install build twine
python -m build
python -m twine check dist/*
```

## Versioning

1. Update `pyproject.toml` version.
2. Update `voicelegacy/__init__.py` if it carries the package version.
3. Update `CHANGELOG.md`.
4. Commit.
5. Tag using semantic versioning, for example `v0.1.0`.

## PyPI policy

The release workflow builds artifacts on tags but does not publish to PyPI automatically. Publishing requires manual `workflow_dispatch` with `publish_pypi=true` and a configured PyPI Trusted Publisher.

Do not publish to PyPI while the README, ethics document, CPML notice, and release notes are stale.
