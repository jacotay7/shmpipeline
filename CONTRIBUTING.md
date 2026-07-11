# Contributing

Bug reports and focused pull requests are welcome. Please include the
operating system, Python version, pyshmem version, and (for GPU behavior)
CUDA, driver, device, and PyTorch versions.

Set up a development environment with:

```bash
python -m pip install -e ".[test,docs]"
```

Before submitting a change, run:

```bash
ruff check .
ruff format --check .
pytest -m "not slow"
sphinx-build -W -b html docs docs/_build/html
python -m build
twine check dist/*
```

Run the slow integration suite separately, and run the full suite on a CUDA
host for GPU changes. New behavior should include regression tests and update
the README, documentation, and `CHANGELOG.md` when it changes a public
contract.

The project supports Linux and macOS. Windows support is intentionally out of
scope. Changes to shared-memory ownership or the pyshmem integration contract
should be reviewed alongside the corresponding pyshmem release.

By contributing, you agree that your contribution is licensed under the
repository's GPL-3.0-only license.
