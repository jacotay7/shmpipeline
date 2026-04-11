# Build Plan

## Purpose

This document now tracks the work required to take the current repository to a
public `1.0.0` release. Earlier foundation and operability milestones are
already shipped. The remaining work is release polish, packaging verification,
and final QA.

## Current Repository State

The repository already provides:

- an installable Python package with CLI and GUI entry points
- validated YAML pipeline configuration with graph introspection before startup
- process-supervised CPU and GPU kernel families built on `pyshmem`
- runtime snapshots, worker health metrics, synthetic inputs, and stream
  viewers
- example pipelines ranging from simple affine transforms to observatory-scale
  adaptive optics flows
- automated tests across config loading, graph validation, kernels, manager
  behavior, CLI flows, GUI flows, and examples
- GitHub Actions CI across Linux, macOS, and Windows, plus a PyPI publish
  workflow

The remaining work is no longer core implementation or package metadata.
Release-facing documentation, packaging metadata, distribution validation, and
final automated QA are now complete. What remains is the release execution
sequence itself and any optional external-service follow-up.

## 1.0.0 Release Goal

Release `shmpipeline` as a stable user-facing package for local shared-memory
CPU and GPU pipelines, with clear installation guidance, verified
distributions, and documented release steps.

## 1.0.0 Exit Criteria

The `1.0.0` release is ready when all of the following are true:

- package metadata and the public package version are aligned on `1.0.0`
- the README and changelog describe the current capabilities without
  scaffold-only or alpha positioning
- CI builds both the source distribution and wheel and validates the generated
  package metadata before publish
- the core user flows are smoke-tested: install, `shmpipeline validate`,
  `shmpipeline describe`, `shmpipeline run`, GUI startup, and example configs
- release notes and the manual release checklist are explicit enough that the
  publish step is routine rather than ad hoc

## Already Shipped

These are no longer release blockers:

- pipeline build/start/pause/resume/stop/shutdown lifecycle management
- CPU and GPU built-in kernel parity for the current kernel family
- graph validation and description workflows for CLI and GUI consumers
- runtime snapshots, rolling worker metrics, and structured failure surfacing
- synthetic input generation for testing, demos, and live GUI use
- desktop GUI editing, validation, and shared-memory viewers
- programmatic third-party kernel extension through the registry
- multi-platform CI and a trusted-publisher-style PyPI workflow

## Remaining Workstreams

### 1. Packaging And Version Metadata

Status: Completed

Scope:

- bump `pyproject.toml` and `src/shmpipeline/__init__.py` to `1.0.0`
- replace the alpha classifier with stable release metadata
- add project URLs and other PyPI-facing metadata that users expect
- keep optional dependency guidance explicit for base, GPU, GUI, and test
  installs

Why this matters:

- this is the minimum metadata work required for a credible `1.0.0` release
- the package page needs to reflect the actual maturity of the repository

### 2. Release-Facing Documentation

Status: Completed

Scope:

- rewrite README sections that still describe the project as a scaffold or
  development-only install
- add a changelog for the `1.0.0` release
- keep extension guidance explicit: registry-based programmatic extension is
  supported now, automatic plugin discovery is not part of `1.0.0`
- document user installs separately from editable source installs

Why this matters:

- users should be able to understand how to install and use the package without
  reverse-engineering the repo layout
- a stable release needs a clear record of what is in scope today

### 3. Distribution Validation

Status: Completed

Scope:

- build the source distribution and wheel in CI
- run `twine check` against generated artifacts
- smoke-test the built wheel in a clean environment and verify the CLI entry
  point starts
- keep the publish workflow aligned with the same build assumptions as CI

Why this matters:

- it prevents packaging regressions from being discovered only at release time
- it verifies the install path users will actually consume

### 4. Final QA Before Publish

Status: Completed

Scope:

- run the full automated test suite on the current release branch state
- smoke-test the CLI against checked-in example configs
- verify GUI behavior in a supported desktop environment
- keep GPU validation explicitly environment-gated when CUDA is unavailable

Completed checks:

- full pytest suite passed locally
- source distribution and wheel built successfully
- package metadata passed `twine check`
- strict Sphinx build passed in a docs-enabled environment
- CLI smoke tests passed against checked-in example configs
- GUI screenshots were captured from the live application and wired into the
  docs

### 5. Release Execution

Status: Pending

Scope:

- cut the GitHub release notes from the changelog
- tag `v1.0.0`
- publish to PyPI through the existing release workflow
- verify install and entry points from the published package

Why this matters:

- publishing should be scripted and repeatable
- the post-publish verification closes the loop on the user installation path

## Remaining Checklist Before Publishing 1.0.0

1. Create the changelog-backed GitHub release notes.
2. Tag and publish `v1.0.0`.
3. Verify installation from PyPI after publish.
4. If you want hosted docs on day one, connect the repository to Read the Docs
  and confirm the first hosted build succeeds.

## Post-1.0 Backlog

These remain good follow-on items after `1.0.0` ships:

- richer pipeline graph visualization
- historical metrics export or external telemetry integration
- automatic plugin discovery for third-party kernels
- more advanced scheduling policies beyond the current local placement surface
- remote or multi-host orchestration
- extra GUI polish such as deeper macOS-native theming work
