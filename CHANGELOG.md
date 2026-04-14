# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic
Versioning.

## [1.0.0] - Pending

This will be the first stable public release of `shmpipeline`.

### Added

- Validated YAML pipeline configuration for CPU and GPU shared-memory streams.
- Process-supervised pipeline execution with explicit build, start, pause,
  resume, stop, and shutdown lifecycle control.
- Built-in CPU and GPU kernel families, including arithmetic, affine,
  centroiding, flattening, and leaky-integrator operations.
- CLI workflows for `validate`, `describe`, and `run`.
- Pipeline graph introspection and validation before worker startup.
- Runtime snapshots, rolling worker metrics, synthetic inputs, and shared-memory
  viewers.
- Desktop GUI for editing, validating, running, and inspecting pipelines.
- Optional HTTP control plane for remote lifecycle commands, JSON snapshots,
  and SSE event streaming.
- Example pipelines ranging from simple affine transforms to observatory-scale
  adaptive optics flows.
- Multi-platform CI and a PyPI publish workflow.

### Changed

- Release metadata now targets the stable `1.0.0` package release.
- The README now leads with user-facing installation guidance instead of
  editable-only development commands.
- CI now validates source and wheel distributions before release publishing.

### Notes

- GPU support remains optional and depends on a compatible PyTorch and CUDA
  environment.
- Third-party kernel extension is supported programmatically through the
  registry; automatic plugin discovery remains post-`1.0.0` work.
