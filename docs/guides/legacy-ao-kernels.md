# Legacy adaptive-optics kernels

`shmpipeline` is a domain-neutral shared-memory compute framework. Its stable
built-in vocabulary consists of generic array operations: copy, flatten,
concatenate, arithmetic, reduction, affine transform, clipping/scaling,
custom operations, and generic stateful numerical integration.

The following 1.x kinds predate the plugin boundary and are now compatibility
only:

- `cpu.shack_hartmann_centroid` and `gpu.shack_hartmann_centroid`
- the `cpu.centroid` alias
- `cpu.tip_tilt_controller` and `gpu.tip_tilt_controller`
- `cpu.tomographic_controller` and `gpu.tomographic_controller`

Resolving one emits a `DeprecationWarning`. They remain behavior-compatible
for all of 1.x and may be removed only in 2.0. `shmpipeline` does not depend on
an AO package.

New AO systems should install `shmpipeline-ao`, select a supported AO mode,
and use its `ao.*` source/kernel/sink entry points. The canonical TOML-first
examples live in that repository under `examples/minimal_closed_loop`,
`examples/reference_10x10`, and `examples/keck_haka`. The old AO directories
in this repository are retained as 1.x compatibility and performance
fixtures, not as current user templates.
