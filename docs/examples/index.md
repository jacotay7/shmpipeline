# Worked Examples

The repository examples cover the domain-neutral runtime and plugin surface.
Historical AO directories remain as 1.x compatibility fixtures; current AO
examples and documentation use `shmpipeline-ao` entry-point kinds.

```{toctree}
:maxdepth: 1

affine-transformation
gpu-affine-transformation
custom-operations
gpu-custom-operations
source-sink-plugins
```

## How to use the examples

Use the examples in three different ways:

- as smoke tests for installation and packaging
- as reference YAML for new pipelines
- as worked examples that show how stream definitions and kernels fit together

For adaptive optics, install `shmpipeline-ao` and follow its TOML-first
quickstart. Do not start a new system from the legacy AO YAML in this
repository.

For runtime inspection, most example configs can be used directly with the CLI:

```bash
shmpipeline validate examples/affine_transformation/pipeline.yaml
shmpipeline describe examples/affine_transformation/pipeline.yaml
```
