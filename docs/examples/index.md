# Worked Examples

The repository includes examples that scale from minimal CPU transforms to larger adaptive-optics control chains.

```{toctree}
:maxdepth: 1

affine-transformation
gpu-affine-transformation
custom-operations
gpu-custom-operations
basic-ao-system
gpu-basic-ao-system
observatory-ao-system
```

## How to use the examples

Use the examples in three different ways:

- as smoke tests for installation and packaging
- as reference YAML for new pipelines
- as worked examples that show how stream definitions and kernels fit together

For runtime inspection, most example configs can be used directly with the CLI:

```bash
shmpipeline validate examples/affine_transformation/pipeline.yaml
shmpipeline describe examples/affine_transformation/pipeline.yaml
```
