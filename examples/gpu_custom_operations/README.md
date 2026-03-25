# GPU Custom Operations Example

This example shows how to fuse a simple image-calibration expression into a
single GPU worker using `gpu.custom_operation`.

The pipeline computes:

```text
(input - dark) / flat
```

where all streams are GPU-backed shared-memory tensors.

Run it with:

```bash
python examples/gpu_custom_operations/run_example.py
```

The expression language is the same restricted expression subset used by the
CPU custom-operation kernel.