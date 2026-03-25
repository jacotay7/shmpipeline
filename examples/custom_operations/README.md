# Custom Operations Example

This example shows how to fuse a simple image-calibration expression into a
single worker process using `cpu.custom_operation`.

The pipeline computes:

```text
(input - dark) / flat
```

where:

- `input` is the streamed wavefront-sensor image
- `dark` is a preloaded dark frame
- `flat` is a preloaded flat-field image

Run it with:

```bash
python examples/custom_operations/run_example.py
```

The expression language accepts a restricted subset of Python expressions:

- elementwise `+`, `-`, `*`, `/`
- unary `+` and `-`
- matrix multiplication with `@`
- intrinsic functions `abs(x)`, `minimum(a, b)`, `maximum(a, b)`, `min(a, b)`, `max(a, b)`, and `clip(x, low, high)`
- numeric constants
- variable names bound through `input` and `auxiliary`