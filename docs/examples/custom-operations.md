# Custom Operations

This example shows how to fuse a small arithmetic expression into one CPU worker using `cpu.custom_operation`.

## Expression

```text
(input - dark) / flat
```

## Files

- `examples/custom_operations/pipeline.yaml`
- `examples/custom_operations/run_example.py`

## Run it

```bash
python examples/custom_operations/run_example.py
```

## What it demonstrates

- `custom_operation` kernels for small fused workflows
- named auxiliary streams for expression bindings
- a safe restricted expression language
- a compact alternative to chaining several simple arithmetic kernels

## Supported expression building blocks

- elementwise `+`, `-`, `*`, `/`
- unary `+` and `-`
- matrix multiplication with `@`
- `abs`, `minimum`, `maximum`, `min`, `max`, and `clip`
- numeric constants

## When to use it

Use a custom-operation kernel when the computation is simple, local, and easier to reason about as one expression than as a longer multi-stage graph.
