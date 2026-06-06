# Troubleshooting

## Validation failures before startup

If `shmpipeline validate` fails, the most common causes are:

- duplicate shared-memory names
- duplicate kernel names
- kernels referencing undefined streams
- invalid storage combinations between streams and kernel kinds
- multiple kernels producing the same output stream

Use `shmpipeline describe pipeline.yaml` to confirm the intended graph shape.

## Worker stays in `waiting-input`

This means the worker process is alive but has not yet received its first trigger input.

Typical causes:

- no external producer is writing the source stream
- the wrong stream is being driven
- a synthetic input writer was not started for a source stream

## Worker becomes `idle`

This means the worker has processed frames before but has not made recent output progress.

Typical causes:

- upstream input has stalled
- the configured timeout is too aggressive for the workload
- a long-running stage is introducing backpressure

## GPU viewer or CPU reader cannot inspect a stream

If a GPU stream must be readable from CPU code, create it with `cpu_mirror: true`.

When attaching to that stream directly with `pyshmem.open(name)` from a process
that *also* has CUDA available, `pyshmem` auto-attaches to the GPU and `read()`
returns a `torch.Tensor` on the device — not a NumPy array. To read the host
mirror as a NumPy array regardless of local CUDA, open it CPU-only:

```python
import pyshmem

mirror = pyshmem.open(name, gpu_device=False)  # requires cpu_mirror: true
frame = mirror.read()                          # numpy.ndarray from the mirror
```

`gpu_device=False` raises `ValueError` if the stream was created without a CPU
mirror.

## GPU runtime requirements

GPU support depends on a compatible CUDA and PyTorch environment. When CUDA is unavailable, GPU-specific tests are skipped and GPU examples will not run.

## Repeated lifecycle testing

For release testing or stress testing, repeatedly exercise:

- `build()`
- `start()`
- `stop()`
- `shutdown()`

This catches stale-handle, teardown, and resource ownership issues earlier than single-run smoke tests.
