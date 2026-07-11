# Performance tuning

Start with the checked-in benchmark rather than tuning a kernel in isolation:

```bash
shmpipeline benchmark pipeline.yaml --duration 5 \
  --source input_frame:random:1000 --json
```

The report's `inter_arrival_ms` (and compatibility alias `latency_ms`) is the
time between terminal publications. It is not an input-to-output timestamp;
the pipeline contract does not require timestamps in payloads. Use an
application timestamp in the frame when end-to-end latency is required.

## Lock and wait polling

`poll_interval` has two related roles in a worker's `KernelConfig`, and both
are forwarded into pyshmem's `locked_many()` for that kernel's lock scope:

- it controls how often the worker checks for a new trigger generation;
- it controls retries while acquiring the multi-stream write locks.

The default (`1e-5`, 10 microseconds) is already tuned for low-latency,
highly contended pipelines. Earlier releases fixed the lock-retry interval
at 1 millisecond regardless of this setting, which capped a single
contended stage at roughly 1.6 kHz; with `poll_interval` now honored end to
end, the same stage reaches roughly 9.5 kHz (see Expected floors below). A
longer interval trades handoff latency for lower CPU usage on lightly
loaded pipelines; there is normally no reason to raise it above the default
for a latency-sensitive pipeline.

Streams created with `notify: true` use pyshmem's Linux futex notification path
for parked readers. Notifications avoid idle polling, but still incur one
wakeup syscall per write and fall back to polling on macOS.

## Worker placement

Use `WorkerPlacementPolicy` to pin workers when the pipeline competes with a
producer or other real-time work. Keep a producer and its first worker on
nearby cores, and leave at least one core for the manager and operating system.
Pinning is a placement tool, not a substitute for measuring lock contention.

## Spawn entry points

The default start method is `spawn`. Programs that construct a manager at
module import time must protect their entry point:

```python
if __name__ == "__main__":
    main()
```

Without that guard Python raises:

> An attempt has been made to start a new process before the current process
> has finished its bootstrapping phase.

## Expected floors

On the release-review Linux RTX 5090 host (Python 3.12, torch 2.10), a raw
pyshmem 64-element write was about 7 microseconds. With the default
`poll_interval` (`1e-5`), a synthetic single-stage `cpu.copy` pipeline
measured roughly 9.5 kHz (p50 81 microseconds, p99 166 microseconds), and a
single-stage `gpu.copy` pipeline measured roughly 3.3 kHz (p50 310
microseconds). A five-stage 120×120 adaptive-optics pipeline — where kernel
compute time rather than lock handoff dominates — reached about 4.1 kHz.
These are orientation points, not acceptance thresholds; run the benchmark
on the target CPU, CUDA driver, and process layout.

## CPU versus GPU

GPU storage removes host copies for sufficiently expensive kernels, but each
stage still pays process synchronization, CUDA IPC, and snapshot overhead. A
small operation with a roughly 10 microsecond CPU floor can be slower on the
GPU path. Prefer CPU streams for light arithmetic and reserve GPU stages for
work large enough to amortize that fixed cost.
