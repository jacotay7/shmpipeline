# Benchmark results

Results are JSON snapshots produced by `benchmark_pipeline.py`. Keep the
machine, Python, pyshmem, and shmpipeline versions in each file so numbers are
comparable. The CPU smoke benchmark is intentionally a loose health check;
hardware-specific throughput belongs in a dated result file rather than a
universal CI threshold.

Example:

```bash
python benchmarks/benchmark_pipeline.py benchmarks/smoke.yaml \
  --duration 5 --warmup 1 --source benchmark_input:random:1000 \
  --json-out benchmarks/results/linux-cpu-$(date +%F).json
```

`rtx5090-linux-2026-07-11.json` and
`rtx5090-linux-2026-07-11-pyshmem-1.1.1.json` are a before/after pair for the
lock-`poll_interval` fix (see `docs/guides/performance.md`): same host, same
day, `pyshmem`/`shmpipeline` versions bumped between them. Use that pair as
the reference for how much a lock-contention fix should move the numbers
before adding a new comparison point.
