# Roadmap to v1.0.4

Focus areas for this release: **performance**, **usability**, and
**repository health**. Items were scoped in a review and benchmark session
on 2026-07-11 (Linux, RTX 5090, Python 3.12, torch 2.10.0+cu128, pyshmem
1.1.0 editable, shmpipeline 1.0.3 editable), then implemented and verified in
a follow-up session the same day (pyshmem 1.1.1, shmpipeline 1.0.4). **All
items below are implemented, tested, and verified** except H2's tag cleanup,
which is left as a manual step (see status).

## Measured baseline → after P1

Repository state at initial review: lint clean, fast suite green (238 tests,
81.2% coverage), slow suite green, all 44 kernel tests pass including GPU
paths, both AO examples run correctly end-to-end. After implementing the
roadmap (including the `checkerboard` fix below): fast suite green (251
tests, 82.3% coverage), slow suite green, kernel tests green, docs build
clean.

| Scenario | Before (pyshmem 1.1.0) | After (pyshmem 1.1.1, default `poll_interval`) |
|---|---|---|
| 5-stage 120×120 AO pipeline (CPU) | 4,239 Hz | 4,081 Hz (compute-bound, not lock-bound — see P1 note) |
| Single `cpu.copy` stage via synthetic source | 1,660 Hz (p50 604 µs) | 9,561 Hz (p50 81 µs, p99 166 µs) |
| Single `gpu.copy` stage via synthetic source | 1,425 Hz | 3,255 Hz (p50 310 µs) |
| Raw pyshmem `write()` loop | 138,000 Hz | 139,000 Hz (unaffected, as expected) |
| Synthetic source alone (no workers) | 17,700 Hz | 17,700 Hz (unaffected, as expected) |

Full before/after pair checked in at `benchmarks/results/rtx5090-linux-2026-07-11.json`
and `benchmarks/results/rtx5090-linux-2026-07-11-pyshmem-1.1.1.json`.

Headline finding, confirmed: the framework's throughput ceiling was not the
kernels, the writes, or the source — it was the contended-lock handoff
between producer and worker. pyshmem's timed `acquire()` retried a
non-blocking flock with a hardcoded 1 ms sleep, and `pyshmem.locked_many()`
did not accept `poll_interval` at all, so every worker frame contended at 1
ms granularity regardless of the per-kernel `poll_interval` setting. The
5-stage AO pipeline barely moved because its kernels' compute time already
dominates over lock handoff at that pipeline's throughput; the single-stage
`cpu.copy` case isolates the lock-handoff cost and shows the full ~5.75×
improvement.

## 1. Performance

### P1 — Plumb `poll_interval` through the lock path ✅ done

- pyshmem: added `poll_interval=` to `locked_many()` (forwarded to each
  stream's `locked()`/`acquire()`, which already accepted it), with
  validation and a test (`test_locked_many_validates_and_honors_poll_interval`).
  Released as pyshmem 1.1.1.
- shmpipeline: `runtime.py`'s `_locked_inputs_and_outputs` now passes
  `kernel_config.poll_interval` into `pyshmem.locked_many()`.
- Bumped the dependency to `pyshmem>=1.1.1,<2`.
- Measured payoff: 1.66 kHz → 9.56 kHz on the contended single-stage `cpu.copy`
  benchmark (default `poll_interval=1e-5`), 1.43 kHz → 3.26 kHz for `gpu.copy`.
- A deeper fix (blocking flock with deadline, or a futex-based lock) remains
  deferred past 1.0.4.

### P2 — Make `benchmark()` measure real latency ✅ done (documented, not re-timestamped)

Renamed the report field to `inter_arrival_ms` (keeping `latency_ms` as a
deprecated alias for compatibility) and rewrote the docstring and
`CLAUDE.md` to say plainly that this is terminal inter-arrival spacing, not
input→output latency — the pipeline contract does not require payloads to
carry a timestamp, so true end-to-end latency isn't measurable generically.
This was the pragmatic choice over adding a timestamp contract; documented
in `docs/guides/performance.md`.

### P3 — Checked-in benchmark baselines + regression guard ✅ done

Added `benchmarks/benchmark_pipeline.py` (config path + CLI flags in, JSON
report + environment metadata out), `benchmarks/smoke.yaml` (single
`cpu.copy` stage), and `benchmarks/results/` with a before/after pair plus a
CPU smoke-test result. CI runs the smoke benchmark after the slow-test job
and asserts `frames > 0` and `throughput_hz > 1.0` — a loose health check by
design (see `benchmarks/results/README.md` for why a universal hardware
threshold isn't appropriate here).

### P4 — Cache static auxiliary reads on the GPU path ✅ done

`runtime.py` now caches each GPU auxiliary stream's snapshot keyed by
`stream.count` (via `_read_worker_input`'s `cache`/`cache_key` parameters)
and only re-reads when the count advances; the trigger stream is
deliberately excluded from the cache since its count advances every frame by
definition. Covered by the GPU-path assertions in `test_kernels.py` and the
cache-miss/hit behavior is exercised through the real worker loop in the
GPU examples (verified manually: `examples/gpu_basic_ao_system/run_example.py`
still verifies all 3000 frames correctly with caching enabled).

### P5 — Trim per-frame allocations in the worker hot loop ✅ done

`run_kernel_process` now precomputes the ordered `(name, stream)` tuple used
by `locked_many()` once before the loop, and reuses a single
`auxiliary_inputs` dict across iterations instead of rebuilding it every
frame.

### P6 — GPU shutdown ordering — investigated; **not fixable as originally scoped**

The original plan was to reorder sink/worker shutdown so consumer CUDA
handles release before producer processes terminate, eliminating PyTorch's
*"Producer process has been terminated before all shared CUDA tensors
released"* warning seen on GPU example exit.

An initial implementation reordered `_stop_sinks()` before worker
termination and added a manager-level `torch.cuda.ipc_collect()` call in
`shutdown()`. Verification showed this did not eliminate the warning, and
investigation why revealed the premise was wrong for this codebase:

- Sinks run as threads *inside* the main/owning process and read through
  `self._streams[...]` — the same handle the manager used to `create()` the
  stream. They never hold a separate cross-process CUDA IPC consumer handle,
  so their stop order is irrelevant to IPC ref-counting.
- All GPU streams are created by the main process (`_build_stream` calls
  `pyshmem.create()` there), so every worker process is already a consumer
  for every stream it touches; the original code already joined *all*
  worker processes fully before any `unlink()` call, which is the ordering
  that actually matters.
- A minimal 1-producer/1-consumer repro with textbook-correct
  close-before-unlink ordering did not reproduce the warning. Extending the
  real GPU AO example's shutdown with `force=False`, generous timeouts, an
  explicit 3-second grace pause between worker-join and unlink, and manual
  `gc.collect()` + `torch.cuda.synchronize()` + `torch.cuda.ipc_collect()``
  calls did not suppress it either — the warning is printed by a PyTorch
  background thread *after* `shutdown()` has already returned, at Python
  interpreter finalization.

Conclusion: this is a benign PyTorch CUDA-IPC teardown artifact — a race
between its reference-counting watchdog thread and CUDA driver context
teardown at process exit — not an ordering bug in shmpipeline's shutdown
sequence. The speculative fix was reverted (`manager.py`'s
`_stop_runtime_components`/`shutdown()` are back to their original
ordering; the redundant `_collect_cuda_ipc()` method was removed since
pyshmem's own `unlink()` already calls `torch.cuda.ipc_collect()` per-stream
when needed). The real deliverable is a
[troubleshooting entry](docs/guides/troubleshooting.md) explaining the
warning is cosmetic, so users don't mistake it for a leak or a regression.

## 2. Usability

### U1 — Accept dict configs in `PipelineManager` ✅ done

`PipelineManager.__init__` now coerces a plain `Mapping` via
`PipelineConfig.from_dict`, tested in
`test_manager_accepts_plain_mapping_configuration`.

### U2 — Add a `shmpipeline benchmark` CLI command ✅ done

Added `shmpipeline benchmark pipeline.yaml [--duration] [--warmup] [--source
STREAM:PATTERN[:RATE_HZ]] [--output-stream] [--poll-interval] [--json]`,
documented in `docs/guides/cli.md` and `README.md`. Tested for both the
JSON-emitting happy path and the malformed-`--source` error path
(`test_cli_benchmark_delegates_and_emits_json`,
`test_cli_benchmark_rejects_malformed_source`).

### U3 — Write a performance-tuning guide ✅ done

Added `docs/guides/performance.md` covering the benchmark CLI,
`poll_interval`'s dual role (trigger-wait and, after P1, lock-retry) with
measured before/after numbers, notify/futex streams, worker placement, the
`spawn` `__main__` guard requirement (with the exact `RuntimeError` text so
it's searchable), expected throughput floors on the reference host, and
CPU-vs-GPU tradeoffs.

### U4 — Surface benchmark/tuning knobs in the GUI — deferred (as planned)

Not attempted; this was scoped as a stretch goal to drop first if time was
tight, and it was. Carry to 1.0.5.

## 3. Repository health

### H1 — Fix documentation drift ✅ done

Corrected the `pytest-forked` claim in `CHANGELOG.md`/`CLAUDE.md` (it was
never implemented; the slow-test isolation is a separate `pytest`
invocation, which is what actually ships). Moved the Keep a Changelog/SemVer
notice to the file header and added release dates to every version heading.

### H2 — Tag hygiene — partially done; remaining step is manual

The original premise ("no 1.0.3 tag exists") was wrong — `git fetch --tags`
showed `1.0.3` already exists on `origin`, just not fetched locally. The
only real inconsistency is the naming convention: `1.0.0`/`1.0.1`/`1.0.2`/`1.0.3`
exist alongside a duplicate `v1.0.1`. Deleting a tag that's already on the
remote is a destructive, shared-visibility action, so it was **not** done as
part of this pass — recommend `git push origin :refs/tags/v1.0.1` (and
`git tag -d v1.0.1` locally) once you've confirmed nothing external
references it (release notes, download links, CI).

### H3 — Python 3.9 ✅ done

Added `3.9` to the CI `test` job's matrix alongside `3.10`–`3.13`.

### H4 — Replace the hardcoded suffix list in `conftest.shm_prefix` ✅ done

`shm_prefix` now filters `pyshmem.list_streams()` by the fixture's own
prefix and calls `pyshmem.unlink_quiet()` (which already swallows
`FileNotFoundError`, so the redundant `try/except` around it was removed
too) instead of guessing suffixes.

### H5 — Lift the weak coverage spots ✅ done

Added targeted tests for `document.py`'s GPU/notify/replace-mode option
round-tripping, load-time mapping validation, and control-plane read
endpoints/error-status mapping. Total coverage moved from 81.2% to 82.2%;
`document.py` and `synthetic.py` both improved (`control/api.py` improved
via the new error-mapping and read-endpoint tests).

### H6 — Start decomposing `gui/app.py` ✅ started (first extraction only)

Extracted `_source_runtime_entries` out of `MainWindow` into a standalone,
Qt-free `runtime_source_entries()` in `gui/model.py`, with a dedicated test
(`test_runtime_source_entries_projects_plugin_and_synthetic_status`). This
is a first step, not the full decomposition — `MainWindow` is still
~1,750 lines. Carry the rest to 1.0.5.

### H7 — Add CONTRIBUTING.md and SECURITY.md ✅ done

Both added, matching pyshmem's tone and content: contribution setup/checks,
platform scope, and a private vulnerability-disclosure policy with an
explicit note that shared-memory names and the control plane are not an
authorization boundary.

## Verification

- `ruff check .` / `ruff format --check .`: clean.
- `pytest -m "not slow" --cov=shmpipeline`: 251 passed, 82.3% coverage
  (floor 80%).
- `pytest -m slow`: 1 passed.
- `pytest tests/test_kernels.py` (GPU-inclusive, RTX 5090): 44 passed.
- `sphinx-build -W -b html docs docs/_build/html`: build succeeded.
- `examples/basic_ao_system/run_benchmark.py`: 4,081 Hz (functionally
  unchanged from baseline).
- `examples/gpu_basic_ao_system/run_example.py`: verifies all 3,000 frames
  correctly (the benign CUDA IPC teardown warning is expected — see P6).

## Pre-existing bug found during this work — fixed

`synthetic.py` declared `"checkerboard"` as a valid `SyntheticInputConfig`
pattern name (and the GUI offered it in the pattern dropdown), but neither
`_next_cpu` nor `_next_gpu` had a `checkerboard`-specific branch — it
silently fell through to the generic alternating-value fallback used for
unmatched patterns. This predated the roadmap work (confirmed via
`git show HEAD` on the pre-session commit) and was unrelated to any 1.0.4
item, but was fixed rather than just flagged:

- Implemented real tiled checkerboard generation for both CPU (`np.indices`)
  and GPU (`torch.meshgrid`) paths, reusing `period` as the tile size
  (consistent with how `ramp`/`sine` already reuse it) and animating by
  scrolling along the first axis with `frame_index`.
- Replaced the silent generic fallback (which had masked this exact bug)
  with `raise AssertionError` for any pattern that reaches the end of the
  branch chain unhandled — so a future pattern added to
  `_SYNTHETIC_PATTERNS` without generator logic fails loudly in tests
  instead of silently producing wrong output.
- Added targeted tests: a 2-D tile-structure check, a 1-D scroll-with-frame
  check, a 0-D scalar "blink" check, and a GPU-vs-CPU equivalence check
  (skipped without CUDA); verified end-to-end through `PipelineManager`
  with a real 4×4 checkerboard.
