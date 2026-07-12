# Tomographic AO Stress Example: Implementation Plan

## 1. Goal

Build a reproducible, synthetic, three-loop adaptive-optics workload that
stresses `shmpipeline` rather than attempting to model detector or atmosphere
physics accurately. The example must:

- run from checked-in YAML on CPU or one CUDA device;
- drive eight hardware-synchronized 256 x 256 WFS camera streams with
  configurable arrival jitter;
- calibrate, centroid, slope-calibrate, synchronize, and combine all eight WFS
  paths before one large tomographic reconstruction;
- close the main 4096-actuator loop and apply final command calibration and
  actuator bounds before a fake hardware sink;
- run two independent 16 x 16 centroid/control/rotation/sink loops at different
  rates at the same time;
- measure sustainable rate, end-to-end latency, jitter, drops, synchronization
  behavior, per-stage timing, and CPU/GPU resource usage;
- expose missing general-purpose pipeline features through tested framework
  additions, not example-only timing hacks.

No scientifically meaningful image data, reconstructor, or command is needed.
All dynamic data and calibration arrays will be deterministic synthetic values.

## 2. Confirmed slope dimensions

The requested dimensions are not mutually consistent:

```text
256 pixels / 4 pixels per subaperture = 64 subapertures per axis
64 x 64 subapertures x 2 centroid coordinates = 8192 slopes per WFS
8192 x 8 WFSs = 65536 slopes
```

The full centroid grid is the intended workload. Slope offset/scaling calibrates
the values but does not remove any of them:

- centroid output per WFS: `(64, 64, 2)` = 8192 values;
- flatten output per WFS: `(8192,)`;
- flattened slopes per WFS: `(8192,)`;
- synchronized concatenation: `8 x 8192 = 65536`;
- reconstructor: `(4096, 65536) @ (65536,) -> (4096,)`.

At float32, the reconstructor is 1 GiB. The runner must report this allocation
before building the pipeline
and fail early with an actionable message when host RAM or CUDA memory is
insufficient.

## 3. Target graph

The main loop has eight parallel front ends followed by a synchronized fan-in:

```text
coordinated camera-set source (frame_id N, per-camera jitter)
  |-- wfs0_raw -> dark/flat -> tiled centroid -> slope offset/gain -> flatten --|
  |-- wfs1_raw -> dark/flat -> tiled centroid -> slope offset/gain -> flatten --|
  |-- wfs2_raw -> dark/flat -> tiled centroid -> slope offset/gain -> flatten --|
  |-- wfs3_raw -> dark/flat -> tiled centroid -> slope offset/gain -> flatten --| synchronized
  |-- wfs4_raw -> dark/flat -> tiled centroid -> slope offset/gain -> flatten --| concatenate
  |-- wfs5_raw -> dark/flat -> tiled centroid -> slope offset/gain -> flatten --| (65536)
  |-- wfs6_raw -> dark/flat -> tiled centroid -> slope offset/gain -> flatten --|
  `-- wfs7_raw -> dark/flat -> tiled centroid -> slope offset/gain -> flatten --|
       -> reconstructor matrix multiply (4096)
       -> leaky integrator / closed-loop state (4096)
       -> final command gain and offset (4096)
       -> per-actuator clip (4096)
       -> fake DM hardware sink
```

The two auxiliary loops are separate DAG branches and deliberately run at
different configured rates, for example 700 Hz and 233 Hz:

```text
16 x 16 source -> spot centroid (2) -> leaky integrator (2)
               -> affine rotation plus bias (2) -> fake tip/tilt sink
```

Running all branches in one `PipelineManager` verifies that unrelated trigger
rates remain independent and that the high-rate main branch does not cause a
low-rate branch to replay stale inputs.

## 4. Existing capabilities to reuse

Use existing built-ins wherever their contracts match:

| Requirement | Existing implementation | Plan |
|---|---|---|
| Dark/flat calibration | `cpu.custom_operation`, `gpu.custom_operation` | Use `(input - dark) * inverse_flat`; static arrays are auxiliary streams. |
| 4 x 4 tiled centroid | `*.shack_hartmann_centroid` | Use `tile_size: 4`, output `(64, 64, 2)`. |
| Slope gain/offset | `*.scale_offset` | Use scalar `gain` and vector offset; document its `gain * input - offset` sign convention. |
| Flatten | `*.flatten` | Flatten each calibrated centroid cube. |
| Matrix multiply plus bias | `*.affine_transform` | Use a `(4096, 65536)` matrix and zero/synthetic bias. |
| Closed-loop state | `*.leaky_integrator` | Configure `u[k] = leak*u[k-1] + gain*residual[k]`. |
| Final command calibration | `*.scale_offset` | Apply command gain and per-actuator offset. |
| Bounds | `*.custom_operation` | Use `clip(input, low, high)`. |
| 2-D rotation | `*.affine_transform` | Use a 2 x 2 rotation matrix and 2-vector bias. |
| CPU single-spot centroid | `cpu.spot_centroid` | Reuse for the CPU profile. |

The current runtime triggers a kernel only from `KernelConfig.input`.
Auxiliary streams are read at their latest values, but the worker does not wait
for them to advance. Therefore wiring seven WFS results as auxiliaries would
not satisfy the eight-new-frames requirement and must not be used as a fake
barrier.

## 5. Framework functionality to add

### 5.1 Synchronized multi-input trigger policy

Extend `KernelConfig` without breaking the existing single-trigger form:

```yaml
- name: synchronize_and_stack_slopes
  kind: cpu.concatenate
  inputs:
    - wfs0_selected_slopes
    - wfs1_selected_slopes
    - wfs2_selected_slopes
    - wfs3_selected_slopes
    - wfs4_selected_slopes
    - wfs5_selected_slopes
    - wfs6_selected_slopes
    - wfs7_selected_slopes
  output: tomo_slopes
  trigger_policy: all_new
  synchronization:
    mode: matching_frame_id
    timeout: 0.010
    on_skew: drop_older
  parameters:
    axis: 0
```

Compatibility rules:

- accept exactly one of `input` and `inputs`;
- normalize legacy `input` to a one-item trigger tuple internally;
- keep `auxiliary` for static or asynchronously updated calibration inputs;
- `any_new` preserves conventional one-stream triggering for a multi-input
  kernel when desired;
- `all_new` runs only after every trigger count exceeds its last consumed
  count;
- acquire all trigger, auxiliary, and output locks in the existing sorted
  order, then re-check all counts before computing;
- snapshot all trigger values in the same lock scope;
- advance the vector of consumed counts only after successful publication;
- expose wait time, skew, skipped generations, and timeout counts in worker
  metrics.

`all_new` alone guarantees freshness, not hardware frame identity. Publication
counts also cannot serve as frame IDs after multiple capacity-one stages: if a
worker skips three trigger publications and computes only the latest value, its
output count advances by one, not three. Equal counts at the ends of the eight
branches can therefore still describe different camera frames.

Add a runtime-managed uint64 `frame_id` token for dynamic streams. A coordinated
source assigns the same token to all eight images from hardware trigger N. Each
single-input kernel copies its trigger token to every output as part of the same
locked publication transaction. The fan-in reads values and tokens under one
lock set, runs only when all tokens match, and propagates that token downstream.
Implementation options are hidden companion pyshmem scalar streams or a small
extension to pyshmem publication metadata; choose after a prototype measures
the overhead. Companion streams keep the change local to `shmpipeline`, but
must participate in lock ordering so a value and token cannot be torn.

`matching_frame_id` plus `drop_older` waits for lagging branches and records
overwritten/advanced tokens instead of combining different IDs. Add
`max_skew_generations` and `max_wait_s` safeguards so a failed camera cannot
stall forever. Counts remain the mechanism for detecting new publications and
missed writes; tokens establish cross-branch identity. Real camera source
plugins must provide the hardware sequence number (or have the coordinated
source assign one) rather than inferring identity from arrival time.

### 5.2 Concatenate kernel

Add CPU and GPU built-ins:

- `cpu.concatenate` / `gpu.concatenate`: variable trigger arity, validates
  dtype and non-concatenated dimensions, writes directly to the output, and
  supports `axis` (the example uses 1-D axis 0);
The concatenate kernel is the first consumer of the new multi-input API. Keep
synchronization in the runtime rather than burying it in this kernel so any
future fusion, averaging, or voting kernel can use the same barrier semantics.

### 5.3 Coordinated multi-output synthetic source

Eight independent `SyntheticSourceController` threads do not model a common
hardware trigger and cannot intentionally inject controlled inter-camera
jitter. Add a coordinated source abstraction that owns multiple output streams:

```yaml
sources:
  - name: synchronized_wfs_cameras
    kind: synthetic.frame_set
    streams: [wfs0_raw, wfs1_raw, wfs2_raw, wfs3_raw,
              wfs4_raw, wfs5_raw, wfs6_raw, wfs7_raw]
    parameters:
      rate_hz: null          # unthrottled for maximum-rate mode
      pattern: random
      seed: 2401
      jitter_us: 25
      drop_probability: 0.0
```

This requires `SourceConfig` to support `stream` or `streams`, analogous to
kernel `output`/`outputs`. One controller generates all eight reusable buffers
for frame N, applies deterministic per-camera jitter, and publishes each stream
once. The source reports per-camera writes, generation skew, requested/effective
rate, missed deadlines, injected drops, and write duration. A `drop_probability`
fault profile is useful for proving barrier timeout and recovery behavior, but
must be disabled in maximum-throughput results.

For CPU, generate or mutate buffers in place. For GPU, generate directly on the
configured CUDA device and avoid a host-to-device copy in every frame. The
source must not allocate a fresh 256 x 256 array per publication.

### 5.4 GPU single-spot centroid

Add `gpu.spot_centroid` with the same shape, output ordering, threshold,
background, and `weight_power` contract as `cpu.spot_centroid`. Pre-create its
coordinate vectors and temporary storage, write into the provided output, and
test CPU/GPU numerical agreement. This lets both auxiliary loops remain fully
on GPU in the GPU profile.

### 5.5 Built-in synthetic/null endpoints

The example needs runnable YAML without requiring users to build and install an
example-local entry-point package. Register general-purpose built-in endpoint
kinds:

- `synthetic.frame_set` for the synchronized cameras;
- `synthetic.array` for each 16 x 16 loop source;
- `null.sink` for the DM and two tip/tilt hardware endpoints.

`null.sink` must consume every observed publication, optionally add a
configurable fake device delay, and report consumed count, missed writes,
effective rate, and consume-time percentiles. It should not print per frame.
This is useful outside this example and avoids measuring terminal polling in
place of the requested hardware boundary.

## 6. Configuration and example files

The completed directory should contain:

```text
examples/tomographic_ao_stress/
  README.md
  IMPLEMENTATION_PLAN.md
  pipeline_cpu.yaml
  pipeline_gpu.yaml
  run_benchmark.py
  generate_calibrations.py
  expected_dimensions.json
  results/README.md
```

Check in both explicit YAML files so `shmpipeline validate`, the GUI, and the
CLI can inspect either graph without preprocessing. Keep their topology and
names identical; change only `storage`, `gpu_device`, and kernel kind prefixes.
Static values are loaded by `run_benchmark.py` after `build()` and before
`start()` (or before sources are released), including:

- eight dark frames and inverse-flat frames, each `(256, 256)`;
- eight slope offset cubes, each `(64, 64, 2)`;
- reconstructor `(4096, 65536)` and bias `(4096,)`;
- final DM offset, low limit, and high limit `(4096,)`;
- two 2 x 2 rotation matrices and two 2-vectors of affine bias.

Use seeded generation and do not check the 1 GiB matrix into git.
`generate_calibrations.py` may optionally materialize memory-mappable CPU
calibrations for repeated runs, but the default runner can generate them once
in memory. Initialization is outside the measured interval.

The GPU YAML should set `cpu_mirror: false` on hot-path streams unless the fake
sink explicitly requires a host view. A GPU-aware null sink should consume on
device so the main benchmark does not accidentally include a 4096-element
device-to-host copy. Provide an opt-in `--cpu-mirror-sink` mode to characterize
the realistic host/device boundary separately.

## 7. Benchmark runner

`run_benchmark.py` is the authoritative stress harness because the current
`PipelineManager.benchmark()` supports only one optional synthetic source and
one output stream. Its CLI should include:

```text
--backend {cpu,gpu}
--duration SECONDS
--warmup SECONDS
--main-rate HZ|unlimited
--loop-a-rate HZ
--loop-b-rate HZ
--camera-jitter-us VALUE
--fault-drop-probability VALUE
--spawn-method {spawn,fork,forkserver}
--cpu-affinity auto|none|LIST
--json-out PATH
--verify-frames N
```

Run phases:

1. Load and validate the selected YAML and registered kinds.
2. Calculate stream bytes, reconstructor bytes, expected worker count, host
   memory headroom, and CUDA memory headroom; print the plan.
3. Build streams and load deterministic calibration data.
4. Start workers and sinks, then release all three sources together.
5. Warm up Numba, BLAS, CUDA contexts, and expression plans. Reset benchmark
   counters after warm-up without resetting integrator state unless requested.
6. Optionally verify a small number of frames for shapes, finite values,
   bounds, monotonic counts, exact WFS generations, and independent auxiliary
   loop rates. Do not perform full NumPy reference reconstruction during the
   timed maximum-rate phase.
7. Measure for the requested duration.
8. Quiesce sources, drain outputs, collect metrics, and shut down even after a
   failure.
9. Emit a versioned JSON report with config hash, git revision, package
   versions, platform, CPU model, affinity, CUDA/GPU details, stream sizes, and
   all measurements.

Support two main benchmark modes:

- **closed-loop latency mode:** publish the next eight-camera set only after
  its DM command reaches the sink. This measures unambiguous end-to-end latency
  and maximum non-pipelined control frequency;
- **saturation throughput mode:** publish unthrottled and allow the
  capacity-one streams to apply latest-value backpressure. This measures
  sustainable output rate and drop behavior under overload.

Both are necessary: output inter-arrival time is not end-to-end latency, and a
capacity-one pipeline can report a high terminal rate while dropping input
generations.

## 8. Metrics and frame tracing

Extend benchmark/runtime reporting with:

- source publications and effective rate per camera/loop;
- terminal sink publications and effective rate for all three loops;
- input-to-sink latency p50/p90/p99/p99.9/max for each loop;
- terminal inter-arrival jitter for each loop;
- per-worker compute time, wait time, lock time, throughput, and processed
  count;
- fan-in generation skew, barrier wait, skipped generations, timeouts, and
  mismatched-generation prevention count;
- pyshmem missed-write counters on every edge;
- deadline misses relative to each configured loop period;
- CPU utilization/RSS per worker and overall;
- CUDA allocated/reserved/peak memory plus device utilization when available;
- matrix-multiply GFLOP/s (134,217,728 matrix elements, or about 0.268 GFLOP
  when one multiply and one add are counted as two operations per main-loop
  frame).

True latency requires frame identity and a source timestamp. Avoid embedding
these in the float image. Key a small benchmark trace table by the propagated
`frame_id`; record its monotonic source timestamp before the camera-set
publications and resolve it when the DM sink consumes the same token. The
association between value and token must be made under the same publication
lock and validated at the fan-in. Timestamp tracing can be optional
instrumentation, but frame-token propagation is required whenever
`matching_frame_id` is configured.

## 9. CPU and GPU performance considerations

### CPU

- The main matrix-vector multiply will dominate. Start with the existing
  `blas_threads: 1` behavior, then benchmark explicit thread counts without
  oversubscribing the many process-per-kernel workers.
- Pin the reconstructor to a physical core or core set near the NUMA node that
  owns its 1 GiB matrix. Record NUMA placement.
- Reserve cores for the manager, three source controllers, and sink threads.
- Consider a fused calibration-plus-centroid kernel only after the unfused
  graph establishes per-handoff cost; the unfused graph is valuable as a
  framework stress test.

### GPU

- A 1 GiB static reconstructor can be read by CUDA IPC, but the current GPU
  worker safe-read/cache path may create a private device snapshot on first
  use. Account for both the shared tensor and cached copy before claiming the
  workload fits.
- Each process owns a CUDA context and current GPU kernels synchronize the
  device on each stage. Report context/memory overhead and synchronization time;
  these may make many tiny GPU stages slower than CPU stages.
- First benchmark the topology as designed. Then add an optional fused WFS
  front-end kernel and, separately, a fused post-reconstructor controller
  kernel to quantify the cost of process and CUDA synchronization boundaries.
- Do not include CUDA initialization or first allocation in steady-state
  results.

## 10. Validation and tests

Add focused tests before relying on benchmark numbers:

1. Config parsing accepts `input` or `inputs`, rejects both/neither, and
   serializes round-trip without losing trigger policy.
2. `all_new` does not run when only seven of eight streams advance.
3. One publication on the eighth stream releases exactly one computation.
4. A fast input advancing multiple times never causes stale replay; skipped
   counts are reported.
5. `matching_frame_id` never combines unequal propagated tokens and follows
   timeout/skew policy deterministically, even when branch workers skip
   different trigger publications.
6. Lock acquisition plus count re-check closes the arrival race between wait
   and snapshot.
7. Worker failure does not commit consumed counts or a partial output.
8. CPU/GPU concatenate validates shapes/dtypes and matches NumPy/Torch.
9. CPU/GPU spot centroids agree for normal, zero-flux, thresholded, and
   background-subtracted images.
10. Coordinated source writes equal generations, injects deterministic jitter,
    and recovers/stops cleanly after an injected camera drop.
11. Null sink reports consumed and missed counts without per-frame allocation.
12. A reduced-size three-loop integration test runs in normal CI.
13. The full CPU config validates in CI; GPU config validation runs without
    requiring a live device where possible, while the full timed workload is
    marked hardware/slow.
14. Benchmark JSON schema and environment metadata are stable and tested.

During the first few verification frames, assert the final DM command is
finite and within every actuator limit, and assert both rotated 2-vectors match
a direct reference calculation. This catches wiring/sign/order mistakes without
turning the performance phase into a correctness benchmark.

## 11. Implementation order

1. Add and test multi-input config/graph representation.
2. Implement and test frame-token propagation, runtime `all_new`, and
   matching-frame barrier metrics.
3. Add CPU/GPU concatenate kernels.
4. Add GPU spot centroid.
5. Add coordinated multi-output sources and null sinks.
6. Check in CPU/GPU YAML and calibration generation.
7. Build the multi-terminal benchmark/trace harness and JSON report.
8. Add reduced integration tests and documentation.
9. Establish CPU and GPU baselines, then profile bottlenecks.
10. Add optional fused variants only when baseline measurements show their
    value; retain the unfused workload as the maximal orchestration stress
    profile.

## 12. Completion criteria

The example is complete when:

- both YAML files pass `shmpipeline validate` and are viewable in the GUI;
- one command runs either CPU or GPU with fake inputs and all three fake
  hardware sinks;
- the reconstructor consumes only synchronized sets in which every WFS has a
  new matching generation;
- the main tomographic vector is exactly 65536 with all 8192 slopes from each
  WFS represented;
- both auxiliary loops sustain their independent configured rates while the
  main loop is loaded;
- reports distinguish end-to-end latency, terminal spacing, throughput, and
  dropped/skipped frames;
- CPU/GPU outputs pass reduced correctness checks;
- an overload/fault run demonstrates measured jitter, backpressure, and barrier
  recovery rather than hanging silently;
- benchmark results include enough environment and configuration metadata to
  reproduce and compare runs.
