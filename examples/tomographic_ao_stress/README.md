# Tomographic AO Stress Example

A reproducible synthetic three-loop adaptive-optics workload that stresses
`shmpipeline`: eight hardware-synchronized 256×256 Shack-Hartmann WFS cameras
feed one fused tomographic controller, alongside two independent 16×16 tip/tilt
loops running at their own rates.

Each WFS uses 4×4 pixel tiles → `(64, 64, 2)` centroids = 8192 slopes; all eight
give a 65536-element tomographic vector reconstructed to 4096 actuators. The
float32 `(4096, 65536)` reconstructor is exactly 1 GiB.

There is **one CPU and one GPU pipeline**, identical in topology:

```text
synthetic.frame_set (8 WFS, one frame_id token per generation)
  └─ tomographic_controller (8 inputs, synchronization: matching_frame_id)
        fuses dark/flat · tiled centroids · slope cal · 4096×65536 reconstruct
        · leaky integrator · command cal · per-actuator clip
     └─ tomo_dm_command → null.sink   (fake DM)
synthetic.array → tt_a_controller → null.sink   (700 Hz tip/tilt)
synthetic.array → tt_b_controller → null.sink   (233 Hz tip/tilt)
```

The fan-in only combines WFS frames carrying a **matching `frame_id` token**, so
the reconstructor consumes synchronized sets even under load; skewed branches are
dropped (`drop_older`) within `max_skew_generations` / `max_wait_s`.

## Run it

Validate or run either pipeline directly — they are self-contained (built-in
`synthetic.frame_set` / `synthetic.array` sources and `null.sink` endpoints):

```bash
shmpipeline validate examples/tomographic_ao_stress/pipeline_gpu.yaml
shmpipeline run examples/tomographic_ao_stress/pipeline_cpu.yaml --duration 5
```

## Benchmark

`run_benchmark.py` substitutes an instrumented camera source (rate control +
frame-id timestamps), loads deterministic calibrations, and reports throughput,
delivery ratio, per-loop rates, `matching_frame_id` skew counters, and — with
`--trace-latency` — true input-to-sink latency percentiles. It emits a versioned
JSON report with git revision, package versions, platform, CPU, and CUDA device.

```bash
# GPU, sustained 150 Hz camera sets, with end-to-end latency tracing
python examples/tomographic_ao_stress/run_benchmark.py \
  --backend gpu --main-rate 150 --warmup 5 --duration 30 \
  --trace-latency --json-out results/gpu_sustained_150hz.json

# CPU (the 1 GiB matrix-vector multiply dominates; run a lower rate)
python examples/tomographic_ao_stress/run_benchmark.py \
  --backend cpu --main-rate 20 --warmup 3 --duration 30 --trace-latency
```

Omit `--main-rate` for unthrottled saturation mode. On the development RTX 5090
the GPU pipeline sustains ~150–200 Hz through the eight-camera matching barrier
(the per-stream GPU consistency syncs are the ceiling) at ~2 ms p50 end-to-end
latency; the CPU pipeline is bounded by host memory bandwidth on the 1 GiB
reconstructor read.

## Files

| File | Purpose |
|------|---------|
| `pipeline_cpu.yaml`, `pipeline_gpu.yaml` | The two self-contained pipelines |
| `run_benchmark.py` | Instrumented stress/latency harness |
| `generate_calibrations.py` | Materialize `.npy` calibrations + `expected_dimensions.json` |
| `expected_dimensions.json` | Every stream's shape/dtype/bytes (≈1 GiB total) |
| `results/` | Checked-in benchmark JSON reports |
| `IMPLEMENTATION_PLAN.md` | Design plan and current status |

The 1 GiB reconstructor and `.npy` calibrations are never committed; they are
regenerated deterministically.
