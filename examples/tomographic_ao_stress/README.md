# Tomographic AO Stress Example

A reproducible synthetic three-loop adaptive-optics workload that stresses
`shmpipeline`: eight hardware-synchronized 256×256 Shack-Hartmann WFS cameras
feed one fused tomographic controller, alongside two independent 16×16 tip/tilt
loops running at their own rates.

Each WFS uses 4×4 pixel tiles → `(64, 64, 2)` centroids = 8192 slopes; all eight
give a 65536-element tomographic vector reconstructed to 4096 actuators. The
float32 `(4096, 65536)` reconstructor is exactly 1 GiB.

The standard CPU and GPU pipelines use identical eight-stream topology:

```text
synthetic.frame_set (8 WFS, one frame_id token per generation)
  └─ tomographic_controller (8 inputs, synchronization: matching_frame_id)
        fuses dark/flat · tiled centroids · slope cal · 4096×65536 reconstruct
        · leaky integrator · command cal · per-actuator clip
     └─ tomo_dm_command → null.sink   (fake DM)
synthetic.array → tt_a_controller → null.sink   (100 Hz tip/tilt)
synthetic.array → tt_b_controller → null.sink   (1000 Hz tip/tilt)
```

The fan-in only combines WFS frames carrying a **matching `frame_id` token**, so
the reconstructor consumes synchronized sets even under load; skewed branches are
dropped (`drop_older`) within `max_skew_generations` / `max_wait_s`.

`pipeline_gpu_batched.yaml` is the maximum-performance form. It transports the
same eight synchronized images as one `(8, 256, 256)` camera cube, vectorizes
all eight centroid calculations, and borrows the locked GPU input rather than
cloning it. This removes seven IPC snapshots and sustains 1 kHz on the
development RTX 5090.

## Run it

Validate or run either pipeline directly — they are self-contained (built-in
`synthetic.frame_set` / `synthetic.array` sources and `null.sink` endpoints):

```bash
shmpipeline validate examples/tomographic_ao_stress/pipeline_gpu.yaml
shmpipeline run examples/tomographic_ao_stress/pipeline_cpu.yaml --duration 5
```

Loading either standard YAML in the GUI and pressing Build/Start is sufficient:
the WFS source runs at 500 Hz, tip/tilt A at 100 Hz, and tip/tilt B at 1000 Hz.
Declarative initializers load seeded nonzero reconstruction/controller data,
inverse flats, rotations, and ±2.5 actuator limits before sources start, so all
three command streams become nonzero without running helper code.

## Benchmark

`run_benchmark.py` substitutes an instrumented camera source (rate control +
frame-id timestamps), loads deterministic calibrations, and reports throughput,
delivery ratio, per-loop rates, `matching_frame_id` skew counters, and — with
`--trace-latency` — true input-to-sink latency percentiles. It emits a versioned
JSON report with git revision, package versions, platform, CPU, and CUDA device.

```bash
# Standard eight-stream GPU path
python examples/tomographic_ao_stress/run_benchmark.py \
  --backend gpu --main-rate 500 --warmup 5 --duration 30 \
  --trace-latency --json-out results/gpu_sustained_500hz.json

# Batched maximum-performance GPU path at 1 kHz
python examples/tomographic_ao_stress/run_benchmark.py \
  --backend gpu --config examples/tomographic_ao_stress/pipeline_gpu_batched.yaml \
  --main-rate 1000 --loop-a-rate 100 --loop-b-rate 1000 \
  --warmup 5 --duration 30 --json-out results/gpu_batched_1000hz.json

# CPU (the 1 GiB matrix-vector multiply dominates; run a lower rate)
python examples/tomographic_ao_stress/run_benchmark.py \
  --backend cpu --main-rate 20 --warmup 3 --duration 30 --trace-latency
```

Omit `--main-rate` for unthrottled saturation mode. On the development RTX 5090
the standard eight-stream path saturates near 534 Hz when requested at 1 kHz;
the batched path runs its controller at 1000 Hz with about 0.812 ms average
compute time. The CPU pipeline remains bounded by host memory bandwidth on the
physical 1 GiB reconstructor read.

## Files

| File | Purpose |
|------|---------|
| `pipeline_cpu.yaml`, `pipeline_gpu.yaml` | Self-contained eight-stream reference pipelines |
| `pipeline_gpu_batched.yaml` | Self-contained 1-kHz-oriented GPU profile |
| `run_benchmark.py` | Instrumented stress/latency harness |
| `generate_calibrations.py` | Materialize `.npy` calibrations + `expected_dimensions.json` |
| `expected_dimensions.json` | Every stream's shape/dtype/bytes (≈1 GiB total) |
| `results/` | Checked-in benchmark JSON reports |
| `IMPLEMENTATION_PLAN.md` | Design plan and current status |

The 1 GiB reconstructor and `.npy` calibrations are never committed; they are
regenerated deterministically.
