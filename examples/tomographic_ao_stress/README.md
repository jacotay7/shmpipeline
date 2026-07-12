# Tomographic AO Stress Example

This in-progress example exercises a full-grid tomographic adaptive-optics
pipeline with eight synchronized 256 x 256 Shack-Hartmann sensors, two
independent tip/tilt loops, and CPU/GPU variants.

Each WFS uses 4 x 4 pixel tiles and produces `(64, 64, 2)` centroids. All 8192
slopes from every WFS are retained, giving a 65536-element tomographic vector.
The float32 `(4096, 65536)` reconstructor occupies exactly 1 GiB.

Validate the graphs without allocating their streams:

```bash
shmpipeline validate examples/tomographic_ao_stress/pipeline_cpu.yaml
shmpipeline validate examples/tomographic_ao_stress/pipeline_gpu.yaml
```

Run the initial synthetic harness:

```bash
python examples/tomographic_ao_stress/run_benchmark.py \
  --backend cpu --warmup 1 --duration 5

python examples/tomographic_ao_stress/run_benchmark.py \
  --backend gpu --profile sustained --main-rate 675 \
  --warmup 5 --duration 30
```

The default `sustained` profile is optimized for long-running fixed camera
rates. On GPU it publishes one synchronized `(8, 256, 256)` camera-set tensor,
fuses dark/flat calibration, all centroids, slope calibration, reconstruction,
integration, command calibration, and bounds into one CUDA worker, and keeps
the small tip/tilt controllers on CPU. This reduces the hot path from 43 CUDA
processes to one CUDA process plus two small CPU workers.

Pass `--gpu-unbatched` to drive the same fused controller from eight separate
`(256, 256)` camera streams through the runtime's `all_new` multi-input barrier
(`pipeline_gpu_sustained.yaml`) instead of one pre-stacked cube. This variant
exercises the synchronized fan-in at full tomographic scale and quantifies the
cost of the eight-stream barrier relative to the batched single-cube source:

```bash
python examples/tomographic_ao_stress/run_benchmark.py \
  --backend gpu --profile sustained --gpu-unbatched --main-rate 100 \
  --warmup 5 --duration 30
```

On the development RTX 5090 host the sustained profile completed a 30-second
run at 675 Hz with 20,250/20,250 main frames delivered, while simultaneously
delivering 21,000/21,000 loop-A frames at 700 Hz and 6,990 terminal loop-B
frames at 233 Hz (one boundary publication was still in flight). The fused main
kernel averaged 1.2465 ms with 5.6 microseconds RMS compute jitter. Overload
began at 700 Hz and saturated near 680 Hz on that host.

For comparison, the CPU sustained profile uses the expanded diagnostic graph,
disables single-core affinity for the reconstructor, and uses 16 OpenBLAS
threads by default:

```bash
python examples/tomographic_ao_stress/run_benchmark.py \
  --backend cpu --profile sustained --main-rate 40 \
  --blas-threads 16 --warmup 3 --duration 30
```

The same host sustained 40 Hz without main-loop loss. Higher requested rates
saturated around 43 Hz because the physical 1 GiB matrix read is limited by
host memory bandwidth.

The main camera-set source publishes the eight images sequentially from one
coordinator thread. The synchronized concatenate stage runs only after all
eight slope streams have advanced. The two 16 x 16 sources run at 700 Hz and
233 Hz by default.

This first build reports terminal publication rates and worker compute metrics.
Propagated hardware frame IDs, true input-to-sink latency, built-in null hardware
sinks, missed-generation metrics, and fault injection remain tracked in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
