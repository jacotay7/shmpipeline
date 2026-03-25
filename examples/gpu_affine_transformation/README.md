# GPU Affine Transformation Example

This example builds a one-stage GPU pipeline that computes:

`output_vector = transform_matrix @ input_vector + offset_vector`

All buffers live in GPU shared memory, so the streamed input, transform matrix,
offset vector, and output vector are all exchanged as CUDA-backed tensors.

The example script configures logging, loads the transform buffers once,
streams thousands of random vectors through the worker, and checks every output
against a local NumPy reference.

Files in this example:

- `pipeline.yaml`: GPU pipeline configuration
- `run_example.py`: verification script that writes CUDA tensors into shared
  memory and checks every result

Run it with:

```bash
python examples/gpu_affine_transformation/run_example.py
```