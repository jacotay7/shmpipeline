# Affine Transformation Example

This example builds a one-stage CPU pipeline that computes:

`output_vector = transform_matrix @ input_vector + offset_vector`

The transform matrix and offset vector live in shared memory just like the
streamed input and output buffers. That means you can build and start the
pipeline once, then update the transform state or feed new inputs from another
process as needed.

The example script is intentionally more than a smoke test. It configures
logging, loads the affine transform buffers, pushes thousands of random input
vectors through the pipeline, and asserts that every output matches the local
reference calculation.

Files in this example:

- `pipeline.yaml`: pipeline configuration
- `run_example.py`: verification script that logs manager activity, streams
    random vectors, and asserts correctness on every result

Run it with:

```bash
python examples/affine_transformation/run_example.py
```