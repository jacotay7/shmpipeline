# CLI Guide

The `shmpipeline` CLI provides four headless workflows:

- validate a pipeline config
- describe the derived graph
- run a pipeline until interrupted or for a bounded duration
- serve one pipeline manager over HTTP and SSE

## Validate

```bash
shmpipeline validate pipeline.yaml
```

This loads the YAML, validates graph constraints, and checks each kernel binding against the registered kernel set without creating shared memory.

## Describe

```bash
shmpipeline describe pipeline.yaml
shmpipeline describe pipeline.yaml --json
```

This derives the graph before worker startup. The JSON form is useful for automation and external tooling.

The graph payload includes:

- shared-memory definitions and roles
- upstream and downstream kernel dependencies
- source streams, sink streams, and orphaned streams
- directed graph edges between streams and kernels

## Run

```bash
shmpipeline run pipeline.yaml
shmpipeline run pipeline.yaml --duration 5.0
shmpipeline run pipeline.yaml --duration 5.0 --json-status
```

The run command builds shared memory, starts workers, polls runtime status, and shuts the pipeline down on exit.

## Serve

```bash
shmpipeline serve pipeline.yaml
shmpipeline serve pipeline.yaml --host 127.0.0.1 --port 8765
shmpipeline serve pipeline.yaml --host 0.0.0.0 --port 8765 --token change-me
```

The serve command hosts one `PipelineManager` behind a small HTTP control plane.

Use it when you want:

- a remote status endpoint for scripts or dashboards
- lifecycle control from another process or machine
- a stable bridge between the in-process manager API and external tooling

See [control plane](control-plane.md) for the endpoint model and Python client example.

## Logging

Set CLI verbosity with `--log-level`:

```bash
shmpipeline --log-level DEBUG describe pipeline.yaml
```

Supported values:

- `DEBUG`
- `INFO`
- `WARNING`
- `ERROR`

## Recommended workflow

1. Validate the config.
2. Describe the graph to confirm dataflow.
3. Run the pipeline with a bounded duration when smoke-testing new configs.
4. Use `--json-status` when integrating with scripts.
