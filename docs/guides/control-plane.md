# Control Plane

`shmpipeline` can expose one `PipelineManager` instance over HTTP so other local processes, browser tooling, or remote machines can control the pipeline state machine without importing your application process directly.

The built-in control plane uses:

- JSON request-response endpoints for commands and snapshots
- JSON request-response endpoints for pulling, validating, and replacing the active editable document
- local discovery records for loopback control servers launched on the same machine
- Server-Sent Events for live worker events and metric updates

## Install

```bash
pip install "shmpipeline[control]"
```

## Start the service

```bash
shmpipeline serve pipeline.yaml --host 127.0.0.1 --port 8765
```

For non-local binding, provide a bearer token:

```bash
shmpipeline serve pipeline.yaml --host 0.0.0.0 --port 8765 --token change-me
```

The server intentionally refuses non-local binding without a token.

## Endpoint surface

Read endpoints:

- `GET /health`
- `GET /info`
- `GET /document`
- `GET /status`
- `GET /snapshot`
- `GET /graph`
- `GET /events`

Document endpoints:

- `PUT /document`
- `POST /document/validate`
- `POST /document/load`

Lifecycle endpoints:

- `POST /commands/build`
- `POST /commands/start`
- `POST /commands/pause`
- `POST /commands/resume`
- `POST /commands/stop`
- `POST /commands/shutdown`

Synthetic-input endpoints:

- `POST /synthetic/start`
- `POST /synthetic/stop`

## Minimal Control GUI

`shmpipeline-control-gui` is the stripped-down control surface for operators who only need to manage servers and drive the state machine.

It focuses on:

- discovering local control servers
- launching and killing local server processes
- switching the config file a server is using
- showing the current server URL, config path, and pipeline state
- driving `start`, `pause`, `stop`, and `teardown`

It intentionally omits the YAML editor, graph view, runtime tables, and shared-memory viewers from the full GUI.

## Python client

```python
from shmpipeline.control import RemoteManagerClient

with RemoteManagerClient("http://127.0.0.1:8765", token="change-me") as client:
    document = client.document()
    client.update_document(document["document"])
    client.build()
    client.start()
    snapshot = client.snapshot()
    print(snapshot["state"])
    print(snapshot["summary"])
    client.pause()
    client.resume()
    client.stop(force=True)
    client.shutdown(force=True)
```

Synthetic-input example:

```python
client.start_synthetic_input(
    {
        "stream_name": "input_frame",
        "pattern": "random",
        "rate_hz": 500.0,
        "seed": 0,
    }
)
```

## Live event stream

`GET /events` returns a text/event-stream response with records such as:

- `worker_started`
- `worker_metrics`
- `worker_failed`
- `worker_stopped`
- `snapshot`

Each `snapshot` event contains the latest manager snapshot after commands such as build, start, stop, shutdown, or synthetic-input changes.

Pipeline failures remain part of the normal status and snapshot payloads so remote GUIs can relay worker errors to users without parsing logs out of band.

## Design constraints

- One service owns one manager instance.
- That same service owns the editable pipeline document associated with that manager.
- Manager operations are serialized so state transitions remain coherent.
- The control plane is a transport layer around the existing manager API, not a second state machine.
- Loopback-only binding is the safe default.