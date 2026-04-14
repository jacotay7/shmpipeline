# Installation

Install the smallest profile that matches how you plan to use `shmpipeline`.

## PyPI installs

```bash
pip install shmpipeline
```

GPU support:

```bash
pip install "shmpipeline[gpu]"
```

Desktop GUI support:

```bash
pip install "shmpipeline[gui]"
```

Remote control service support:

```bash
pip install "shmpipeline[control]"
```

## Source installs

Editable source install:

```bash
pip install -e .
```

Full local development environment:

```bash
pip install -e ".[control,gpu,gui,test,docs]"
```

## Platform notes

- CPU pipelines are the default path and have the smallest dependency surface.
- GPU pipelines require a compatible PyTorch and CUDA environment.
- The desktop GUI requires Qt bindings through `PySide6` and plotting support through `pyqtgraph`.
- The desktop GUI extra also installs the local control-plane dependencies so the full editor can auto-launch a loopback server for local workflows.
- The remote control plane requires the `control` extra, which installs FastAPI, Uvicorn, and the Python HTTP client dependency.
- The repository CI runs on Linux, macOS, and Windows. GPU tests are skipped automatically when CUDA is unavailable.

## Verify the install

Validate a checked-in example config:

```bash
shmpipeline validate examples/affine_transformation/pipeline.yaml
```

Describe the derived graph:

```bash
shmpipeline describe examples/affine_transformation/pipeline.yaml --json
```

Start the GUI after installing the GUI extra:

```bash
shmpipeline-gui
```

Start the lightweight remote control GUI:

```bash
shmpipeline-control-gui
```
