"""Pipeline graph introspection and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shmpipeline.config import PipelineConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.registry import KernelRegistry, get_default_registry


@dataclass(frozen=True)
class GraphEdge:
    """One directed graph edge between a stream and a kernel."""

    source: str
    target: str
    role: str
    stream: str


class PipelineGraph:
    """Derived graph view of one pipeline configuration."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._shared_by_name = config.shared_memory_by_name
        self._kernels_by_name = {
            kernel.name: kernel for kernel in config.kernels
        }
        self._sources_by_name = {
            source.name: source for source in config.sources
        }
        self._sinks_by_name = {sink.name: sink for sink in config.sinks}
        self._kernel_producers = {
            stream_name: [] for stream_name in self._shared_by_name
        }
        self._source_producers = {
            stream_name: [] for stream_name in self._shared_by_name
        }
        self._kernel_consumers = {
            stream_name: [] for stream_name in self._shared_by_name
        }
        self._sink_consumers = {
            stream_name: [] for stream_name in self._shared_by_name
        }
        for kernel in config.kernels:
            self._kernel_producers[kernel.output].append(kernel.name)
            self._kernel_consumers[kernel.input].append(kernel.name)
            for binding in kernel.auxiliary:
                self._kernel_consumers[binding.name].append(kernel.name)
        for source in config.sources:
            self._source_producers[source.stream].append(source.name)
        for sink in config.sinks:
            self._sink_consumers[sink.stream].append(sink.name)

        self._producers = {
            stream_name: [
                *self._source_producers[stream_name],
                *self._kernel_producers[stream_name],
            ]
            for stream_name in self._shared_by_name
        }
        self._consumers = {
            stream_name: [
                *self._kernel_consumers[stream_name],
                *self._sink_consumers[stream_name],
            ]
            for stream_name in self._shared_by_name
        }

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "PipelineGraph":
        """Build a graph view from one loaded configuration."""
        return cls(config)

    @property
    def edges(self) -> tuple[GraphEdge, ...]:
        """Return the directed edges in stream-kernel-stream form."""
        edges: list[GraphEdge] = []
        for source in self.config.sources:
            edges.append(
                GraphEdge(
                    source=source.name,
                    target=source.stream,
                    role="source",
                    stream=source.stream,
                )
            )
        for kernel in self.config.kernels:
            edges.append(
                GraphEdge(
                    source=kernel.input,
                    target=kernel.name,
                    role="input",
                    stream=kernel.input,
                )
            )
            for binding in kernel.auxiliary:
                edges.append(
                    GraphEdge(
                        source=binding.name,
                        target=kernel.name,
                        role=f"auxiliary:{binding.alias}",
                        stream=binding.name,
                    )
                )
            edges.append(
                GraphEdge(
                    source=kernel.name,
                    target=kernel.output,
                    role="output",
                    stream=kernel.output,
                )
            )
        for sink in self.config.sinks:
            edges.append(
                GraphEdge(
                    source=sink.stream,
                    target=sink.name,
                    role="sink",
                    stream=sink.stream,
                )
            )
        return tuple(edges)

    def source_streams(self) -> tuple[str, ...]:
        """Return streams that are only written externally into the graph."""
        explicit_sources = {source.stream for source in self.config.sources}
        return tuple(
            sorted(
                explicit_sources
                | {
                    name
                    for name, producers in self._kernel_producers.items()
                    if not producers
                    and self._kernel_consumers[name]
                    and name not in explicit_sources
                }
            )
        )

    def sink_streams(self) -> tuple[str, ...]:
        """Return streams that are produced but not consumed downstream."""
        explicit_sinks = {sink.stream for sink in self.config.sinks}
        return tuple(
            sorted(
                explicit_sinks
                | {
                    name
                    for name, consumers in self._kernel_consumers.items()
                    if not consumers
                    and self._kernel_producers[name]
                    and name not in explicit_sinks
                }
            )
        )

    def orphaned_streams(self) -> tuple[str, ...]:
        """Return streams unused by all kernels."""
        return tuple(
            sorted(
                name
                for name in self._shared_by_name
                if not self._producers[name] and not self._consumers[name]
            )
        )

    def upstream_kernels(self, kernel_name: str) -> tuple[str, ...]:
        """Return kernels that feed any input of the target kernel."""
        kernel = self._kernels_by_name[kernel_name]
        upstream: set[str] = set()
        for stream_name in kernel.all_inputs:
            upstream.update(self._kernel_producers[stream_name])
        upstream.discard(kernel_name)
        return tuple(sorted(upstream))

    def downstream_kernels(self, kernel_name: str) -> tuple[str, ...]:
        """Return kernels that consume the target kernel's output."""
        kernel = self._kernels_by_name[kernel_name]
        downstream = set(self._kernel_consumers[kernel.output])
        downstream.discard(kernel_name)
        return tuple(sorted(downstream))

    def validation_errors(self) -> list[str]:
        """Return graph-level validation errors.

        The current graph validation rejects ambiguous write ownership where
        more than one kernel produces the same shared-memory stream.
        """
        errors: list[str] = []
        for stream_name in sorted(self._shared_by_name):
            producers = self._producers[stream_name]
            if len(producers) > 1:
                producer_list = ", ".join(sorted(producers))
                kernel_producers = self._kernel_producers[stream_name]
                source_producers = self._source_producers[stream_name]
                if kernel_producers and not source_producers:
                    errors.append(
                        "shared memory "
                        f"{stream_name!r} has multiple producer kernels: "
                        f"{producer_list}"
                    )
                elif source_producers and not kernel_producers:
                    errors.append(
                        "shared memory "
                        f"{stream_name!r} has multiple producer sources: "
                        f"{producer_list}"
                    )
                else:
                    errors.append(
                        "shared memory "
                        f"{stream_name!r} has multiple producers: "
                        f"{producer_list}"
                    )
        return errors

    def raise_for_errors(self) -> None:
        """Raise the first graph validation error, if any exists."""
        errors = self.validation_errors()
        if errors:
            raise ConfigValidationError(errors[0])

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph into a CLI- and GUI-friendly mapping."""
        return {
            "shared_memory": [
                {
                    "name": spec.name,
                    "shape": list(spec.shape),
                    "dtype": str(spec.dtype),
                    "storage": spec.storage,
                    "gpu_device": spec.gpu_device,
                    "producers": tuple(sorted(self._producers[spec.name])),
                    "consumers": tuple(sorted(self._consumers[spec.name])),
                    "role": self._stream_role(spec.name),
                }
                for spec in self.config.shared_memory
            ],
            "sources": [
                {
                    "name": source.name,
                    "kind": source.kind,
                    "stream": source.stream,
                    "parameters": dict(source.parameters),
                    "downstream_kernels": tuple(
                        sorted(self._kernel_consumers[source.stream])
                    ),
                }
                for source in self.config.sources
            ],
            "kernels": [
                {
                    "name": kernel.name,
                    "kind": kernel.kind,
                    "input": kernel.input,
                    "output": kernel.output,
                    "auxiliary": kernel.auxiliary_by_alias,
                    "upstream_kernels": self.upstream_kernels(kernel.name),
                    "downstream_kernels": self.downstream_kernels(kernel.name),
                }
                for kernel in self.config.kernels
            ],
            "sinks": [
                {
                    "name": sink.name,
                    "kind": sink.kind,
                    "stream": sink.stream,
                    "parameters": dict(sink.parameters),
                    "upstream_kernels": tuple(
                        sorted(self._kernel_producers[sink.stream])
                    ),
                }
                for sink in self.config.sinks
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "role": edge.role,
                    "stream": edge.stream,
                }
                for edge in self.edges
            ],
            "source_streams": self.source_streams(),
            "sink_streams": self.sink_streams(),
            "orphaned_streams": self.orphaned_streams(),
        }

    def describe(self) -> str:
        """Return a human-readable pipeline graph summary."""
        lines = ["Pipeline Graph", "", "Shared Memory:"]
        for spec in self.config.shared_memory:
            producers = self._producers[spec.name] or ["external"]
            consumers = self._consumers[spec.name] or ["none"]
            lines.append(
                "- "
                f"{spec.name} "
                f"[{spec.storage} {tuple(spec.shape)} {spec.dtype}] "
                f"role={self._stream_role(spec.name)} "
                f"producers={', '.join(producers)} "
                f"consumers={', '.join(consumers)}"
            )
        lines.append("")
        lines.append("Sources:")
        if self.config.sources:
            for source in self.config.sources:
                downstream = self._kernel_consumers[source.stream] or [
                    "external"
                ]
                lines.append(
                    "- "
                    f"{source.name} ({source.kind}) "
                    f"stream={source.stream} "
                    f"downstream={', '.join(downstream)}"
                )
        else:
            lines.append("- none")
        lines.append("")
        lines.append("Kernels:")
        if self.config.kernels:
            for kernel in self.config.kernels:
                auxiliary = kernel.auxiliary_by_alias
                auxiliary_text = (
                    ", ".join(
                        f"{alias}={stream_name}"
                        for alias, stream_name in auxiliary.items()
                    )
                    if auxiliary
                    else "none"
                )
                upstream = self.upstream_kernels(kernel.name) or ("external",)
                downstream = self.downstream_kernels(kernel.name) or (
                    "terminal",
                )
                lines.append(
                    "- "
                    f"{kernel.name} ({kernel.kind}) "
                    f"input={kernel.input} output={kernel.output} "
                    f"auxiliary={auxiliary_text} "
                    f"upstream={', '.join(upstream)} "
                    f"downstream={', '.join(downstream)}"
                )
        else:
            lines.append("- none")
        lines.append("")
        lines.append("Sinks:")
        if self.config.sinks:
            for sink in self.config.sinks:
                upstream = self._kernel_producers[sink.stream] or ["external"]
                lines.append(
                    "- "
                    f"{sink.name} ({sink.kind}) "
                    f"stream={sink.stream} "
                    f"upstream={', '.join(upstream)}"
                )
        else:
            lines.append("- none")

        orphaned = self.orphaned_streams()
        if orphaned:
            lines.append("")
            lines.append("Orphaned Shared Memory:")
            for stream_name in orphaned:
                lines.append(f"- {stream_name}")

        errors = self.validation_errors()
        if errors:
            lines.append("")
            lines.append("Validation Errors:")
            for error in errors:
                lines.append(f"- {error}")

        return "\n".join(lines)

    def _stream_role(self, stream_name: str) -> str:
        if stream_name in self.orphaned_streams():
            return "orphan"
        if (
            stream_name in self.source_streams()
            and stream_name in self.sink_streams()
        ):
            return "source+sink"
        if stream_name in self.source_streams():
            return "source"
        if stream_name in self.sink_streams():
            return "sink"
        return "intermediate"


def validate_pipeline_config(
    config: PipelineConfig,
    *,
    registry: KernelRegistry | None = None,
) -> list[str]:
    """Return all config, graph, and kernel-binding validation errors."""
    errors = PipelineGraph.from_config(config).validation_errors()
    registry = registry or get_default_registry()
    shared_by_name = config.shared_memory_by_name
    for kernel in config.kernels:
        try:
            registry.validate(kernel, shared_by_name)
        except ConfigValidationError as exc:
            errors.append(str(exc))
    for source in config.sources:
        try:
            registry.validate_source(source, shared_by_name)
        except ConfigValidationError as exc:
            errors.append(str(exc))
    for sink in config.sinks:
        try:
            registry.validate_sink(sink, shared_by_name)
        except ConfigValidationError as exc:
            errors.append(str(exc))
    return errors
