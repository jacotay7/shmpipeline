"""Sink plugin abstractions executed by the manager thread runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from shmpipeline._endpoint_base import _EndpointBase
from shmpipeline.config import SharedMemoryConfig, SinkConfig
from shmpipeline.errors import ConfigValidationError


@dataclass(frozen=True)
class SinkContext:
    """Static information available to one sink instance."""

    config: SinkConfig
    shared_memory: Mapping[str, SharedMemoryConfig]
    auxiliary_streams: Mapping[str, Any] = field(default_factory=dict)

    @property
    def stream_spec(self) -> SharedMemoryConfig:
        """Return the shared-memory specification consumed by the sink."""
        return self.shared_memory[self.config.stream]

    @property
    def auxiliary_specs(self) -> dict[str, SharedMemoryConfig]:
        """Return auxiliary shared-memory specs keyed by alias."""
        return {
            binding.alias: self.shared_memory[binding.name]
            for binding in self.config.auxiliary
        }


class Sink(_EndpointBase, ABC):
    """Base class for sink plugins managed by the runtime.

    Sinks are manager-owned thread plugins. Implementations normally override
    :meth:`validate_config` for parameter checks, optionally perform setup and
    teardown in :meth:`open` and :meth:`close`, and implement :meth:`consume`
    to handle payloads read from the bound stream.
    """

    kind = "sink.base"
    storage = "cpu"

    def __init__(self, context: SinkContext) -> None:
        _EndpointBase.__init__(self)
        self.context = context

    @classmethod
    def validate_config(
        cls,
        config: SinkConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        """Validate storage compatibility before the sink is built."""
        if shared_memory[config.stream].storage != cls.storage:
            raise ConfigValidationError(
                f"sink {config.name!r} of kind {cls.kind!r} requires "
                f"{cls.storage} shared memory for {config.stream!r}"
            )
        for binding in config.auxiliary:
            if shared_memory[binding.name].storage != cls.storage:
                raise ConfigValidationError(
                    f"sink {config.name!r} of kind {cls.kind!r} requires "
                    f"{cls.storage} shared memory for auxiliary {binding.name!r}"
                )

    def open(self) -> None:
        """Prepare the sink before the runtime thread starts."""

    @abstractmethod
    def consume(self, value: Any) -> None:
        """Handle one payload read from the configured stream."""

    def close(self) -> None:
        """Release any external resources owned by the sink."""
