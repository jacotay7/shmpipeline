"""Source plugin abstractions executed by the manager thread runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from shmpipeline._endpoint_base import _EndpointBase
from shmpipeline.config import SharedMemoryConfig, SourceConfig
from shmpipeline.errors import ConfigValidationError


@dataclass(frozen=True)
class SourceContext:
    """Static information available to one source instance."""

    config: SourceConfig
    shared_memory: Mapping[str, SharedMemoryConfig]
    auxiliary_streams: Mapping[str, Any] = field(default_factory=dict)

    @property
    def stream_spec(self) -> SharedMemoryConfig:
        """Return the shared-memory specification written by the source."""
        return self.shared_memory[self.config.stream]

    @property
    def auxiliary_specs(self) -> dict[str, SharedMemoryConfig]:
        """Return auxiliary shared-memory specs keyed by alias."""
        return {
            binding.alias: self.shared_memory[binding.name]
            for binding in self.config.auxiliary
        }


class Source(_EndpointBase, ABC):
    """Base class for source plugins managed by the runtime.

    Sources are manager-owned thread plugins. Implementations normally
    override :meth:`validate_config` for parameter checks, optionally perform
    setup and teardown in :meth:`open` and :meth:`close`, and implement
    :meth:`read` to return the next payload for the bound stream.
    """

    kind = "source.base"
    storage = "cpu"

    def __init__(self, context: SourceContext) -> None:
        _EndpointBase.__init__(self)
        self.context = context

    @classmethod
    def validate_config(
        cls,
        config: SourceConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        """Validate storage compatibility before the source is built."""
        if shared_memory[config.stream].storage != cls.storage:
            raise ConfigValidationError(
                f"source {config.name!r} of kind {cls.kind!r} requires "
                f"{cls.storage} shared memory for {config.stream!r}"
            )
        for binding in config.auxiliary:
            if shared_memory[binding.name].storage != cls.storage:
                raise ConfigValidationError(
                    f"source {config.name!r} of kind {cls.kind!r} requires "
                    f"{cls.storage} shared memory for auxiliary {binding.name!r}"
                )

    def open(self) -> None:
        """Prepare the source before the runtime thread starts."""

    @abstractmethod
    def read(self) -> Any | None:
        """Return the next payload for the configured stream.

        Returning ``None`` indicates that no new payload is currently ready and
        the runtime should sleep for the configured poll interval before trying
        again.
        """

    def close(self) -> None:
        """Release any external resources owned by the source."""
