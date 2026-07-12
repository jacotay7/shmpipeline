"""A single-stream synthetic array source.

``synthetic.array`` drives one CPU or GPU stream with deterministic pattern
frames, reusing :class:`~shmpipeline.synthetic.SyntheticPatternGenerator`.  It
paces itself to an optional ``rate_hz`` without depending on the controller's
coarse poll interval, so a 700 Hz auxiliary loop keeps its rate.  It exists so
the example YAML is runnable through ``shmpipeline run`` without an external
producer or an example-local entry-point package.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from shmpipeline.config import SharedMemoryConfig, SourceConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.source import Source
from shmpipeline.synthetic import (
    SyntheticPatternGenerator,
    synthetic_config_from_parameters,
)


class SyntheticArraySource(Source):
    """Publish deterministic synthetic frames into one stream at a fixed rate."""

    kind = "synthetic.array"

    @classmethod
    def validate_config(
        cls,
        config: SourceConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        """Accept CPU or GPU streams and validate the optional rate."""
        if config.stream not in shared_memory:
            raise ConfigValidationError(
                f"source {config.name!r} references unknown stream "
                f"{config.stream!r}"
            )
        rate = config.parameters.get("rate_hz")
        if rate is not None and (
            not isinstance(rate, (int, float)) or rate <= 0.0
        ):
            raise ConfigValidationError(
                f"source {config.name!r} rate_hz must be a positive number "
                "or null"
            )
        # Surface pattern/dtype errors at validation time, not first read.
        synthetic_config_from_parameters(config.stream, config.parameters)

    def __init__(self, context) -> None:
        super().__init__(context)
        parameters = context.config.parameters
        rate = parameters.get("rate_hz")
        self._interval = None if rate is None else 1.0 / float(rate)
        self._next_deadline: float | None = None
        spec = context.stream_spec
        self._generator = SyntheticPatternGenerator(
            synthetic_config_from_parameters(
                context.config.stream, parameters
            ),
            shape=spec.shape,
            dtype=spec.dtype,
            storage=spec.storage,
            gpu_device=spec.gpu_device,
        )

    def read(self) -> Any:
        """Return the next paced synthetic frame for the configured stream."""
        if self._interval is not None:
            now = time.perf_counter()
            if self._next_deadline is None:
                self._next_deadline = now
            remaining = self._next_deadline - now
            if remaining > 0.0 and self._stop_event is not None:
                if self._stop_event.wait(remaining):
                    return None
            self._next_deadline = max(
                self._next_deadline + self._interval, time.perf_counter()
            )
        return self._generator.next_frame()
