"""Worker placement policies for pipeline processes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from shmpipeline.config import KernelConfig


class WorkerPlacementPolicy(Protocol):
    """Protocol for CPU-slot assignment strategies."""

    def cpu_slot_for(
        self,
        *,
        kernel: KernelConfig,
        index: int,
        cpu_count: int,
    ) -> int | None:
        """Return the CPU slot to pin the worker to, if any."""

    def describe(self) -> str:
        """Return a short human-readable policy description."""


@dataclass(frozen=True)
class RoundRobinPlacementPolicy:
    """Assign workers to CPU slots in declaration order."""

    offset: int = 0
    enabled: bool = True

    def cpu_slot_for(
        self,
        *,
        kernel: KernelConfig,
        index: int,
        cpu_count: int,
    ) -> int | None:
        del kernel
        if not self.enabled or cpu_count <= 0:
            return None
        return (index + self.offset) % cpu_count

    def describe(self) -> str:
        """Return a short policy name."""
        return "round-robin"


@dataclass(frozen=True)
class NoAffinityPlacementPolicy:
    """Disable explicit CPU affinity selection."""

    def cpu_slot_for(
        self,
        *,
        kernel: KernelConfig,
        index: int,
        cpu_count: int,
    ) -> int | None:
        del kernel, index, cpu_count
        return None

    def describe(self) -> str:
        """Return a short policy name."""
        return "none"


def normalize_placement_policy(
    policy: WorkerPlacementPolicy | None,
) -> WorkerPlacementPolicy:
    """Return the configured worker-placement policy or the default."""
    if policy is None:
        return RoundRobinPlacementPolicy()
    return policy
