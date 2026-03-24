"""Public package surface for shmpipeline."""

from shmpipeline.config import KernelConfig, PipelineConfig, SharedMemoryConfig
from shmpipeline.manager import PipelineManager
from shmpipeline.state import PipelineState

__version__ = "0.1.0"

__all__ = [
    "KernelConfig",
    "PipelineConfig",
    "PipelineManager",
    "PipelineState",
    "SharedMemoryConfig",
    "__version__",
]