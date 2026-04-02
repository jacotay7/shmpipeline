"""Public package surface for shmpipeline."""

from shmpipeline.config import KernelConfig, PipelineConfig, SharedMemoryConfig
from shmpipeline.graph import PipelineGraph
from shmpipeline.manager import PipelineManager
from shmpipeline.state import PipelineState
from shmpipeline.synthetic import SyntheticInputConfig

__version__ = "0.1.0"

__all__ = [
    "KernelConfig",
    "PipelineConfig",
    "PipelineGraph",
    "PipelineManager",
    "PipelineState",
    "SharedMemoryConfig",
    "SyntheticInputConfig",
    "__version__",
]
