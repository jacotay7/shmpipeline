"""Exception hierarchy for shmpipeline."""


class ShmPipelineError(Exception):
    """Base exception for all package-specific failures."""


class ConfigValidationError(ShmPipelineError):
    """Raised when pipeline configuration is invalid."""


class StateTransitionError(ShmPipelineError):
    """Raised when a manager method is called in an invalid state."""


class WorkerProcessError(ShmPipelineError):
    """Raised when a worker process fails while executing a kernel."""


class KernelExecutionError(ShmPipelineError):
    """Raised when a kernel produces invalid runtime output."""