Core API
========

Package Surface
---------------

.. automodule:: shmpipeline

The top-level package re-exports the primary user-facing objects from the
underlying modules so most applications can import from ``shmpipeline``
directly.

Common imports include:

- :class:`shmpipeline.config.PipelineConfig`
- :class:`shmpipeline.manager.PipelineManager`
- :class:`shmpipeline.graph.PipelineGraph`
- :class:`shmpipeline.registry.KernelRegistry`
- :class:`shmpipeline.kernel.Kernel`
- :class:`shmpipeline.kernel.KernelContext`
- :class:`shmpipeline.synthetic.SyntheticInputConfig`
- :func:`shmpipeline.registry.get_default_registry`

Configuration Models
--------------------

.. automodule:: shmpipeline.config
   :members:
   :show-inheritance:

Kernel Base API
---------------

.. automodule:: shmpipeline.kernel
   :members:
   :show-inheritance:

Graph Introspection
-------------------

.. automodule:: shmpipeline.graph
   :members:
   :show-inheritance:

Pipeline Manager
----------------

.. automodule:: shmpipeline.manager
   :members:
   :show-inheritance:

Kernel Registry
---------------

.. automodule:: shmpipeline.registry
   :members:
   :show-inheritance:

Synthetic Inputs
----------------

.. automodule:: shmpipeline.synthetic
   :members:
   :show-inheritance:

State Model
-----------

.. automodule:: shmpipeline.state
   :members:
   :show-inheritance:

CLI Module
----------

.. automodule:: shmpipeline.cli
   :members:
   :show-inheritance:
