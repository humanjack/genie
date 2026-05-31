"""Built-in tools, each exposed via a ``make_<name>(...)`` factory.

Factories bind a session root or sandbox backend and return a configured
:class:`~genie.tools.base.Tool`, so the registry/loop call handlers with only
the model-provided arguments.
"""
