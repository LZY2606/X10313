"""MacroScope: local-first hygienic macro workbench.

Modules
-------
- syntax/lexer: tokens, source spans, syntax objects
- patterns/interp: syntax-rules engine and the explicit-macro mini interpreter
- expander: set-of-scopes hygienic expansion, binding explanation, provenance
- store/service: append-only event log in SQLite with replay
- app: FastAPI + SVG review UI
"""

ALGORITHM_VERSION = "macroscope-expander-1.0.0"
