"""Local-first hygienic macro expansion workbench.

Compiler-toolchain flavored: lex a small Lisp-like language into tokens with
source spans, parse into syntax objects carrying scope sets, expand macros
hygienically (sets-of-scopes), resolve identifier bindings, and record every
step as machine-readable events with stable identities.
"""

__version__ = "0.1.0"
