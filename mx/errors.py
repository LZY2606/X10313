"""Domain errors carrying a syntax-origin chain (never just a message)."""


class MxError(Exception):
    def __init__(self, message, origin_chain=None, phase="expand", diagnostic_id=None):
        super().__init__(message)
        self.message = message
        self.origin_chain = list(origin_chain or [])
        self.phase = phase
        self.diagnostic_id = diagnostic_id

    def to_dict(self):
        return {
            "kind": self.__class__.__name__,
            "message": self.message,
            "phase": self.phase,
            "diagnostic_id": self.diagnostic_id,
            "origin_chain": self.origin_chain,
        }


class LexError(MxError):
    def __init__(self, message, line, col, offset):
        super().__init__(message, phase="lex", diagnostic_id=f"LEX:{line}:{col}")
        self.line = line
        self.col = col
        self.offset = offset


class ParseError(MxError):
    def __init__(self, message, span=None):
        super().__init__(message, phase="parse", diagnostic_id="PARSE")
        self.span = span


class UnboundError(MxError):
    def __init__(self, message, origin_chain=None):
        super().__init__(message, origin_chain=origin_chain, phase="resolve",
                         diagnostic_id="UNBOUND")


class AmbiguousBindingError(MxError):
    def __init__(self, message, candidates, origin_chain=None):
        super().__init__(message, origin_chain=origin_chain, phase="resolve",
                         diagnostic_id="AMBIGUOUS")
        self.candidates = candidates


class BudgetExceeded(MxError):
    def __init__(self, message, limit):
        super().__init__(message, phase="expand", diagnostic_id="BUDGET")
        self.limit = limit


class RuntimeMxError(MxError):
    def __init__(self, message, origin_chain=None):
        super().__init__(message, origin_chain=origin_chain, phase="runtime",
                         diagnostic_id="RUNTIME")


class FormatError(MxError):
    """Explicit, recoverable import-format error (with coordinate basis)."""

    def __init__(self, message, coordinate=None, line=None):
        super().__init__(message, phase="import", diagnostic_id="IMPORT-FORMAT")
        self.coordinate = coordinate
        self.line = line

    def to_dict(self):
        d = super().to_dict()
        d["coordinate"] = self.coordinate
        d["line"] = self.line
        return d
