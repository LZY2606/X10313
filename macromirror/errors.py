class MMError(Exception):
    """Base class; carries the guilty syntax object for source back-mapping."""

    def __init__(self, message, guilty=None, stack=(), kind="error"):
        super().__init__(message)
        self.message = message
        self.guilty = guilty
        self.stack = tuple(stack)
        self.kind = kind


class SyntaxFormError(MMError):
    pass


class MacroRaise(MMError):
    """Explicit (syntax-error ...) raised from inside a transformer."""


class BudgetExhausted(MMError):
    """Recursive expansion ran past its budget; partial graph is retained."""


class AmbiguousBinding(MMError):
    def __init__(self, message, guilty, candidates):
        super().__init__(message, guilty, (), kind="ambiguous")
        self.candidates = tuple(candidates)
