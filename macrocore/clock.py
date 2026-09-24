"""Virtual clock: deterministic, injectable time source (no wall-clock dependence)."""


class VirtualClock:
    def __init__(self, start: float = 1_700_000_000.0, step: float = 0.001):
        self._t = start
        self._step = step

    def now(self) -> float:
        self._t += self._step
        return self._t
