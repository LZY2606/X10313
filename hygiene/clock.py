"""Deterministic virtual clock and fixed-seed generator.

All timestamps and generated symbols in the system flow through a Clock so
that tests run with virtual time and a fixed seed. Real mode derives only
from an explicit start time; nothing is read from the host clock implicitly
during a derivation (the derivation is a pure replay of recorded events).
"""
from __future__ import annotations

import numpy as np


class Clock:
    def __init__(self, start_ms: int = 0, step_ms: int = 1, seed: int = 20260925):
        self.now_ms = int(start_ms)
        self.step_ms = int(step_ms)
        self._rng = np.random.default_rng(seed)
        self.seed = seed

    def tick(self) -> int:
        t = self.now_ms
        self.now_ms += self.step_ms
        return t

    def peek(self) -> int:
        return self.now_ms

    def rand16(self) -> str:
        return f"{int(self._rng.integers(0, 0x10000)):04x}"
