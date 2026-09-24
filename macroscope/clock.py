"""Deterministic clock and stable id factory.

Domain identity never depends on wall-clock time: event order comes from the
append-only sequence. The clock only decorates events; tests inject a virtual
clock and a fixed seed.
"""

import numpy as np


class Clock:
    def __init__(self, now_ms=0, seed=20260924):
        self._now_ms = int(now_ms)
        self._counter = 0
        self._rng = np.random.default_rng(seed)

    def now_ms(self):
        return self._now_ms

    def advance(self, ms=1):
        self._now_ms += int(ms)
        return self._now_ms

    def ulid(self, prefix):
        # 48-bit timestamp + 40 bits of seeded randomness + local counter.
        self._counter += 1
        rand = int(self._rng.integers(0, 1 << 31, dtype=np.int64))
        body = (self._now_ms << 48) ^ (rand << 16) ^ (self._counter & 0xFFFF)
        return f"{prefix}_{body & ((1 << 96) - 1):024x}"
