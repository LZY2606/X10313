"""Deterministic identity and virtual time.

Every entity carries a stable identity derived from a per-project counter and a
content hash, so export -> delete -> import preserves identity, order and
provenance.  The clock is virtual: tests inject a fixed schedule.
"""
from __future__ import annotations

import hashlib
import json


class IdGen:
    """Deterministic id generator: <prefix>-<project_seq>-<content_hash8>."""

    def __init__(self, seed: int = 0):
        self.seq = seed

    def next(self, prefix: str, content: object = None) -> str:
        self.seq += 1
        if content is None:
            digest = "00000000"
        else:
            blob = json.dumps(content, sort_keys=True, ensure_ascii=False,
                              default=str).encode("utf-8")
            digest = hashlib.sha256(blob).hexdigest()[:8]
        return f"{prefix}-{self.seq:06d}-{digest}"


class VirtualClock:
    """Monotonic virtual clock; tests pin it, the app advances it."""

    def __init__(self, start: int = 1_700_000_000_000):
        self.now = start

    def tick(self, step: int = 1000) -> int:
        self.now += step
        return self.now


def fingerprint(obj: object) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
