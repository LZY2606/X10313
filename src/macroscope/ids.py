"""Stable identity hashing and a deterministic virtual clock.

Every identity in the system is derived from canonical content so that a
full replay (and an export/delete/import round trip) produces byte-identical
ids, ordering and provenance relations.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def digest(parts: Any, prefix: str, length: int = 16) -> str:
    """Return ``prefix_<first length hex chars of sha256(canonical parts)>``.

    ``parts`` may be any JSON-serialisable structure; tuples become JSON
    arrays and keys are sorted so the result is order-insensitive for maps
    but order-sensitive for sequences (which is what provenance needs).
    """
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=_default)
    h = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return f"{prefix}_{h[:length]}"


def _default(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    if hasattr(obj, "as_fingerprint"):
        return obj.as_fingerprint()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


class VirtualClock:
    """Deterministic monotonic clock; never reads the wall clock.

    Time is reported as ``vt<N>``.  Replays with the same seed produce the
    same timeline.
    """

    def __init__(self, seed: int = 20260925) -> None:
        self._tick = 0
        self.seed = seed

    def now(self) -> str:
        self._tick += 1
        return f"vt{self.seed:x}-{self._tick:06d}"

    def peek(self) -> str:
        return f"vt{self.seed:x}-{self._tick:06d}"

    def reset(self) -> None:
        self._tick = 0
