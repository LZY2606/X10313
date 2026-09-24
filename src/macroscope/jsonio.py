"""Canonical JSON encoding for persistence and export."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from .ctenv import ListVal


def canonical_default(obj: Any) -> Any:
    if isinstance(obj, ListVal):
        return [canonical_default(x) for x in obj.items]
    if isinstance(obj, (frozenset, set)):
        return sorted(obj, key=str)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    return str(obj)


def dumps(obj: Any, *, sort_keys: bool = False) -> str:
    return json.dumps(_normalize(obj), sort_keys=sort_keys,
                      ensure_ascii=False, separators=(",", ":"))


def dumps_pretty(obj: Any) -> str:
    return json.dumps(_normalize(obj), ensure_ascii=False, indent=2,
                      sort_keys=False)


def _normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, ListVal):
        return [_normalize(v) for v in obj.items]
    if isinstance(obj, (frozenset, set)):
        return sorted((_normalize(v) for v in obj), key=str)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
