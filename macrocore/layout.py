"""Deterministic SVG tree layout (NumPy coordinates)."""
from __future__ import annotations

import numpy as np

X_STEP = 20.0
Y_STEP = 52.0
MARGIN = 28.0


def layout(nodes: list[dict], roots: list[int]) -> dict:
    by_id = {n["id"]: n for n in nodes}
    positions: dict[int, tuple[float, float]] = {}
    counter = {"x": 0}

    def measure(node_id, depth):
        node = by_id[node_id]
        children = [c for c in node["children"] if c in by_id]
        if not children:
            x = counter["x"] * X_STEP + MARGIN
            counter["x"] += 1
        else:
            xs = [measure(child, depth + 1) for child in children]
            x = float(np.mean(np.array(xs, dtype=float)))
        positions[node_id] = (x, depth * Y_STEP + MARGIN)
        return x

    root_xs = [measure(root, 0) for root in roots]
    width = max(counter["x"], 1) * X_STEP + 2 * MARGIN
    max_depth = {"v": 0}

    def depth(node_id, d):
        max_depth["v"] = max(max_depth["v"], d)
        node = by_id[node_id]
        for child in node["children"]:
            if child in by_id:
                depth(child, d + 1)

    for root in roots:
        depth(root, 0)
    height = (max_depth["v"] + 1) * Y_STEP + 2 * MARGIN
    if not root_xs:
        width, height = 200.0, 80.0
    return {"positions": [{"id": nid, "x": x, "y": y}
                          for nid, (x, y) in sorted(positions.items())],
            "width": width, "height": height}
