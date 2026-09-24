from __future__ import annotations

import numpy as np

TIME_FACTORS_TO_SECONDS = {
    "ms": 0.001,
    "s": 1.0,
    "min": 60.0,
}


def normalize_quantities(tokens, algorithm_version, params_fingerprint, source_version_id):
    numeric_tokens = [
        token
        for token in tokens
        if token["kind"] == "number" and token.get("numeric_value") is not None
    ]
    if not numeric_tokens:
        return [], []

    units = [token.get("unit") or "dimensionless" for token in numeric_tokens]
    values = np.array([float(token["numeric_value"]) for token in numeric_tokens], dtype=np.float64)
    factors = np.array(
        [TIME_FACTORS_TO_SECONDS.get(unit, np.nan) for unit in units], dtype=np.float64
    )
    normalized = values * factors

    derived, diagnostics = [], []
    for index, token in enumerate(numeric_tokens):
        unit = units[index]
        common = {
            "id": f"Q:{token['id']}",
            "status": "current",
            "algorithm_version": algorithm_version,
            "params_fingerprint": params_fingerprint,
            "direct_source": {
                "source_version_id": source_version_id,
                "token_id": token["id"],
                "span": token["span"],
                "raw_representation": token["raw"],
            },
            "raw_value": token["raw"],
            "raw_unit": unit,
            "unit_basis": "SI-second" if unit in TIME_FACTORS_TO_SECONDS else unit,
        }
        if unit in TIME_FACTORS_TO_SECONDS:
            derived.append(
                {
                    **common,
                    "value": float(normalized[index]),
                    "unit": "s",
                    "normalization_factor": float(factors[index]),
                    "status": "current",
                }
            )
        else:
            diagnostic = {
                "id": f"D:Q:{token['id']}",
                "code": "unit_not_normalized",
                "severity": "warning",
                    "message": f"unit {unit!r} has no enabled conversion target; raw value remains in place",
                "span": token["span"],
                "token_id": token["id"],
                "status": "unknown",
                "algorithm_version": algorithm_version,
                "params_fingerprint": params_fingerprint,
                "direct_source": common["direct_source"],
            }
            diagnostics.append(diagnostic)
    return derived, diagnostics
