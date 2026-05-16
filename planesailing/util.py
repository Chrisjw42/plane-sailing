"""Conversion helpers for turning rs1090-decoded DataFrame rows into Postgres-friendly values.

These exist because:
- pandas/numpy types (np.int64, np.float64, pd.Timestamp) are not JSON-serialisable, so
  anything bound for a JSONB column needs sanitising.
- rs1090 may emit floats where the schema wants ints (e.g. altitude).
- NaN must be coerced to None for both nullable columns and JSON.
"""

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


def last_non_null(group: pd.DataFrame, col: str) -> Any:
    if col not in group.columns:
        return None
    s = group[col].dropna()
    return s.iloc[-1] if not s.empty else None


def json_safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: json_safe(x) for k, x in v.items()}
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def merged_raw(group: pd.DataFrame) -> dict[str, Any]:
    """Collapse a sorted DataFrame group into a dict using last-non-null per column."""
    merged: dict[str, Any] = {}
    for _, row in group.iterrows():
        for k, v in row.dropna().items():
            merged[k] = json_safe(v)
    return merged


def to_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f):
            return None
        return int(f)
    except (TypeError, ValueError):
        return None


def to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def clean_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None
