import re
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any

@dataclass
class ColumnProfile:
    original_name: str
    normalized_name: str
    inferred_type: str
    null_count: int
    null_ratio: float
    unique_count: int
    uniqueness_ratio: float
    samples: List[Any]
    statistics: Dict[str, Any]
    pattern_hints: List[str]

import math
from datetime import datetime, date
import numpy as np

def _normalize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _normalize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_normalize_for_json(v) for v in obj]
    elif pd.isna(obj):  # covers pd.NA, np.nan, None
        return None
    elif isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj

@dataclass
class DatasetProfile:
    row_estimate: int
    is_exact_count: bool
    columns: List[ColumnProfile]

    def to_dict(self) -> Dict[str, Any]:
        return _normalize_for_json(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetProfile":
        cols = [ColumnProfile(**c) for c in data.get("columns", [])]
        return cls(
            row_estimate=data.get("row_estimate", 0),
            is_exact_count=data.get("is_exact_count", False),
            columns=cols
        )

def normalize_name(name: str) -> str:
    """Creates a normalized representation for analysis only."""
    name = str(name).lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    return name.strip('_')

def infer_type(series: pd.Series) -> str:
    s = series.dropna()
    if len(s) == 0:
        return "mixed/unknown"

    ptype = pd.api.types.infer_dtype(s, skipna=True)
    if ptype == "boolean":
        return "boolean"
    if ptype == "integer":
        return "integer"
    if ptype == "floating":
        return "float"
    if ptype in ("datetime64", "datetime", "date"):
        return "datetime"
    
    # Check if a string column could actually be datetime
    if ptype == "string":
        try:
            # We sample a few rows to avoid expensive parsing of the entire series
            pd.to_datetime(s.head(10), format="mixed", errors='raise')
            return "datetime"
        except Exception:
            pass

    return "string"

def profile_dataframe(df: pd.DataFrame, exact_count: bool = False) -> DatasetProfile:
    # Handle duplicate columns by deduplication (e.g. A, A -> A, A_1)
    if df.columns.duplicated().any():
        df = df.copy()
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            idx = cols[cols == dup].index.values.tolist()
            cols.loc[idx] = [f"{dup}_{i}" if i != 0 else str(dup) for i in range(len(idx))]
        df.columns = cols

    total_rows = len(df)
    columns_profile = []
    
    for col in df.columns:
        series = df[col]
        inferred = infer_type(series)
        null_c = int(series.isna().sum())
        null_r = float(null_c / total_rows) if total_rows > 0 else 0.0
        
        # Uniqueness (approximate if sampled)
        nunique = int(series.nunique(dropna=True))
        valid_rows = total_rows - null_c
        uniqueness_r = float(nunique / valid_rows) if valid_rows > 0 else 0.0
        
        # Samples
        samples = series.dropna().head(5).tolist()
        
        # Statistics
        stats = {}
        if inferred in ("integer", "float"):
            stats["min"] = float(series.min()) if not pd.isna(series.min()) else None
            stats["max"] = float(series.max()) if not pd.isna(series.max()) else None
        elif inferred == "datetime":
            try:
                temp_series = pd.to_datetime(series, format="mixed", errors='coerce').dropna()
                stats["min"] = str(temp_series.min()) if not temp_series.empty else None
                stats["max"] = str(temp_series.max()) if not temp_series.empty else None
            except Exception:
                pass
        
        # Pattern hints
        hints = []
        if uniqueness_r > 0.9:
            hints.append("identifier-like")
        if inferred in ("float", "integer"):
            hints.append("numeric")
        if inferred == "string" and uniqueness_r < 0.1 and valid_rows > 10:
            hints.append("categorical-like")
            
        c_prof = ColumnProfile(
            original_name=str(col),
            normalized_name=normalize_name(str(col)),
            inferred_type=inferred,
            null_count=null_c,
            null_ratio=null_r,
            unique_count=nunique,
            uniqueness_ratio=uniqueness_r,
            samples=samples,
            statistics=stats,
            pattern_hints=hints
        )
        columns_profile.append(c_prof)
        
    return DatasetProfile(
        row_estimate=total_rows,
        is_exact_count=exact_count,
        columns=columns_profile
    )
