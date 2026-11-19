"""
Safe DataFrame and Type Normalization Utilities.

Provides robust, crash-proof dataframe helpers for:
- Boolean series conversion (handles bool, str, numeric, missing, mixed)
- Numeric conversion with string cleaning
- Safe percentage and rate calculation with defined denominators
- Column validation and empty dataframe guards
- Batch test dataframe normalization and metric computation
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def safe_to_bool_series(series: Any) -> pd.Series:
    """
    Safely convert any input into a boolean pd.Series.

    Handles:
    - Boolean: True, False
    - Strings: 'yes', 'no', 'true', 'false', '1', '0', 'y', 'n', 't', 'f', '✅ Yes', '❌ No'
    - Numerics: 1, 0, 1.0, 0.0, non-zero -> True, 0 -> False
    - Missing: None, np.nan, pd.NA, NaT, '' -> False
    - Unexpected values: False (never crashes)

    Returns:
        pd.Series of dtype bool.
    """
    if series is None:
        return pd.Series([], dtype=bool)

    if not isinstance(series, pd.Series):
        try:
            series = pd.Series(series)
        except Exception:
            return pd.Series([], dtype=bool)

    if len(series) == 0:
        return pd.Series([], dtype=bool)

    # If already pure boolean dtype
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    # If numeric dtype
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0.0

    # For object, string, categorical, or mixed types
    def _parse_bool_value(val: Any) -> bool:
        if val is None or pd.isna(val):
            return False
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float, np.integer, np.floating)):
            return bool(val != 0)
        
        # String parsing
        s = str(val).strip().lower()
        if not s:
            return False
        
        # Check positive indicators
        if s in {"true", "1", "1.0", "yes", "y", "t", "pass", "detected", "valid"}:
            return True
        if "yes" in s or "true" in s or "✅" in s:
            return True
        
        return False

    return series.apply(_parse_bool_value).astype(bool)


def safe_to_numeric_series(series: Any, default: float = 0.0) -> pd.Series:
    """
    Safely convert any input into a numeric float pd.Series.
    Strips symbols like '%', 'ms', and whitespace. Missing/invalid become default.
    """
    if series is None:
        return pd.Series([], dtype=float)

    if not isinstance(series, pd.Series):
        try:
            series = pd.Series(series)
        except Exception:
            return pd.Series([], dtype=float)

    if len(series) == 0:
        return pd.Series([], dtype=float)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(default).astype(float)

    def _clean_str(val: Any) -> Any:
        if val is None or pd.isna(val):
            return default
        if isinstance(val, (int, float)):
            return val
        s = str(val).strip().rstrip("%").rstrip("ms").rstrip("s").strip()
        return s

    cleaned = series.apply(_clean_str)
    return pd.to_numeric(cleaned, errors="coerce").fillna(default).astype(float)


def safe_percentage(numerator: Any, denominator: Any, default: float = 0.0) -> float:
    """
    Safely compute numerator / denominator as a fraction (0.0 to 1.0).
    Guards against division by zero, NaN, None, negative denominators.
    """
    try:
        num = float(numerator)
        den = float(denominator)
        if den <= 0 or np.isnan(den) or np.isinf(den) or np.isnan(num) or np.isinf(num):
            return default
        return float(num / den)
    except (ValueError, TypeError, ZeroDivisionError):
        return default


def safe_mean(series: Any, default: float = 0.0) -> float:
    """Safely calculate mean of a series. Returns default if empty or all-NaN."""
    num_series = safe_to_numeric_series(series, default=np.nan)
    if len(num_series) == 0 or num_series.dropna().empty:
        return default
    return float(num_series.dropna().mean())


def validate_required_columns(df: Any, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that df is a DataFrame and contains all required columns.
    Returns (is_valid, list_of_missing_columns).
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return False, list(required_columns)
    
    missing = [c for c in required_columns if c not in df.columns]
    return len(missing) == 0, missing


def normalize_batch_dataframe(df: Any) -> pd.DataFrame:
    """
    Standardize a batch prediction dataframe into consistent schema and types:
    Columns:
    - Image: str
    - Source: str ('Indian', 'External', 'Unknown')
    - Expected: str (e.g. 'A')
    - Predicted: str (e.g. 'A' or 'None')
    - Confidence: float (0.0 to 1.0)
    - Hand Detected: bool (True or False)
    - Result: str ('PASS' or 'FAIL')
    - Latency (ms): float
    - Status: str
    - Notes: str
    """
    standard_columns = [
        "Image", "Source", "Expected", "Predicted",
        "Confidence", "Hand Detected", "Result", "Latency (ms)",
        "Status", "Notes"
    ]

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=standard_columns)

    res = df.copy()

    # Column mapping dictionary for varied CSV schemas
    col_map = {
        "image_filename": "Image",
        "filename": "Image",
        "image": "Image",
        "source": "Source",
        "expected_letter": "Expected",
        "expected": "Expected",
        "predicted_letter": "Predicted",
        "predicted": "Predicted",
        "confidence": "Confidence",
        "hand_detected": "Hand Detected",
        "detected": "Hand Detected",
        "correct": "Result",
        "result": "Result",
        "processing_time_ms": "Latency (ms)",
        "latency_ms": "Latency (ms)",
        "latency": "Latency (ms)",
        "status": "Status",
        "notes": "Notes",
        "message": "Notes"
    }

    # Rename existing columns
    rename_dict = {col: col_map[col.lower()] for col in res.columns if col.lower() in col_map and col not in standard_columns}
    res = res.rename(columns=rename_dict)

    # Ensure required columns exist
    if "Image" not in res.columns:
        res["Image"] = [f"sample_{i}.jpg" for i in range(len(res))]
    if "Source" not in res.columns:
        res["Source"] = "Unknown"
    if "Expected" not in res.columns:
        res["Expected"] = ""
    if "Predicted" not in res.columns:
        res["Predicted"] = "None"
    if "Confidence" not in res.columns:
        res["Confidence"] = 0.0
    if "Hand Detected" not in res.columns:
        res["Hand Detected"] = False
    if "Result" not in res.columns:
        res["Result"] = "FAIL"
    if "Latency (ms)" not in res.columns:
        res["Latency (ms)"] = 0.0
    if "Status" not in res.columns:
        res["Status"] = "completed"
    if "Notes" not in res.columns:
        res["Notes"] = ""

    # Normalize types
    res["Image"] = res["Image"].fillna("").astype(str)
    res["Source"] = res["Source"].fillna("Unknown").astype(str)
    res["Expected"] = res["Expected"].fillna("").astype(str).str.upper()
    res["Predicted"] = res["Predicted"].fillna("None").astype(str)
    res["Confidence"] = safe_to_numeric_series(res["Confidence"], default=0.0)
    res["Hand Detected"] = safe_to_bool_series(res["Hand Detected"])
    
    # Result normalization: 'PASS' if true/pass, 'FAIL' otherwise
    def _normalize_result(val: Any) -> str:
        if isinstance(val, str) and val.strip().upper() in {"PASS", "TRUE", "1"}:
            return "PASS"
        if isinstance(val, (bool, np.bool_)) and val:
            return "PASS"
        if isinstance(val, (int, float)) and val == 1:
            return "PASS"
        return "FAIL"

    res["Result"] = res["Result"].apply(_normalize_result)
    res["Latency (ms)"] = safe_to_numeric_series(res["Latency (ms)"], default=0.0)
    res["Status"] = res["Status"].fillna("unknown").astype(str)
    res["Notes"] = res["Notes"].fillna("").astype(str)

    return res[standard_columns]


def calculate_batch_metrics(df: Any) -> Dict[str, Any]:
    """
    Calculate summary KPI metrics from a batch test dataframe with zero-division safety.
    """
    norm_df = normalize_batch_dataframe(df) if isinstance(df, pd.DataFrame) else pd.DataFrame()
    
    total_tested = len(norm_df)
    if total_tested == 0:
        return {
            "total_tested": 0,
            "total_detected": 0,
            "total_correct": 0,
            "detection_rate": 0.0,
            "overall_accuracy": 0.0,
            "accuracy_on_detected": 0.0,
            "avg_latency_ms": 0.0
        }

    detected_series = safe_to_bool_series(norm_df["Hand Detected"])
    pass_series = norm_df["Result"] == "PASS"

    total_detected = int(detected_series.sum())
    total_correct = int(pass_series.sum())

    det_rate = safe_percentage(total_detected, total_tested)
    overall_acc = safe_percentage(total_correct, total_tested)
    acc_on_det = safe_percentage(total_correct, total_detected) if total_detected > 0 else 0.0
    avg_lat = safe_mean(norm_df["Latency (ms)"], default=0.0)

    return {
        "total_tested": total_tested,
        "total_detected": total_detected,
        "total_correct": total_correct,
        "detection_rate": det_rate,
        "overall_accuracy": overall_acc,
        "accuracy_on_detected": acc_on_det,
        "avg_latency_ms": avg_lat
    }


def format_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format batch dataframe for user-friendly presentation in Streamlit.
    Formats numbers as percentages, latencies as ms, and status with emojis.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    disp = df.copy()
    if "Confidence" in disp.columns:
        disp["Confidence"] = safe_to_numeric_series(disp["Confidence"]).apply(lambda c: f"{c:.1%}")
    if "Latency (ms)" in disp.columns:
        disp["Latency (ms)"] = safe_to_numeric_series(disp["Latency (ms)"]).apply(lambda l: f"{l:.1f}")
    if "Hand Detected" in disp.columns:
        disp["Hand Detected"] = safe_to_bool_series(disp["Hand Detected"]).apply(
            lambda h: "✅ Yes" if h else "❌ No"
        )
    if "Result" in disp.columns:
        disp["Result"] = disp["Result"].apply(
            lambda r: "✅ PASS" if str(r).upper() == "PASS" else "❌ FAIL"
        )

    return disp
