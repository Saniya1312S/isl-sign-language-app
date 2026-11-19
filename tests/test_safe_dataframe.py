"""
Regression and Unit Tests for Safe DataFrame and Type Normalization.

Tests all requirements:
1. Regression test for AttributeError: Can only use .str accessor with string values!
2. Safe boolean normalization across bool, string, numeric, None, NaN, pd.NA, mixed.
3. Safe numeric conversion with '%' and 'ms' stripping.
4. Safe percentage and metric calculations with zero-division guards.
5. Empty DataFrame and missing columns handling.
6. Schema normalization and batch test metrics.
"""
import numpy as np
import pandas as pd
import pytest

from src.safe_dataframe import (
    calculate_batch_metrics,
    format_display_dataframe,
    normalize_batch_dataframe,
    safe_mean,
    safe_percentage,
    safe_to_bool_series,
    safe_to_numeric_series,
    validate_required_columns,
)


def test_regression_line_517_bool_dtype_no_str_crash():
    """
    Regression test for the exact user crash:
    AttributeError: Can only use .str accessor with string values!
    When Hand Detected is a boolean column (loaded from CSV), safe_to_bool_series must not crash.
    """
    # DataFrame with boolean Hand Detected (exactly what pd.read_csv produces on boolean CSVs)
    df_bool = pd.DataFrame({
        "Hand Detected": [True, False, True, True],
        "Result": ["PASS", "FAIL", "PASS", "PASS"]
    })

    # The unsafe code: (df_bool["Hand Detected"].str.contains("Yes")).mean() -> CRASHES!
    with pytest.raises(AttributeError):
        _ = df_bool["Hand Detected"].str.contains("Yes")

    # Safe implementation: must not crash and must produce exact mathematical result
    detected = safe_to_bool_series(df_bool["Hand Detected"])
    det_rate = detected.mean()
    assert det_rate == 0.75
    assert detected.tolist() == [True, False, True, True]


def test_safe_to_bool_series_varied_types():
    """Test all supported types: bool, strings, numerics, missing, unexpected."""
    # 1. Pure booleans
    s_bool = pd.Series([True, False, True])
    assert safe_to_bool_series(s_bool).tolist() == [True, False, True]

    # 2. String representations
    s_str = pd.Series(["Yes", "No", "yes", "NO", "TRUE", "False", "1", "0", "✅ Yes", "❌ No"])
    assert safe_to_bool_series(s_str).tolist() == [
        True, False, True, False, True, False, True, False, True, False
    ]

    # 3. Numeric values
    s_num = pd.Series([1, 0, 1.0, 0.0, -1])
    assert safe_to_bool_series(s_num).tolist() == [True, False, True, False, True]

    # 4. Missing values
    s_missing = pd.Series([None, np.nan, pd.NA, ""])
    assert safe_to_bool_series(s_missing).tolist() == [False, False, False, False]

    # 5. Mixed types in single series
    s_mixed = pd.Series([True, "Yes", 1, "❌ No", 0, None, np.nan, "random_text"])
    assert safe_to_bool_series(s_mixed).tolist() == [
        True, True, True, False, False, False, False, False
    ]

    # 6. Empty series
    s_empty = pd.Series([], dtype=object)
    res_empty = safe_to_bool_series(s_empty)
    assert len(res_empty) == 0
    assert res_empty.dtype == bool


def test_safe_to_numeric_series():
    """Test safe conversion to float series with text cleanup."""
    s = pd.Series(["95.5%", "12.3 ms", "0.5", 42, None, "invalid", "100%"])
    res = safe_to_numeric_series(s, default=0.0)
    assert res.iloc[0] == 95.5
    assert res.iloc[1] == 12.3
    assert res.iloc[2] == 0.5
    assert res.iloc[3] == 42.0
    assert res.iloc[4] == 0.0
    assert res.iloc[5] == 0.0
    assert res.iloc[6] == 100.0


def test_safe_percentage_guards():
    """Test zero-division and edge cases in percentage calculation."""
    assert safe_percentage(10, 20) == 0.5
    assert safe_percentage(0, 20) == 0.0
    assert safe_percentage(10, 0) == 0.0
    assert safe_percentage(10, -5) == 0.0
    assert safe_percentage(np.nan, 20) == 0.0
    assert safe_percentage(10, np.nan) == 0.0
    assert safe_percentage(10, None) == 0.0


def test_safe_mean():
    """Test safe mean calculation on empty and valid series."""
    assert safe_mean(pd.Series([10.0, 20.0, 30.0])) == 20.0
    assert safe_mean(pd.Series([], dtype=float)) == 0.0
    assert safe_mean(pd.Series([None, np.nan])) == 0.0


def test_validate_required_columns():
    """Test column presence validation."""
    df = pd.DataFrame({"A": [1], "B": [2]})
    valid, missing = validate_required_columns(df, ["A", "B"])
    assert valid is True
    assert missing == []

    valid, missing = validate_required_columns(df, ["A", "C", "D"])
    assert valid is False
    assert set(missing) == {"C", "D"}

    valid, missing = validate_required_columns(None, ["A"])
    assert valid is False


def test_normalize_batch_dataframe_empty():
    """Test normalization on empty DataFrame."""
    empty_df = pd.DataFrame()
    norm = normalize_batch_dataframe(empty_df)
    assert isinstance(norm, pd.DataFrame)
    assert "Hand Detected" in norm.columns
    assert len(norm) == 0


def test_normalize_batch_dataframe_csv_schema():
    """Test normalization of reports/indian_alphabet_sample_results.csv schema."""
    raw_csv_df = pd.DataFrame({
        "image_filename": ["A.jpg", "B.jpg"],
        "expected_letter": ["A", "B"],
        "predicted_letter": ["A", "C"],
        "confidence": ["0.95", 0.88],
        "hand_detected": [True, "False"],
        "correct": [True, False],
        "processing_time_ms": ["15.2", 12.0]
    })
    norm = normalize_batch_dataframe(raw_csv_df)
    assert norm["Image"].tolist() == ["A.jpg", "B.jpg"]
    assert norm["Expected"].tolist() == ["A", "B"]
    assert norm["Predicted"].tolist() == ["A", "C"]
    assert norm["Hand Detected"].tolist() == [True, False]
    assert norm["Result"].tolist() == ["PASS", "FAIL"]
    assert norm["Confidence"].tolist() == [0.95, 0.88]
    assert norm["Latency (ms)"].tolist() == [15.2, 12.0]


def test_calculate_batch_metrics():
    """Test batch metrics calculation on valid and empty data."""
    # Empty
    empty_metrics = calculate_batch_metrics(pd.DataFrame())
    assert empty_metrics["total_tested"] == 0
    assert empty_metrics["detection_rate"] == 0.0
    assert empty_metrics["overall_accuracy"] == 0.0

    # Populated
    df = pd.DataFrame({
        "Image": ["1.jpg", "2.jpg", "3.jpg", "4.jpg"],
        "Hand Detected": [True, True, True, False],
        "Result": ["PASS", "PASS", "FAIL", "FAIL"],
        "Latency (ms)": [10.0, 20.0, 30.0, 0.0]
    })
    metrics = calculate_batch_metrics(df)
    assert metrics["total_tested"] == 4
    assert metrics["total_detected"] == 3
    assert metrics["total_correct"] == 2
    assert metrics["detection_rate"] == 0.75
    assert metrics["overall_accuracy"] == 0.50
    assert metrics["accuracy_on_detected"] == pytest.approx(2 / 3)


def test_format_display_dataframe():
    """Test display formatting for Streamlit presentation."""
    df = pd.DataFrame({
        "Image": ["A.jpg"],
        "Confidence": [0.9543],
        "Hand Detected": [True],
        "Result": ["PASS"],
        "Latency (ms)": [12.34]
    })
    disp = format_display_dataframe(df)
    assert disp["Confidence"].iloc[0] == "95.4%"
    assert disp["Hand Detected"].iloc[0] == "✅ Yes"
    assert disp["Result"].iloc[0] == "✅ PASS"
    assert disp["Latency (ms)"].iloc[0] == "12.3"
