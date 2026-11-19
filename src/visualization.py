"""
Visualization helpers for HUD overlays, charts, and Streamlit components.
"""
from __future__ import annotations

import cv2
import numpy as np


def draw_hud(image: np.ndarray, prediction_info: dict) -> np.ndarray:
    """Overlay clean prediction HUD on an image frame."""
    out = image.copy()
    h, w = out.shape[:2]

    # Semi-transparent top bar
    cv2.rectangle(out, (0, 0), (w, 50), (20, 20, 20), -1)

    label = prediction_info.get("predicted_letter")
    conf = prediction_info.get("confidence", 0.0)
    time_ms = prediction_info.get("processing_time_ms", 0.0)
    status = prediction_info.get("status", "")

    if status == "success" and label:
        txt = f"Predicted: {label} ({conf:.0%}) | {time_ms:.1f}ms"
        color = (30, 210, 180)  # Teal
    elif status == "no_hand_detected":
        txt = "No Hand Detected"
        color = (80, 80, 240)  # Red
    else:
        txt = f"Uncertain ({label} {conf:.0%})"
        color = (40, 200, 240)  # Yellow

    cv2.putText(out, txt, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return out
