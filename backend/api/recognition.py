"""
Recognition API Endpoints.
Provides:
- POST /api/predict: runs model inference on uploaded or camera frame
- GET /api/samples: lists authentic Indian ISL alphabet samples
- GET /api/samples/{letter}: evaluates or retrieves a specific sample
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.config import ALPHABET_SAMPLES_DIR, SAMPLES_DIR
from backend.services.predictor_service import (
    decode_image_base64,
    encode_image_base64,
    run_prediction_on_image,
)

router = APIRouter(prefix="/api", tags=["Recognition"])


class PredictRequest(BaseModel):
    image: str  # Base64 data URL (e.g. data:image/jpeg;base64,...)
    source: Optional[str] = "webcam"


@router.post("/predict")
def predict_sign_endpoint(req: PredictRequest) -> Dict[str, Any]:
    """Run model prediction on a base64 encoded image frame."""
    if not req.image:
        raise HTTPException(status_code=400, detail="No image data provided.")

    cv_img = decode_image_base64(req.image)
    if cv_img is None:
        raise HTTPException(status_code=400, detail="Failed to decode base64 image data.")

    result = run_prediction_on_image(cv_img)
    return result


@router.post("/predict-upload")
def predict_upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Run model prediction on an uploaded image file."""
    contents = file.file.read()
    import numpy as np
    np_arr = np.frombuffer(contents, np.uint8)
    cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if cv_img is None:
        raise HTTPException(status_code=400, detail="Could not read uploaded image.")

    result = run_prediction_on_image(cv_img)
    return result


@router.get("/samples")
def list_alphabet_samples() -> Dict[str, Any]:
    """List all available authentic Indian ISL alphabet reference samples."""
    samples = []
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for letter in letters:
        sample_path = ALPHABET_SAMPLES_DIR / f"{letter}.jpg"
        if not sample_path.exists():
            sample_path = SAMPLES_DIR / f"Indian_{letter}.jpg"
        
        exists = sample_path.exists()
        samples.append({
            "letter": letter,
            "filename": sample_path.name if exists else f"{letter}.jpg",
            "available": exists
        })

    return {"samples": samples, "total": len(samples)}


@router.get("/samples/{letter}")
def get_sample_image_and_prediction(letter: str) -> Dict[str, Any]:
    """Retrieve sample image and run prediction for a specific letter (A-Z)."""
    clean_letter = letter.strip().upper()
    if len(clean_letter) != 1 or not clean_letter.isalpha():
        raise HTTPException(status_code=400, detail="Invalid letter. Must be A-Z.")

    sample_path = ALPHABET_SAMPLES_DIR / f"{clean_letter}.jpg"
    if not sample_path.exists():
        sample_path = SAMPLES_DIR / f"Indian_{clean_letter}.jpg"

    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample for letter '{clean_letter}' not found.")

    cv_img = cv2.imread(str(sample_path))
    if cv_img is None:
        raise HTTPException(status_code=500, detail="Failed to load sample image file.")

    pred_res = run_prediction_on_image(cv_img)
    raw_b64 = encode_image_base64(cv_img)

    return {
        "letter": clean_letter,
        "filename": sample_path.name,
        "raw_image": raw_b64,
        "prediction": pred_res
    }
