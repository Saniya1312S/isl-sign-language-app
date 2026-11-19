"""
Main FastAPI Application Entry Point.
Serves both the REST API endpoints and static frontend files.

Run with:
    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.recognition import router as recognition_router
from backend.api.translation import router as translation_router
from backend.api.gemini import router as gemini_router
from backend.config import FRONTEND_DIR, SAMPLES_DIR, ALPHABET_SAMPLES_DIR
from backend.services.predictor_service import get_predictor

app = FastAPI(
    title="Indian Sign Language Recognition System",
    description="Full-stack accessible ISL alphabet fingerspelling, sentence translator, and AI assistant.",
    version="2.0.0"
)

# CORS Middleware for local frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(recognition_router)
app.include_router(translation_router)
app.include_router(gemini_router)


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint confirming model status, features, and endpoints."""
    predictor = get_predictor()
    meta = predictor.metadata
    return {
        "status": "healthy",
        "app_name": "ISL Sign Language Recognition Website",
        "model_name": meta.get("model_name", "Support Vector Machine"),
        "feature_dimensions": meta.get("n_features", 171),
        "classes_count": len(predictor.classes),
        "two_hand_supported": True,
        "classes": predictor.classes
    }


# Mount sample images for direct browser image viewing
if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")

# Mount frontend assets
if (FRONTEND_DIR / "css").exists():
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
if (FRONTEND_DIR / "js").exists():
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")
if (FRONTEND_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")


@app.get("/")
def serve_index():
    """Serve frontend index.html."""
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
