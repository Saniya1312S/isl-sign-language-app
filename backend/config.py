"""
Backend Configuration for ISL Recognition Full-Stack Web Application.
"""
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if present
ENV_PATH = ROOT_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=True)

MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "final_isl_model.joblib"
ALT_MODEL_PATH = MODELS_DIR / "alphabet_model.pkl"
META_PATH = MODELS_DIR / "model_metadata.json"

SAMPLES_DIR = ROOT_DIR / "samples"
ALPHABET_SAMPLES_DIR = SAMPLES_DIR / "indian_isl_alphabet"

FRONTEND_DIR = ROOT_DIR / "frontend"

# Server configuration
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))


def get_gemini_api_key() -> str:
    """Retrieve Gemini API key from environment dynamically."""
    return os.getenv("GEMINI_API_KEY", "").strip()


DEFAULT_GEMINI_API_KEY = get_gemini_api_key()

