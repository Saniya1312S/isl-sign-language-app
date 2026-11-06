"""Central configuration: paths and constants. No absolute paths anywhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
SAMPLES_DIR = ROOT / "samples"
DOCS_DIR = ROOT / "docs"

LANDMARKS_CSV = DATA_DIR / "landmarks_alphabet.csv.gz"
ALPHABET_MODEL = MODELS_DIR / "alphabet_model.joblib"
ALPHABET_META = MODELS_DIR / "alphabet_meta.json"
WORDS_MODEL = MODELS_DIR / "words_model.joblib"
WORDS_META = MODELS_DIR / "words_meta.json"
HINDI_MODEL = MODELS_DIR / "hindi_model.joblib"
HINDI_META = MODELS_DIR / "hindi_meta.json"

RANDOM_STATE = 42
N_LANDMARKS = 21                 # MediaPipe Hands returns 21 landmarks per hand
MAX_HANDS = 2

# Class labels used in the app
ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Live recognition behaviour (all tunable from the UI)
DEFAULT_MIN_CONFIDENCE = 0.60    # a prediction below this is treated as "not sure"
DEFAULT_HOLD_FRAMES = 12         # frames a letter must stay stable to be typed
