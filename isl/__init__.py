"""ISL Gesture-to-Speech: Indian Sign Language recognition, translation and voice."""
import os
from pathlib import Path

# Set environment directories locally so neither Keras nor Matplotlib run into sandbox permissions
_base_dir = Path(__file__).resolve().parent.parent
os.environ.setdefault("KERAS_HOME", str(_base_dir / ".keras"))
os.environ.setdefault("MPLCONFIGDIR", str(_base_dir / ".matplotlib"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

__version__ = "2.1.0"
