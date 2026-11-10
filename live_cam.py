"""
Live Real-Time ISL (Indian Sign Language) Desktop Camera Application.

Ultra-low latency (60+ FPS), zero-lag, dedicated desktop camera feed with:
- MediaPipe hand tracking & 21 skeletal landmarks
- SOTA Invariant Feature Extraction & ML/DL Inference
- Smooth Leaky Debounce Spelling Buffer
- Real-time Multilingual Translation (English, Hindi, Marathi)
- Background Audio Speech Synthesis (Voiceover)
- Interactive Hotkeys & Modern HUD

Usage:
    python live_cam.py
"""
import os
import sys
import time
import threading
from pathlib import Path
import cv2
import numpy as np

# Safe environment paths
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("KERAS_HOME", str(ROOT / ".keras"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

sys.path.insert(0, str(ROOT))
from isl.config import MODELS_DIR, ALPHABET
from isl.hands import HandDetector, draw_hands
from isl.predictor import SignPredictor
from isl.spelling import SpellingBuffer
from isl.translate import LANGS, translate
from isl.runtime import load_bundles

# Color palette (BGR)
BG_DARK = (24, 20, 18)
CARD_BG = (38, 32, 28)
TEAL = (180, 210, 30)
CYAN = (240, 200, 40)
GREEN = (80, 200, 100)
YELLOW = (50, 200, 240)
WHITE = (250, 250, 250)
GRAY = (140, 130, 120)
RED = (60, 60, 240)
AMBER = (20, 150, 255)

def play_speech_async(text: str, lang_code: str = "en"):
    """Speak text in a non-blocking daemon thread so the video loop never stutters."""
    def _worker():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 145)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            # Fallback to gTTS if pyttsx3 fails or language is Hindi/Marathi
            try:
                from gtts import gTTS
                import tempfile
                import playsound
                tts = gTTS(text=text, lang=lang_code)
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                    temp_path = fp.name
                tts.save(temp_path)
                playsound.playsound(temp_path)
                os.remove(temp_path)
            except Exception:
                pass
    t = threading.Thread(target=_worker, daemon=True)
    t.start()

def main():
    print("=" * 60)
    print("Starting Indian Sign Language (ISL) Real-Time Camera")
    print("Hotkeys:")
    print("  [SPACE]     : Add space to sentence")
    print("  [BACKSPACE] : Delete last letter")
    print("  [C]         : Clear sentence")
    print("  [S]         : Speak aloud")
    print("  [L]         : Switch language (English -> Hindi -> Marathi)")
    print("  [M]         : Toggle Model (Classical ML <-> PyTorch DL)")
    print("  [Q] / [ESC] : Quit")
    print("=" * 60)

    # 1. Load Model Bundles
    bundles = load_bundles()
    ml_predictor = None
    dl_predictor = None
    
    if "alphabet" in bundles:
        ml_predictor = SignPredictor(bundles["alphabet"], min_confidence=0.55)
    
    dl_path = MODELS_DIR / "isl_dl_model.pth"
    if dl_path.exists():
        try:
            dl_predictor = SignPredictor(dl_path, min_confidence=0.55)
        except Exception as e:
            print("Notice: DL model could not be loaded yet:", e)

    current_model_type = "DL (ResNet-MLP)" if dl_predictor else "ML (HistGB)"
    active_predictor = dl_predictor if dl_predictor else ml_predictor

    if not active_predictor:
        print("Error: No models found in models/ directory. Run `python scripts/train_production_models.py` first.")
        return

    # 2. Camera & Detector Setup
    detector = HandDetector(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    buffer = SpellingBuffer(hold_frames=7, hold_seconds=0.6)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Warning: Could not open camera 0. Trying camera 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: No camera could be opened.")
            return

    languages = ["English", "Hindi", "Marathi"]
    lang_idx = 0
    current_lang = languages[lang_idx]
    last_translation = ""
    last_translated_word = ""

    fps_t = deque_time = time.time()
    fps = 30.0

    cv2.namedWindow("ISL Real-Time Translator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ISL Real-Time Translator", 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(EPS, now - fps_t)) if (now - fps_t) > 0 else fps
        fps_t = now

        # Mirror view for natural interaction
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Detect hands
        hands = detector.detect_bgr(frame)
        annotated = draw_hands(frame, hands)

        # Predict sign
        pred = active_predictor.predict_hands(hands)
        typed_letter = buffer.update(pred.label)
        if typed_letter:
            play_speech_async(typed_letter, "en")

        snap = buffer.snapshot()
        current_text = snap.text

        # Update translation if new word formed
        if current_lang != "English" and current_text.strip():
            if current_text.strip() != last_translated_word:
                tr = translate(current_text.strip(), LANGS[current_lang])
                if tr and tr.text:
                    last_translation = tr.text
                    last_translated_word = current_text.strip()

        # ---------------- HUD Overlays ----------------
        # 1. Top Header Bar
        cv2.rectangle(annotated, (0, 0), (w, 54), BG_DARK, -1)
        cv2.putText(annotated, "ISL Gesture-to-Speech", (20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.9, TEAL, 2, cv2.LINE_AA)
        
        # Badges
        cv2.putText(annotated, f"Model: {current_model_type}", (380, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, CYAN, 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Lang: {current_lang}", (660, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, YELLOW, 2, cv2.LINE_AA)
        cv2.putText(annotated, f"{fps:.1f} FPS", (w - 120, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)

        # 2. Left Prediction Card
        card_w, card_h = 320, 220
        cv2.rectangle(annotated, (20, 70), (20 + card_w, 70 + card_h), CARD_BG, -1)
        cv2.rectangle(annotated, (20, 70), (20 + card_w, 70 + card_h), GRAY, 1)
        
        if pred.label:
            cv2.putText(annotated, f"{pred.label}", (40, 160), cv2.FONT_HERSHEY_DUPLEX, 3.2, TEAL, 5, cv2.LINE_AA)
            cv2.putText(annotated, f"{pred.confidence:.0%}", (180, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 2, cv2.LINE_AA)
            
            # Progress bar for holding sign
            bar_y = 185
            cv2.rectangle(annotated, (40, bar_y), (40 + 240, bar_y + 16), BG_DARK, -1)
            cv2.rectangle(annotated, (40, bar_y), (40 + int(240 * snap.progress), bar_y + 16), GREEN, -1)
            cv2.putText(annotated, "HOLD TO TYPE", (40, bar_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1, cv2.LINE_AA)
            
            # Top 3 guesses
            if pred.top_k:
                top_str = " | ".join(f"{l}: {p:.0%}" for l, p in pred.top_k)
                cv2.putText(annotated, top_str, (40, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        else:
            msg = "Show Hand Gesture" if pred.reason == "no_hand" else "Hold Steady..."
            cv2.putText(annotated, msg, (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, AMBER, 2, cv2.LINE_AA)

        # 3. Bottom Subtitle / Sentence Panel
        panel_h = 100
        cv2.rectangle(annotated, (0, h - panel_h), (w, h), BG_DARK, -1)
        
        shown_text = current_text if current_text else "Sign letters to construct words..."
        cv2.putText(annotated, f"Text: {shown_text}", (24, h - panel_h + 38), cv2.FONT_HERSHEY_DUPLEX, 0.95, WHITE, 2, cv2.LINE_AA)
        
        if current_lang != "English" and last_translation:
            cv2.putText(annotated, f"{current_lang}: {last_translation}", (24, h - panel_h + 76), cv2.FONT_HERSHEY_DUPLEX, 0.85, YELLOW, 2, cv2.LINE_AA)
        else:
            cv2.putText(annotated, "[SPACE] Space | [BACKSPACE] Delete | [C] Clear | [S] Speak | [L] Language | [M] Model | [Q] Quit",
                        (24, h - panel_h + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GRAY, 1, cv2.LINE_AA)

        cv2.imshow("ISL Real-Time Translator", annotated)

        # ---------------- Keyboard Controls ----------------
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key == 32:  # Space
            buffer.add_space()
        elif key in (8, 127):  # Backspace
            buffer.backspace()
        elif key in (ord('c'), ord('C')):
            buffer.clear()
            last_translation = ""
        elif key in (ord('s'), ord('S')):
            text_to_speak = last_translation if (current_lang != "English" and last_translation) else buffer.text.strip()
            if text_to_speak:
                play_speech_async(text_to_speak, LANGS.get(current_lang, "en"))
        elif key in (ord('l'), ord('L')):
            lang_idx = (lang_idx + 1) % len(languages)
            current_lang = languages[lang_idx]
            last_translation = ""
            print(f"Language set to: {current_lang}")
        elif key in (ord('m'), ord('M')):
            if active_predictor == dl_predictor and ml_predictor:
                active_predictor = ml_predictor
                current_model_type = "ML (HistGB)"
            elif dl_predictor:
                active_predictor = dl_predictor
                current_model_type = "DL (ResNet-MLP)"
            print(f"Active Model switched to: {current_model_type}")

    cap.release()
    cv2.destroyAllWindows()
    print("ISL Camera application closed cleanly.")

if __name__ == "__main__":
    main()
