"""
Interactive ISL Data Collection Tool.

Allows collecting new hand landmark samples from webcam with MediaPipe:
- Choose letter (A-Z) or custom gesture label
- Specify number of samples (e.g. 50-100)
- Real-time countdown & visual feedback
- Appends clean normalized coordinates directly to CSV / interim dataset.

Usage:
    python scripts/collect_data.py --label A --samples 60
"""
import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("KERAS_HOME", str(ROOT / ".keras"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

sys.path.insert(0, str(ROOT))
from isl.config import DATA_DIR
from isl.hands import HandDetector, draw_hands

COORD_COLS = [f"h{h}_{j}_{a}" for h in range(2) for j in range(21) for a in "xyz"]
META_COLS = ["source", "label", "file", "frame_idx", "img_w", "img_h", "n_hands",
             "h0_label", "h0_score", "h1_label", "h1_score"]

def main():
    parser = argparse.ArgumentParser(description="Collect custom ISL sign landmark samples")
    parser.add_argument("--label", type=str, default="A", help="Sign label (e.g. A-Z, Namaste, Hello)")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to capture")
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "custom_landmarks.csv"), help="Output CSV path")
    args = parser.parse_args()

    label = args.label.upper().strip()
    target_samples = args.samples
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"ISL Data Collector: Recording '{label}' ({target_samples} samples)")
    print("Hold your hand sign steady in front of the camera.")
    print("Press [SPACE] to start recording.")
    print("Press [Q] or [ESC] to cancel.")
    print("=" * 60)

    detector = HandDetector(static_image_mode=False)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    recording = False
    captured = 0
    collected_rows = []
    countdown = 3
    start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        hands = detector.detect_bgr(frame)
        annotated = draw_hands(frame, hands)

        # UI Overlays
        cv2.rectangle(annotated, (0, 0), (w, 60), (24, 20, 18), -1)
        cv2.putText(annotated, f"Sign: '{label}' | Hands seen: {len(hands)}", (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (180, 210, 30), 2)

        if not recording:
            cv2.putText(annotated, "Press [SPACE] to start recording", (20, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 240), 2)
        else:
            if countdown > 0:
                elapsed = time.time() - start_time
                if elapsed >= 1.0:
                    countdown -= 1
                    start_time = time.time()
                cv2.putText(annotated, f"Starting in {countdown}...", (w // 2 - 120, h // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
            else:
                # Capture frame if hands detected
                if hands and captured < target_samples:
                    hands_sorted = sorted(hands, key=lambda hand: hand.wrist_x)[:2]
                    row = {
                        "source": "Custom_Live",
                        "label": label,
                        "file": f"custom_{label}_{captured:04d}.jpg",
                        "frame_idx": captured,
                        "img_w": w,
                        "img_h": h,
                        "n_hands": len(hands_sorted),
                        "h0_label": hands_sorted[0].label if len(hands_sorted) > 0 else "",
                        "h0_score": round(hands_sorted[0].score, 4) if len(hands_sorted) > 0 else 0.0,
                        "h1_label": hands_sorted[1].label if len(hands_sorted) > 1 else "",
                        "h1_score": round(hands_sorted[1].score, 4) if len(hands_sorted) > 1 else 0.0,
                    }
                    coords = np.zeros((2, 21, 3), dtype=np.float32)
                    for i, hand_obj in enumerate(hands_sorted):
                        coords[i] = hand_obj.landmarks
                    flat = coords.reshape(-1)
                    row.update({c: float(v) for c, v in zip(COORD_COLS, flat)})
                    collected_rows.append(row)
                    captured += 1
                    time.sleep(0.04)  # Small gap so frames are slightly distinct

                # Progress bar
                prog = captured / target_samples
                cv2.rectangle(annotated, (0, h - 20), (w, h), (30, 30, 30), -1)
                cv2.rectangle(annotated, (0, h - 20), (int(w * prog), h), (80, 200, 100), -1)
                cv2.putText(annotated, f"Captured: {captured}/{target_samples}", (20, h - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 200, 100), 2)

                if captured >= target_samples:
                    print(f"Captured {captured} samples successfully!")
                    break

        cv2.imshow("ISL Data Collection", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key == 32 and not recording:
            recording = True
            start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

    if collected_rows:
        df_new = pd.DataFrame(collected_rows)
        if out_path.exists():
            df_existing = pd.read_csv(out_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(out_path, index=False)
            print(f"Appended {len(df_new)} rows to {out_path} (Total: {len(df_combined)} rows).")
        else:
            df_new.to_csv(out_path, index=False)
            print(f"Saved {len(df_new)} rows to {out_path}.")

if __name__ == "__main__":
    main()
