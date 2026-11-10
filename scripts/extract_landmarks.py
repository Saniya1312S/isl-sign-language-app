"""
STEP 1 - Turn images into hand-landmark tables (resumable).

For every image we run MediaPipe Hands once and save the raw landmarks. The two hands are
stored in a fixed order: sorted by wrist x-position (image-left first). We do NOT rely on
MediaPipe's Left/Right label for the order, but we keep the label so the *original*
project's feature convention can still be reproduced for comparison.

Output (one CSV, one row per image):
    source, label, file, frame_idx, img_w, img_h, n_hands,
    h0_label, h0_score, h1_label, h1_score,
    h0_0_x, h0_0_y, h0_0_z, ... h1_20_z          (2 hands x 21 landmarks x 3 = 126 numbers)

Usage (example - folder names as in the original project):
    python scripts/extract_landmarks.py \
        --source Indian=Data/alphabet/Indian \
        --source ISL_Dataset_2=Data/alphabet/ISL_Dataset_2

The script caches every class in data/interim/ so you can stop and re-run it at any time.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isl.config import DATA_DIR, LANDMARKS_CSV                       # noqa: E402
from isl.hands import HandDetector                                    # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png")
COORD_COLS = [f"h{h}_{j}_{a}" for h in range(2) for j in range(21) for a in "xyz"]
META_COLS = ["source", "label", "file", "frame_idx", "img_w", "img_h", "n_hands",
             "h0_label", "h0_score", "h1_label", "h1_score"]


def frame_index(name: str) -> int:
    """'123.jpg' -> 123,  'A (12).jpg' -> 12,  otherwise -1."""
    nums = re.findall(r"\d+", Path(name).stem)
    return int(nums[-1]) if nums else -1


def row_for_image(path: Path, detector: HandDetector, source: str, label: str):
    img = cv2.imread(str(path))
    if img is None:
        return None
    hands = sorted(detector.detect_bgr(img), key=lambda h: h.wrist_x)[:2]
    row = {"source": source, "label": label, "file": path.name, "frame_idx": frame_index(path.name),
           "img_w": img.shape[1], "img_h": img.shape[0], "n_hands": len(hands),
           "h0_label": "", "h0_score": 0.0, "h1_label": "", "h1_score": 0.0}
    coords = np.zeros((2, 21, 3), dtype=np.float32)
    for i, h in enumerate(hands):
        coords[i] = h.landmarks
        row[f"h{i}_label"] = h.label
        row[f"h{i}_score"] = round(h.score, 4)
    flat = coords.reshape(-1)
    row.update({c: float(v) for c, v in zip(COORD_COLS, flat)})
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True,
                    help="NAME=FOLDER where FOLDER contains one sub-folder per class (repeatable)")
    ap.add_argument("--out", default=str(LANDMARKS_CSV))
    ap.add_argument("--cache", default=str(DATA_DIR / "interim"))
    ap.add_argument("--limit-per-class", type=int, default=0, help="0 = use every image")
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="stop after finishing the current class once this many seconds passed (0 = no limit)")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    todo = []
    for spec in args.source:
        name, folder = spec.split("=", 1)
        for cdir in sorted(p for p in Path(folder).iterdir() if p.is_dir()):
            todo.append((name, cdir.name, cdir))

    detector = HandDetector(static_image_mode=True, min_detection_confidence=0.5)
    done = skipped = 0
    for name, label, cdir in todo:
        cfile = cache / f"{name}__{label}.pkl"
        if cfile.exists():
            skipped += 1
            continue
        if args.max_seconds and time.time() - t0 > args.max_seconds:
            print(f"[time limit] stopping; re-run the same command to continue "
                  f"({len(todo) - done - skipped} classes left)")
            break
        files = sorted((p for p in cdir.iterdir() if p.suffix.lower() in IMG_EXT),
                       key=lambda p: (frame_index(p.name), p.name))
        if args.limit_per_class:
            files = files[: args.limit_per_class]
        rows = [r for r in (row_for_image(p, detector, name, label) for p in files) if r is not None]
        pd.DataFrame(rows, columns=META_COLS + COORD_COLS).to_pickle(cfile)
        det = sum(r["n_hands"] > 0 for r in rows)
        done += 1
        print(f"{name:14s} {label:>3s}: {len(rows):5d} images, hand found in {det / max(len(rows), 1):6.1%}"
              f"   [{time.time() - t0:5.0f}s]", flush=True)

    # merge every cached class into the final CSV once all classes are done
    if skipped + done == len(todo):
        parts = [pd.read_pickle(cache / f"{n}__{l}.pkl") for n, l, _ in todo]
        df = pd.concat(parts, ignore_index=True)
        df[COORD_COLS] = df[COORD_COLS].round(5)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False, compression="gzip")
        print(f"\nSAVED {len(df):,} rows -> {args.out}")
        print(df.groupby("source").size().to_string())
    detector.close()


if __name__ == "__main__":
    main()
