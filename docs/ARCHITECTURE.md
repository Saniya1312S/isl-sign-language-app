# Architecture and design decisions

```
 Browser camera ──WebRTC──▶ SignVideoProcessor.process_bgr()
                                 │
                     isl/hands.py  (MediaPipe Hands → 21 landmarks per hand)
                                 │
                     isl/features.py  (order hands · normalise · engineer · 171 numbers)
                                 │
                     isl/predictor.py (scikit-learn Pipeline → letter + confidence)
                                 │
                     isl/spelling.py  (smooth · hold · re-arm → typed text)
                         │                          │
              isl/translate.py              isl/assistant.py (Gemini)
                         │                          │
                     isl/voice.py  (browser speech · gTTS)
```

## Module map

| File | Responsibility | Why it is separate |
|---|---|---|
| `isl/config.py` | paths and constants | no absolute `D:\...` paths anywhere |
| `isl/hands.py` | MediaPipe wrapper | the only file that touches MediaPipe |
| `isl/features.py` | landmarks → feature vector | identical code for training and live use (no train/serve skew) |
| `isl/model_io.py` | save/load model + JSON metadata | a model always travels with the feature mode it needs |
| `isl/predictor.py` | frame → prediction | applies the confidence threshold |
| `isl/spelling.py` | per-frame predictions → typed text | pure Python, thread-safe, fully unit-tested |
| `isl/live.py` | WebRTC video processor | `process_bgr()` is a plain function, testable without a browser |
| `isl/translate.py`, `vocab.py` | offline glossary + online translation | explicit “unavailable” status instead of fake text |
| `isl/voice.py` | browser speech + gTTS | works on a cloud server, unlike `pyttsx3` |
| `isl/assistant.py` | Gemini wrapper | key never in code; injectable client for tests |
| `views/*.py` | Streamlit pages | UI only; all logic lives in `isl/` |
| `scripts/*.py` | the data-science pipeline | numbered steps, each writes a file in `results/` |

## Key design decisions (and the problem each one solves)

1. **Browser camera (WebRTC) instead of `cv2.VideoCapture(0)`.**
   OpenCV opens a camera on the *server*. That fails on any deployment, fails when another program holds the camera, and needed a blocking
   `while` loop inside Streamlit. WebRTC streams the user's own camera to the app; the page never freezes. Snapshot and upload modes are
   fall-backs when WebRTC cannot connect.
2. **Normalised, position-sorted features instead of raw coordinates with Left/Right slots.**
   Raw coordinates change when the signer moves or the hand is closer to the camera. MediaPipe's Left/Right label depends on whether the
   image is mirrored, so the same sign could fill different columns. The new features describe only hand *shape*, order hands by position,
   and training uses mirror augmentation. (Measured: the old model falls to ~1% when the label convention flips.)
3. **Three evaluation protocols.** Consecutive video frames are near-identical, so a random split leaks. Evaluation uses held-out
   *blocks* of frames and a completely different external dataset.
4. **Smoothing + hold time + re-arm** (`spelling.py`) so a flickering classifier does not type garbage.
5. **Pinned dependencies.** `mediapipe==0.10.21` is the last release that bundles its hand models and keeps `mp.solutions`.
   An unpinned install today gets a newer MediaPipe (checked: 0.10.33 and 1.2.0) where `mp.solutions` no longer exists.
6. **Secrets handling.** The Gemini key comes from the sidebar, Streamlit secrets or an environment variable and is never written to disk.
7. **Results are files, the dashboard only reads them.** Every number on the dashboard can be reproduced by a script.
