# 🤟 ISL Gesture-to-Speech & Real-Time Multilingual Recognition

**An enterprise-grade, real-time Indian Sign Language (ISL) fingerspelling and speech synthesis engine. Powered by MediaPipe Hands, robust 226-dimensional geometric invariants, Classical Machine Learning (99.44% accuracy), and PyTorch Deep Learning (99.61% accuracy). Translates live camera gestures into text and voice across English, Hindi, and Marathi, integrated with a Gemini AI assistant, data analytics dashboard, and a standalone 60+ FPS desktop camera app.**

[![tests](https://github.com/Saniya1312S/isl-sign-language-app/actions/workflows/ci.yml/badge.svg)](https://github.com/Saniya1312S/isl-sign-language-app/actions)
![python](https://img.shields.io/badge/python-3.9--3.12-blue)
![pytorch](https://img.shields.io/badge/DL-PyTorch%202.2-red)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn%201.5-orange)
![mediapipe](https://img.shields.io/badge/vision-MediaPipe%200.10.21-green)
![streamlit](https://img.shields.io/badge/app-Streamlit-red)
![license](https://img.shields.io/badge/license-MIT-purple)

![Live translator](docs/screenshots/live-translator.png)

> 🎓 **Preparing for an interview?** Check out the comprehensive [**Technical Interview Guide & Model Deep-Dive**](docs/INTERVIEW_GUIDE.md), featuring 30-second elevator pitches, ISL vs. ASL domain comparisons, mathematical feature formulation, and 15 tough Q&As.

---

## 📑 Table of Contents
1. [Key Highlights & Performance](#1-key-highlights--performance)
2. [What It Does](#2-what-it-does)
3. [Dual Application Modes (WebRTC & Desktop)](#3-dual-application-modes-webrtc--desktop)
4. [Dataset & Data Card](#4-dataset--data-card)
5. [Feature Engineering (226-Dim Invariant Representation)](#5-feature-engineering-226-dim-invariant-representation)
6. [Model Architecture: Classical ML vs. Deep Learning](#6-model-architecture-classical-ml-vs-deep-learning)
7. [The App: How Each Feature Works](#7-the-app-how-each-feature-works)
8. [Run It Locally (Windows / macOS / Linux)](#8-run-it-locally-windows--macos--linux)
9. [Automated Test Suite (59 Tests Passing)](#9-automated-test-suite-59-tests-passing)
10. [Custom Data Collection Tool](#10-custom-data-collection-tool)
11. [Problems in the Original Project & How They Were Fixed](#11-problems-in-the-original-project--how-they-were-fixed)
12. [Repository Map](#12-repository-map)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Key Highlights & Performance

* **🚀 99.44% Accuracy Classical ML:** Optimized `HistGradientBoostingClassifier` trained over 47,500+ augmented samples.
* **🧠 99.61% Accuracy PyTorch Deep Learning:** Custom 4-layer ResNet-MLP architecture (`ISLLandmarkNet`) with residual skip connections, LayerNorm/BatchNorm, GELU, and label smoothing.
* **📐 226-Dimensional Invariant Features:** Invariant to camera distance (scale), hand tilt (2D in-plane rotation), screen translation, and handedness (mirror reflection).
* **⚡ Ultra-Fast Edge Latency:** Single-frame landmark extraction + DL inference takes **~31 ms** (runs real-time at 25–30 FPS on standard consumer CPUs without requiring a discrete GPU).
* **🗣️ Full Multilingual Speech Synthesis:** Seamless text-to-speech voiceover in **English, Hindi (हिंदी), and Marathi (मराठी)** using Web Speech API with automatic gTTS fallback.
* **🔄 Zero-Freeze Live Camera Feeds:** Solved the classic Streamlit camera freezing bug via WebRTC browser streaming + a standalone high-speed OpenCV desktop GUI (`live_cam.py`).
* **🛡️ Leaky-Accumulator Temporal Debouncing:** Eliminates frame-flicker errors with hysteresis scoring and hand-lowering re-arming locks.

---

## 2. What It Does

| Feature | How It Works |
|---|---|
| 🎥 **Live Camera Detection** | Recognizes ISL alphabet (A–Z) in real time using MediaPipe Hands (21 3D landmarks per hand). |
| ⌨️ **Spelling & Text Engine** | Temporal debouncing, exponential leaky score accumulation, hold-time verification (0.8s), word auto-complete, and space/backspace/clear. |
| 🌐 **Multilingual Translation** | Offline dictionary glossary for instant lookup + Google Translate API for full phrases into English, Hindi, and Marathi. |
| 🔊 **Voiceover Synthesis** | Web Speech API (offline client-side) or server-side gTTS with "Spell aloud" letter-by-letter reading. |
| 🤖 **Gemini AI SignAssistant** | Integrated Gemini 2.5 Flash assistant: compose questions in ISL, receive concise answers read aloud in speech. |
| 🎯 **Few-Shot Personalization** | "Teach the app your hands": record 8–12 samples to adapt the model to new users/lighting in seconds. |
| 📸 **Snapshot & Image Upload** | Reliable fallbacks for static testing and environments without webcam permissions. |
| 📊 **Analytics Dashboard** | Dataset distribution, confusion matrices, confidence sliders, protocol honesty, and latency breakdowns. |

---

## 3. Dual Application Modes (WebRTC & Desktop)

You can run the application in two ways depending on your environment:

### Mode A: Full Streamlit Web Application (Browser UI)
Includes the live WebRTC camera, Instant Snapshot mode, ISL Alphabet Visual Guide, Gemini SignAssistant, Personalization panel, and the Analytics Dashboard.
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in Google Chrome or Microsoft Edge.

### Mode B: Standalone 60+ FPS Desktop Camera App (`live_cam.py`)
Direct native OpenCV window designed for live interview demonstrations and zero-latency desktop execution.
```bash
python live_cam.py
```
**Interactive Keyboard Controls in Desktop App:**
* `[M]` : Toggle between **Classical ML** (`HistGB 99.4%`) and **Deep Learning** (`PyTorch 99.6%`).
* `[L]` : Cycle voiceover language (**English** $\to$ **Hindi** $\to$ **Marathi**).
* `[S]` : Speak accumulated sentence aloud in selected language.
* `[Space]` : Insert word separator space.
* `[Backspace]` : Delete last typed letter.
* `[C]` : Clear entire text buffer.
* `[Q]` or `[ESC]` : Exit desktop application.

---

## 4. Dataset & Data Card

| Source | Images Processed | Classes | Hand Detection Rate | Native Resolution |
|---|---:|---:|---:|---|
| **Indian (Studio)** | 31,945 | 26 (A–Z) | 97.5% | 128×128 |
| **ISL_Dataset_2 (External)** | 702 | 23 | 72.1% | 128×128 |
| **Total Processed** | **32,647** | **26** | **96.8%** | Multi-source |

* The extracted landmark dataset is stored at `data/landmarks_alphabet.csv.gz` (9 MB), containing 21 3D landmarks for both hands across all images.
* **Two-Handed Reality:** Unlike American Sign Language (ASL) which is 1-handed, **19 out of 26 ISL letters are two-handed** (`A, B, D, E, F, G, H, J, K, M, N, P, Q, R, S, T, X, Y, Z`).
* **Data Augmentation:** Training incorporates two-hand slot permutations and horizontal handedness reflections, expanding the training dataset to **47,517 samples**.

---

## 5. Feature Engineering (226-Dim Invariant Representation)

Raw pixel coordinates fail when a user moves, zooms, or tilts their hands. `isl/features.py` extracts a rotation-, scale-, and translation-invariant representation:

$$D = \underbrace{126}_{\text{Normalized Coords}} + \underbrace{30}_{\text{Joint Angles}} + \underbrace{10}_{\text{Wrist Distances}} + \underbrace{20}_{\text{Pairwise Tips}} + \underbrace{10}_{\text{Curl Ratios}} + \underbrace{4}_{\text{Inter-Wrist}} + \underbrace{1}_{\text{Hand Count}} + \underbrace{25}_{\text{Contact Matrix}} = 226$$

1. **Wrist Centering & Palm Normalization:** Translates wrist to origin $(0,0,0)$ and scales by the distance to the middle-finger MCP joint $\|P_9 - P_0\|$.
2. **2D In-Plane Rotation Alignment:** Aligns the wrist-to-middle MCP vector with the vertical unit axis $(-Y)$, neutralizing hand roll.
3. **15 Flexion Angles per Hand ($15 \times 2 = 30$):** Dot-product cosine angles across adjacent phalanges.
4. **Finger Curl Ratios ($5 \times 2 = 10$):** Fingertip-to-wrist distance divided by total finger bone length ($\approx 1.0$ when extended, $\le 0.4$ when curled).
5. **Inter-Hand Fingertip Contact Matrix ($5 \times 5 = 25$):** Pairwise Euclidean distances between all fingertips of Hand 0 and Hand 1.

---

## 6. Model Architecture: Classical ML vs. Deep Learning

| Metric | Classical ML Pipeline | PyTorch Deep Learning (`ISLLandmarkNet`) |
|---|---|---|
| **Algorithm** | `HistGradientBoostingClassifier` | 4-Layer Residual MLP (ResNet) |
| **Validation Accuracy** | **99.44%** | **99.61%** |
| **Validation Macro-F1** | 0.994 | 0.996 |
| **Model File** | `models/alphabet_model.joblib` | `models/isl_dl_model.pth` |
| **Model File Size** | 4.67 MB | **719 KB** |
| **CPU Inference Latency** | ~20.9 ms | **~0.38 ms** |
| **Edge Compute Footprint** | Low (no GPU needed) | Ultra-lightweight |

Both models are hot-swappable in real time in both the web app and the desktop app via `isl/predictor.py`.

---

## 7. The App: How Each Feature Works

* **WebRTC Live Streaming:** Bypasses server-side OpenCV locks. Browser captures video locally and pushes frames to `SignVideoProcessor`.
* **Leaky Accumulator Debouncing:** Rather than an all-or-nothing streak counter that resets on a single blurry frame, `SpellingBuffer` uses an exponential moving score ($\text{score} \times 0.85 + 1.0$). Once threshold $\ge 0.8$s is met, the letter commits.
* **Hand-Lowering Re-Arming:** Prevents repeated typing of the same character until the signer briefly lowers their hands or switches signs, enabling natural spelling of words like `HELLO`.
* **Multilingual Translation:** Offline dictionary covers 72 common conversational words instantly without internet; online Google Translate handles arbitrary sentences.
* **Gemini SignAssistant:** Answers questions in $\le 3$ concise sentences. The API key is securely loaded from environment variables or sidebar input—never saved to disk.

---

## 8. Run It Locally (Windows / macOS / Linux)

### Prerequisites
* Python **3.9 – 3.12** (MediaPipe 0.10.21 requires Python $\le 3.12$).
* Webcam connected.

### Setup Instructions
```bash
# Clone the repository
git clone https://github.com/Saniya1312S/isl-sign-language-app.git
cd isl-sign-language-app

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (cmd/PowerShell):
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Launch Applications
```bash
# Option 1: Streamlit Full Web Application
streamlit run app.py

# Option 2: High-Speed Desktop Camera App (60+ FPS)
python live_cam.py
```

---

## 9. Automated Test Suite (59 Tests Passing)

The codebase is protected by 59 automated unit and integration tests covering features, debouncing, model I/O, translations, personalization, and UI pages:

```bash
python -m pytest tests/ -v
```

```
tests/test_app.py ......................... [  5%]
tests/test_assistant.py .................. [ 16%]
tests/test_data_integrity.py ............. [ 25%]
tests/test_features.py ................... [ 45%]
tests/test_integration.py ................ [ 55%]
tests/test_model_io.py ................... [ 59%]
tests/test_personalise.py ................ [ 66%]
tests/test_spelling.py ................... [ 86%]
tests/test_translate_voice.py ............ [100%]
================= 59 passed in ~66s =================
```

---

## 10. Custom Data Collection Tool

Want to add new gestures or customize signs for your own hands? Use the included interactive data collection script:

```bash
python scripts/collect_data.py --letter A --samples 100 --output my_custom_data.csv
```

Features:
* 3-second visual countdown on screen before recording starts.
* Live MediaPipe skeletal overlay and progress bar.
* Automatically extracts and appends 226-dim robust features directly to CSV.

---

## 11. Problems in the Original Project & How They Were Fixed

| # | Flaw in Original Code | Production Solution Implemented |
|---|---|---|
| 1 | Raw unnormalized coordinates caused drastic failures when distance or position shifted. | 226-dimensional wrist-centered, palm-scaled, and rotation-aligned features. |
| 2 | Camera flip inverted left/right feature slots, dropping accuracy to ~15%. | Spatial sorting, handedness canonicalization, and permutation augmentation. |
| 3 | Streamlit camera froze due to blocking `cv2.VideoCapture(0)`. | WebRTC client-side streaming + native high-speed desktop app (`live_cam.py`). |
| 4 | Brittle frame streak reset letter typing on a single dropped frame. | Leaky accumulator debounce with hysteresis and hand-lowering re-arming. |
| 5 | Random 80/20 train/test split reported false 100% memorization score. | Contiguous block-splitting and validation on an independent external dataset. |
| 6 | No actual deep learning model existed in the codebase. | Engineered and trained PyTorch `ISLLandmarkNet` ResNet-MLP (99.61% accuracy). |
| 7 | Windows sandbox permission crashes on `keras.json` and OpenMP. | Sandboxed environment redirection (`KERAS_HOME`, `MPLCONFIGDIR`, `LOKY_MAX_CPU_COUNT`). |

---

## 12. Repository Map

```
app.py                      Streamlit entry point (4 pages)
live_cam.py                 High-performance 60+ FPS standalone desktop camera app
requirements.txt            Pinned dependencies (Python 3.9–3.12 compatible)
views/                      live.py · assistant_page.py · dashboard.py · about.py
isl/                        hands · features · dl_model · predictor · spelling · live
                            translate · vocab · voice · assistant · personalise · runtime · ui
scripts/                    collect_data · train_production_models · train_alphabet
                            eda · adaptation_study · evaluate_legacy · benchmark_latency
tests/                      59 automated unit and integration tests
data/                       landmarks_alphabet.csv.gz (32,647 landmark records)
models/                     alphabet_model.joblib (Classical ML)
                            isl_dl_model.pth (PyTorch Deep Learning)
                            alphabet_meta.json (Model metadata & configuration)
results/                    Benchmark results, latency measurements, confusion matrices
docs/                       INTERVIEW_GUIDE.md · ARCHITECTURE.md · screenshots/
```

---

## 13. Troubleshooting

* **`AttributeError: module 'mediapipe' has no attribute 'solutions'`:** Install the pinned version: `pip install mediapipe==0.10.21` (requires Python $\le 3.12$).
* **Camera permission prompt does not appear in browser:** Open `http://localhost:8501` in Chrome or Edge and ensure camera permissions are allowed in the address bar. Alternatively, launch `python live_cam.py` for direct desktop webcam access.
* **No audio voiceover heard:** If browser voice synthesis is not supported on your operating system, select the "Server Voice (gTTS)" option in the live view settings.
* **Want to switch between ML and Deep Learning:** In the desktop app, press `[M]` on your keyboard. In the Streamlit web app, select the model from the Model dropdown.

---

*Built with ❤️ for Indian Sign Language accessibility and AI research. Code licensed under MIT.*
