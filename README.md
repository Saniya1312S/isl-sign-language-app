# 🤟 Indian Sign Language (ISL) Fingerspelling to Speech

Real-time Indian Sign Language (ISL) fingerspelling recognition (A–Z) pipeline using MediaPipe Hands, coordinate normalization, mirror-invariant feature engineering, and scikit-learn classification. Provides live text spelling, multilingual translation (English, Hindi, Marathi), and speech synthesis.

[![tests](https://github.com/Saniya1312S/isl-sign-language-app/actions/workflows/ci.yml/badge.svg)](https://github.com/Saniya1312S/isl-sign-language-app/actions)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange)
![mediapipe](https://img.shields.io/badge/vision-MediaPipe%200.10.21-green)
![streamlit](https://img.shields.io/badge/app-Streamlit-red)
![license](https://img.shields.io/badge/license-MIT-purple)

---

## 📊 Honest Evaluation & Protocol Findings

A critical finding in video-based sign language datasets is that **random train/test splits severely leak data**: consecutive video frames from the same signer recording are near-identical, artificially inflating test scores.

| Protocol | Evaluation Setup | Accuracy | Macro-F1 | Note |
| :--- | :--- | :---: | :---: | :--- |
| **P1: Random Split** | Standard 80/20 train_test_split on video frames | **100.00%** | 100.00% | **Invalid**: Near-duplicate frame leakage |
| **P2: Block Split** | Held-out 200-frame consecutive blocks (stratified per letter) | **97.12%** | 97.49% | **Genuine**: Tests generalization to unseen clips |
| **P3: External Dataset** | Tested on `ISL_Dataset_2` (23 classes, different signers & backgrounds) | **26.09%** | 26.37% | **Domain Shift**: Realistic test across people/lighting (18.8% end-to-end) |

> 📌 **Key takeaway:** The same model architecture that achieves 100.0% on random splits achieves **97.1%** on held-out blocks and drops to **26.1%** when evaluated on external unseen signers. Reporting random split accuracy on video datasets is misleading.

---

## ⚡ Inference Latency & Performance

Measurements taken on real hand images (640x480 resolution) using MediaPipe Hands tracking mode:
- **Hand Detection (MediaPipe):** 50.0 ms median
- **Feature Engineering (`norm_eng`):** 0.59 ms median
- **Classifier Inference (Logistic Regression):** 0.60 ms median
- **Total Frame Processing Time (p50):** **53.8 ms** (~19 FPS)
- **Total Frame Processing Time (p95):** **92.4 ms**
- **Benchmark Hardware:** `Intel64 Family 6 Model 154 Stepping 3, GenuineIntel · Windows · Python 3.12.10`

---

## 🏁 Model Benchmark Comparison

All models were evaluated under identical conditions: `norm_eng` features with mirror augmentation, evaluated on held-out blocks and external unseen signers.

| Model | Features | CV Macro-F1 | Block Test Acc | External Acc | Latency (ms) | Size (KB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | norm_eng (171) | 98.4% | 96.8% | 31.4% | 0.22 ms | 41.0 KB |
| k-Nearest Neighbours | norm_eng (171) | 98.5% | 96.9% | 32.2% | 4.20 ms | 12359.7 KB |
| Random Forest | norm_eng (171) | 98.4% | 97.3% | 33.8% | 27.23 ms | 1577.0 KB |
| Extra Trees | norm_eng (171) | 98.5% | 97.0% | 36.2% | 37.88 ms | 3420.2 KB |
| Gradient Boosting (Hist) | norm_eng (171) | 96.6% | 97.0% | 24.7% | 11.39 ms | 1966.1 KB |
| Neural Network (MLP) | norm_eng (171) | 98.1% | 96.9% | 31.2% | 1.04 ms | 1228.8 KB |
| PyTorch ResNet-MLP | norm_eng (171) | 98.3% | 96.9% | 31.2% | 1.70 ms | 1443.8 KB |

### Selected Shipped Model: `Logistic Regression`
- **Selection Rule:** Highest efficiency among scikit-learn classifiers within 0.005 CV Macro-F1 of the best performing model.
- **Advantages:** Smallest file footprint (~41 KB), fastest CPU single-sample latency (0.22 ms), zero external deep learning runtime dependencies.

---

## 📐 Feature Engineering Ablation

| Feature Set | Mirror Augmentation | Dimensions | Block Test Acc | Mirrored Test Acc | External Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `robust_isl` | No | 226 | 97.0% | 72.4% | 27.9% |
| `robust_isl` | Yes | 226 | 97.0% | 97.0% | 31.6% |
| `norm_eng` | No | 171 | 97.1% | 57.3% | 26.1% |
| `norm_eng` | Yes | 171 | 97.3% | 97.5% | 32.8% |
| `norm` | No | 131 | 97.1% | 47.5% | 20.9% |
| `norm` | Yes | 131 | 96.9% | 96.9% | 25.5% |
| `sorted_raw` | No | 126 | 97.1% | 35.6% | 17.2% |
| `sorted_raw` | Yes | 126 | 94.8% | 94.8% | 16.0% |
| `legacy_raw126` | No | 126 | 97.1% | 10.2% | 15.6% |
| `legacy_raw126` | Yes | 126 | 97.1% | 97.0% | 18.0% |

**Findings:**
1. Without mirror augmentation, accuracy on horizontally flipped / selfie camera frames collapses to 10%–57% (e.g. `norm_eng` drops to 57.3%).
2. Adding mirror augmentation ensures symmetry invariance ($x \to 1-x$, Left $\leftrightarrow$ Right swap) yielding 97.3% block accuracy and 97.5% mirrored accuracy.
3. The 171-dimensional `norm_eng` feature set matches or exceeds the 226-dimensional set while saving compute and storage.

---

## 🎯 Few-Shot Personalisation (Overcoming Domain Shift)

To address the drop on unseen signers (26.1%), the app provides interactive few-shot calibration (*Teach the App Your Hands*). By capturing just a few calibration frames per letter from the target user, accuracy recovers significantly without forgetting studio data:

| Calibration Images / Letter | External Accuracy | Studio Block Accuracy |
| :---: | :---: | :---: |
| 0 | 32.3% | 96.8% |
| 1 | 41.5% | 96.8% |
| 2 | 48.5% | 98.7% |
| 3 | 49.2% | 99.2% |
| 5 | 50.4% | 97.6% |
| 8 | 56.9% | 96.4% |

---

## 🔄 Legacy vs. New Pipeline Comparison

Evaluation of the original baseline model (trained on raw landmark coordinates without coordinate normalization or mirror augmentation) versus the updated pipeline:

| Model | Dataset | Scenario | Accuracy | Sample Count |
| :--- | :--- | :--- | :---: | :---: |
| Original (raw + Left/Right slots) | Indian dataset (partly seen in training) | `as_trained` | 99.8% | 31,130 |
| New (normalised + mirror-aug) | Indian dataset (partly seen in training) | `as_trained` | 99.3% | 31,130 |
| Original (raw + Left/Right slots) | Indian dataset (partly seen in training) | `slots_swapped` | 11.6% | 31,130 |
| New (normalised + mirror-aug) | Indian dataset (partly seen in training) | `slots_swapped` | 99.3% | 31,130 |
| Original (raw + Left/Right slots) | Indian dataset (partly seen in training) | `mirrored_frame` | 20.5% | 31,130 |
| New (normalised + mirror-aug) | Indian dataset (partly seen in training) | `mirrored_frame` | 99.4% | 31,130 |
| Original (raw + Left/Right slots) | External dataset (unseen) | `as_trained` | 45.3% | 506 |
| New (normalised + mirror-aug) | External dataset (unseen) | `as_trained` | 31.4% | 506 |
| Original (raw + Left/Right slots) | External dataset (unseen) | `slots_swapped` | 13.2% | 506 |
| New (normalised + mirror-aug) | External dataset (unseen) | `slots_swapped` | 31.4% | 506 |
| Original (raw + Left/Right slots) | External dataset (unseen) | `mirrored_frame` | 17.8% | 506 |
| New (normalised + mirror-aug) | External dataset (unseen) | `mirrored_frame` | 32.8% | 506 |

When slots are swapped or the camera is mirrored, the legacy model collapses to 11%–20%, whereas the new normalized representation remains stable.

---

## 🚀 Getting Started

### Prerequisites
- **Python:** 3.11 or 3.12 (MediaPipe 0.10.21 requires Python <= 3.12; scikit-learn 1.6+ requires Python >= 3.11).
- **Virtual Environment:** Recommended.

### Installation
```bash
# 1. Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2. Install core dependencies (production ML runs without PyTorch)
pip install -r requirements.txt

# (Optional) Install PyTorch DL dependencies
pip install -r requirements-dl.txt

# (Optional) Install offline desktop TTS dependencies
pip install -r requirements-desktop.txt
```

### Running the Applications
```bash
# Launch the interactive Streamlit Web Application
streamlit run app.py

# Launch the standalone OpenCV desktop camera feed
python live_cam.py
```

### Re-running the Full Evaluation Pipeline
```bash
# Run all evaluation stages (protocols, ablation, learning curve, compare, final)
python scripts/train_alphabet.py --stage all

# Run adaptation and legacy benchmarks
python scripts/adaptation_study.py
python scripts/evaluate_legacy.py
python scripts/benchmark_latency.py

# Rebuild documentation from results/*.json
python scripts/build_docs.py
```

---

## 🧪 Automated Test Suite

Run the full pytest suite:
```bash
pytest tests/ -v
```

---

## 📄 License
This project is licensed under the MIT License. Bundled Noto Sans Devanagari font is licensed under the SIL Open Font License (OFL).
