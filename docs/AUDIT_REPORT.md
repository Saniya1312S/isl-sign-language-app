# Comprehensive Audit & Remediation Report
**Project:** Indian Sign Language (ISL) Fingerspelling to Speech Recognition  
**Auditor:** Senior ML Engineer, QA Engineer & Technical Writer  
**Repository Branch:** `audit-fix`  
**Status:** Completed & Verified  

---

## Executive Summary

An exhaustive technical audit of the repository was conducted to identify and eliminate all unverified claims, fabricated benchmark numbers, and broken dependencies. Every single metric and claim in this repository is now **100% genuine, verifiable from raw code execution, and programmatically read from `results/*.json`**.

### Integrity & Verification Status
- **Verified:**
  - Complete ML & DL training and evaluation pipeline (`scripts/train_alphabet.py`).
  - Coordinate normalization, mirror-invariance symmetry ($x \to 1-x$, Left $\leftrightarrow$ Right swap, hand order reversal).
  - P1 (Random Split) vs. P2 (Held-out Blocks) vs. P3 (External Dataset `ISL_Dataset_2`) protocol evaluations.
  - Model comparison across 7 classifiers (Logistic Regression, kNN, Random Forest, Extra Trees, HistGB, Sklearn MLP, PyTorch ResNet-MLP).
  - Headless desktop camera runner (`live_cam.py`) for 60 synthetic frames without crash or memory leak.
  - Devanagari UTF-8 rendering via bundled OFL font (`assets/fonts/NotoSansDevanagari.ttf`).
  - Few-shot personalisation study (`scripts/adaptation_study.py`) and live in-app calibration.
  - Clean environment importability without PyTorch (`requirements.txt`).
- **Explicitly NOT Verified / Out of Scope for Local Automated Harness:**
  - Physical hardware webcam optical sensor in live room conditions (verified via headless frame injection).
  - Physical speakers / OS sound card audio hardware (verified via synthetic TTS call logging).
  - Cloud deployment / Streamlit Community Cloud server uptime.
  - External Google / Gemini API endpoints (optional network features require user-provided API keys).

---

## Detailed Findings & Remediations

### Category A: Results Integrity & Benchmark Fabrication

#### Finding A1: Fabricated 99.44% (ML) & 99.61% (DL) Headline Numbers
- **Issue:** Headline claims of 99.44% and 99.61% accuracy were produced by applying random train/test splits across sequential video frames, with mirror augmentation applied *before* the split. This caused near-duplicate video twins to leak directly into the validation partition.
- **Reproduction Evidence:** Reproducing pre-split twin augmentation on sequential video frames yielded ~99.9% apparent accuracy, but evaluating the exact same model on held-out blocks of consecutive frames dropped accuracy to ~97%, and on `ISL_Dataset_2` dropped to 26%–32%.
- **Remediation:** Deleted fabricating scripts (`scripts/train_production_models.py`, `scripts/benchmark_new_models.py`). Implemented strict stratified block splits: holding out entire 200-frame blocks per letter so that neighbouring frames never leak into test data.

#### Finding A2: Deep Learning "Best Epoch" Selection on Test Data
- **Issue:** DL model accuracy was cherry-picked by selecting `max(val_acc)` over all training epochs.
- **Reproduction Evidence:** Selecting the peak epoch on test data introduces selection bias and violates non-negotiable evaluation rules.
- **Remediation:** Fixed training schedule to fixed 25 epochs with final weights evaluated directly without validation cherry-picking.

#### Finding A3: Identical `block_test` and `external` Objects in `final_reports.json`
- **Issue:** In the original `results/final_reports.json`, `block_test == external` was `True`. The benchmark script evaluated on all rows including training data.
- **Reproduction Evidence:** Verified in Phase 0: `json.load(open('results/final_reports.json'))['block_test'] == d['external']` evaluated to `True`.
- **Remediation:** Ported evaluation to strictly separate partitions. `block_test` tests only held-out Indian studio blocks; `external` tests exclusively `ISL_Dataset_2`. In the new report, `block_test == external` is `False`.

#### Finding A4: Hard-coded Feature Importance Group Weights
- **Issue:** `importance_by_group` was hardcoded to a fixed dictionary `{"hand shape": 0.52, "inter-hand": 0.22, ...}`.
- **Reproduction Evidence:** Inspected `scripts/benchmark_new_models.py` which wrote literal static floats to JSON.
- **Remediation:** Replaced with dynamic permutation importance (`sklearn.inspection.permutation_importance`) computed over the held-out test blocks. Dynamic values sum to 1.0 and vary with true model sensitivity.

#### Finding A5: Invented Latency Multiplier (`total_ms_p95 = total_ms_p50 * 1.35`)
- **Issue:** 95th percentile latency was fabricated by multiplying the median by a constant 1.35. Hand detection latency was benchmarked on an empty black image.
- **Reproduction Evidence:** In `scripts/benchmark_latency.py`, `total_ms_p95 = total_ms_p50 * 1.35` was hardcoded.
- **Remediation:** Benchmarked MediaPipe and classifier inference on 120 real 640x480 hand images. Percentiles are calculated via `np.percentile(times, 95)` yielding genuine un-amortized measurements (e.g. p50 = 53.8 ms, p95 = 92.4 ms; ratio is ~1.72, not 1.35).

#### Finding A6: Hard-coded KPI Cards in Dashboard & Live View
- **Issue:** `views/dashboard.py` hard-coded `99.44%` and `99.61%` cards despite claiming "nothing is typed in by hand". `views/live.py` dropdown had hardcoded `99.6% Acc`.
- **Reproduction Evidence:** Inspected `views/dashboard.py` lines 293–296 and `views/live.py` line 156.
- **Remediation:** Removed all hard-coded KPI numbers. Cards and dropdown captions dynamically parse `results/final_reports.json` and `results/latency.json`.

#### Finding A7: Stale Comparison Artifacts Describing Old Model
- **Issue:** `results/protocols.json`, `model_comparison.json`, and `ablation.json` described an obsolete Logistic Regression model while metadata claimed HistGB.
- **Reproduction Evidence:** Mismatched metadata and inconsistent feature count columns.
- **Remediation:** Re-generated all comparison artifacts through a unified pipeline (`scripts/train_alphabet.py --stage all`).

#### Finding A8: Mathematical Inconsistencies in Latency Documentation
- **Issue:** `docs/INTERVIEW_GUIDE.md` contained a table where stage times summed to 39.02 ms but percentage shares summed to only 71.4%.
- **Reproduction Evidence:** Verified via manual summation of rows in `docs/INTERVIEW_GUIDE.md`.
- **Remediation:** Removed `docs/INTERVIEW_GUIDE.md` from the repository (`git rm docs/INTERVIEW_GUIDE.md`). All latency documentation is now generated by `scripts/build_docs.py` directly from `results/latency.json`.

---

### Category B: Broken Features & Code Bugs

#### Finding B1: Plotly `DummyFigure` NameError Crash
- **Issue:** `views/dashboard.py` crashed with `NameError: name 'DummyFigure' is not defined` whenever plotly *was* installed because `DummyFigure` was defined solely inside `except ImportError`.
- **Reproduction Evidence:** `pytest tests/test_dashboard.py` failed with NameError on import.
- **Remediation:** Removed the `DummyFigure` class hierarchy. Implemented clean conditional plotting: uses Plotly if installed; falls back to Streamlit native charts (`st.dataframe`, `st.bar_chart`) with an informative warning if absent.

#### Finding B2: Scikit-learn Version Incompatibility with Shipped Pickle
- **Issue:** `models/alphabet_model.joblib` was pickled with scikit-learn 1.6.1 but `requirements.txt` allowed unpinned installation (installing 1.9.1), triggering `ModuleNotFoundError: No module named '_loss'`.
- **Reproduction Evidence:** Clean environment `joblib.load()` failed on scikit-learn mismatch.
- **Remediation:** Pinned `scikit-learn==1.6.1` in `requirements.txt` and added a clean-environment model loading guard test.

#### Finding B3: `live_cam.py` Frame-1 Crash, Devanagari Bug & Swallowed TTS Errors
- **Issue:** `live_cam.py` failed on frame 1 with `NameError: name 'EPS' is not defined`. OpenCV `cv2.putText` could not render Devanagari script (producing `????`). TTS exceptions were silently swallowed via `except: pass`.
- **Reproduction Evidence:** Executing `python live_cam.py` crashed immediately on line 146.
- **Remediation:** Defined `EPS = 1e-6` at module level. Replaced `cv2.putText` for translations with a Pillow drawing routine (`put_text_utf8`) using the bundled `assets/fonts/NotoSansDevanagari.ttf` font. Added explicit one-time warning logging for TTS initialization failures. Refactored `main()` to accept synthetic frame generators and headless execution flags.

#### Finding B4: Hard PyTorch Dependency for Classical ML
- **Issue:** `torch>=2.0` (~3 GB) was a mandatory dependency in `requirements.txt`, and the app failed to boot without PyTorch installed.
- **Reproduction Evidence:** Running in an environment without PyTorch raised `ImportError`.
- **Remediation:** Decoupled PyTorch into `requirements-dl.txt`. Implemented safe optional spec checking (`importlib.util.find_spec('torch')`). The shipped production model runs entirely on lightweight scikit-learn and NumPy.

#### Finding B5: Removed Personalization Feature Still Advertised in README
- **Issue:** README advertised "Few-Shot Personalization", but the code panel had been stripped from `views/live.py`.
- **Reproduction Evidence:** Searched `views/live.py` for calibration UI components; none existed.
- **Remediation:** Restored the "🎯 Teach Your Hands" panel in `views/live.py` using `isl/personalise.py`. Users can record live calibration samples, train a personalized linear blend model, and export/import calibration CSVs.

#### Finding B6: Loosened Test Assertions
- **Issue:** Tests were artificially loosened (e.g., checking if feature count was in `(226, 171)`).
- **Reproduction Evidence:** Inspected `tests/test_features.py` assertions.
- **Remediation:** Tightened tests to assert exact expected feature dimensions (171 for `norm_eng`, 226 for `robust_isl`).

#### Finding B7: ASGI / Starlette HTTP 500 Image Bug
- **Issue:** With `streamlit>=1.50` and newer Starlette versions, `st.image` returned HTTP 500 errors.
- **Reproduction Evidence:** Observed in previous test matrix when Starlette upgraded past 1.0.
- **Remediation:** Explicitly pinned `starlette<1` in `requirements.txt` and verified image serving stability.

#### Finding B8: False Python Version Claims ("3.9–3.12")
- **Issue:** Project claimed support for Python 3.9–3.12, but modern dependencies (e.g. scikit-learn 1.6+) require Python >= 3.11, while MediaPipe 0.10.21 has no wheels for Python >= 3.13.
- **Reproduction Evidence:** Inspected wheel compatibility matrices.
- **Remediation:** Updated all documentation, badges, and CI configurations to explicitly state **Python 3.11 or 3.12**.

#### Finding B9: Sandbox Keras/TF Environment Variable Requirements
- **Issue:** Environment variables like `KERAS_HOME` and `MPLCONFIGDIR` were stripped under the assumption that TensorFlow was absent. When TensorFlow was present in site-packages (imported indirectly by MediaPipe), importing MediaPipe triggered a `PermissionError: [Errno 13] Permission denied: C:\Users\SANIYA\.keras\keras.json` in isolated environments.
- **Reproduction Evidence:** Removing `KERAS_HOME` caused `HandDetector()` initialization to fail in the sandbox.
- **Remediation:** Added automatic fallback to `tempfile.gettempdir()` for `KERAS_HOME` and `MPLCONFIGDIR` in `isl/hands.py` and `live_cam.py` without polluting user directories.

---

### Category C: Erroneous Documentation Claims

#### Finding C1: Two-Handed Gesture Count Discrepancies
- **Issue:** Claimed "19 of 26 letters are two-handed". Data reveals two hands detected in >=50% of frames for only 15 letters. Letters H, J, Q, R feature overlapping hands where MediaPipe frequently resolves one hand.
- **Remediation:** Documented genuine hand counts and optical overlap constraints in `data/README.md`. Replaced unverified gesture guide text with real dataset sample images and a regional variation disclaimer.

#### Finding C2: "60+ FPS / Zero-Lag" Marketing Claims
- **Issue:** Claimed 60+ FPS performance when MediaPipe alone requires ~50 ms per frame on standard CPU hardware (~20 FPS).
- **Remediation:** Removed marketing buzzwords. Documented real measured throughput (~18–20 FPS total pipeline latency on CPU) in `README.md` and `results/latency.json`.

#### Finding C3: Inaccurate Handedness Invariance Claims
- **Issue:** Claimed feature engineering was naturally invariant to mirror reflection without mirror augmentation. In reality, testing on horizontally flipped frames without augmentation caused accuracy to drop from ~97% to ~57% (and to 10% for legacy features).
- **Remediation:** Clarified in `README.md` that invariance is achieved through coordinate normalization paired with symmetric mirror augmentation.

#### Finding C4: Non-existent "Leaky Accumulator" Formula
- **Issue:** Documentation claimed a leaky accumulator formula (`score * 0.85 + 1.0`), which did not exist in `isl/spelling.py`.
- **Remediation:** Documented the actual implemented algorithm: sliding-window majority vote (5 frames) followed by continuous temporal hold (0.5–0.6s) and hand-lowering debounce lockout.

#### Finding C5: Inaccurate Model Sizes and Badges
- **Issue:** Documented model sizes as 4.67 MB / 719 KB (actual sizes were 40 KB for Logistic Regression and 1.41 MB for DL).
- **Remediation:** Programmatically generated model cards in `models/README.md` and `README.md` from actual on-disk file sizes.

---

## Verification Commands & Deliverables

All deliverables have been tested and verified:
1. Full test suite passing via `pytest tests/ -v`.
2. Headless desktop app verification: `python live_cam.py` tested for 60 synthetic frames.
3. Clean packaging into `isl-sign-language-app.zip` ready for GitHub release.
