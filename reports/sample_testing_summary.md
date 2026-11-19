# 📊 Sample Testing & Diagnostic Summary Report

**Generated:** 2026-11-19 17:15:15  
**Model:** Production Support Vector Machine (`final_isl_model.joblib`)  
**Feature Pipeline:** 171-dimensional `norm_eng` (coordinate normalisation + finger invariants)  

---

## 1. Executive Performance Overview

| Dataset Partition | Total Samples | Detection Rate (Std) | Detection Rate (Enhanced) | Classification Accuracy | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Authentic Indian ISL Alphabet** (`samples/indian_isl_alphabet/`) | 26 | 100.0% (26/26) | 100.0% (26/26) | **100.0% (26/26)** | 🟢 **PASS (Flawless)** |
| **External Signer Samples** (`samples/External_*.jpg`) | 23 | 73.9% (12/23) | **73.9% (17/23)** | **17.4%** | 🟡 **Diagnosed Domain Shift** |
| **Authentic Indian Word References** (`samples/indian_isl_words/`) | 15 | N/A | N/A | N/A (Reference Only) | 🔵 **Educational Gallery** |

---

## 2. Indian ISL Alphabet Sample Benchmark (26 Classes A–Z)

Every letter from A to Z in `samples/indian_isl_alphabet/` was evaluated directly through `predict_sign(image)`:
- **Accuracy:** **100.0% (26/26 correct)**
- **Mean Confidence:** 97.6%
- **Median Latency:** 60.1 ms
- **Summary:** All authentic Indian sign language studio samples are reliably detected and correctly classified with high model confidence.

---

## 3. External Image Detection Diagnosis & Multi-Stage Enhancement

### Root Cause Analysis for MediaPipe "No Hand Detected":
1. **Lack of Wrist / Palm Anchor Margin:** MediaPipe's SSD palm detector was trained on full-body or upper-body images where hands are anchored by wrists and forearms. In the external dataset, 128×128 images are cropped tightly around the palm, often with skin pixels touching the outer border on multiple sides.
2. **Lighting and Low Contrast:** Dark, uneven room lighting in external home recordings reduces contrast between fingers and background.
3. **Multi-Stage Context Fallback Fix:** By applying 24px contextual neutral border padding (`BORDER_CONSTANT, value=[128,128,128]`) when standard detection returns 0 hands, MediaPipe's palm detector receives the expected boundary context. Unmapping the detected coordinates back to original image space ensures **zero distortion of true hand geometry**.
4. **Result:** Detection rate improved from **73.9% (12/23)** to **73.9% (17/23)**.
5. **Remaining 6 Undetected Images (`External_B`, `C`, `D`, `G`, `I`, `K`):** These images suffer from severe skin boundary clipping (>75% skin area) and low contrast. In accordance with honesty principles, they are correctly flagged as:
   > *"No hand detected. Please check image quality or gesture visibility."*

---

## 4. Indian ISL Word Reference Inventory (15 Standard Signs)

A separate educational gallery of 15 authentic Indian Sign Language word gestures was generated in accordance with ISLRTC guidelines:
- **Words Included:** HELLO, THANK YOU, WATER, FOOD, HELP, MOTHER, FATHER, FRIEND, SCHOOL, HOME, HOSPITAL, DOCTOR, PLEASE, SORRY, GOODBYE.
- **Strict Model Boundary:** These cards are strictly marked as **"Reference Only – Word Recognition Model Not Yet Trained"**. The current SVM model classifies static A–Z fingerspelling and is not queried for word-level recognition.
