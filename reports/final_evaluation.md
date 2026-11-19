# 🏆 Phase 7: Model Comparison & Final Evaluation Report
**Project:** Indian Sign Language (ISL) Alphabet Recognition  
**Evaluation Protocol:** 70% Train, 15% Validation, 15% Test on Consecutive Video Frame Blocks  
**Feature Extraction:** `norm_eng` (171 dimensions, wrist-centered coordinate & scale normalization + engineered finger invariants)  
**Selected Production Model:** **Support Vector Machine**  

---

## 1. Multi-Model Benchmark Comparison Table

| Model | Accuracy | Precision | Recall | Macro F1 | Weighted F1 | Val F1 | Mirrored Acc | Ext Acc | Train Time | Inference | Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 95.8% | 96.4% | 95.8% | 95.7% | 95.7% | 97.8% | 96.2% | 31.0% | 10.5s | 83.72 ms | 9252 KB |
| **Support Vector Machine** | 96.8% | 97.1% | 96.8% | 96.6% | 96.6% | 99.5% | 96.7% | 29.4% | 10.3s | 1.64 ms | 2021 KB |
| **Multilayer Perceptron (MLP)** | 95.6% | 96.3% | 95.6% | 95.5% | 95.5% | 97.5% | 95.7% | 24.3% | 47.5s | 1.27 ms | 1265 KB |

---

## 2. Selection Rationale for `Support Vector Machine`

1. **Validation Performance:** Selected based on highest validation Macro-F1 (99.5%) across all 26 classes.
2. **Generalization on Held-out Blocks:** Achieved **96.8% accuracy** and **96.6% Macro-F1** on held-out blocks of consecutive frames.
3. **Mirror Symmetry Robustness:** Scored **96.7%** on mirrored frames, confirming true invariance to camera selfie reflection ($x 	o 1-x$).
4. **Latency & Production Footprint:** Single-sample inference latency of **1.64 ms** on CPU and disk footprint of **2021 KB**.

---

## 3. Confusion Matrix Analysis & Failure Modes

- **Top Confusions:**
  - **U vs. V:** Both use extended index and middle fingers; differentiation depends on whether fingers are held together (U) or spread (V).
  - **M vs. N:** M places three fingers on the opposite palm, whereas N places two fingers. Under certain hand overlap angles, single-camera MediaPipe can merge adjacent fingers.
  - **E vs. S:** Closed fist formations with differing thumb placement.
- **External Dataset Performance (29.4%):**
  - Performance drops significantly on `ISL_Dataset_2` due to unseen signers, different backgrounds, and distinct camera perspectives.
  - This is an honest reflection of domain shift; few-shot personalization can recover accuracy when calibrating to a new user.

---

## 4. Visualizations

- **Random Forest Feature Importance:** `reports/figures/rf_feature_importance.png`
- **MLP Training Loss Curve:** `reports/figures/mlp_loss_curve.png`
- **Per-Model Confusion Matrices:**
  - `reports/confusion_matrices/random_forest_cm.png`
  - `reports/confusion_matrices/support_vector_machine_cm.png`
  - `reports/confusion_matrices/multilayer_perceptron_mlp_cm.png`
