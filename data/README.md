# 📁 Dataset Documentation & Data Card

## Overview
This repository utilizes two Indian Sign Language (ISL) fingerspelling landmark datasets:

1. **Indian Dataset (Primary Studio Dataset):**
   - **Total Images:** 31,945
   - **Image Dimensions:** 128x128
   - **Classes:** 26 (A through Z)
   - **Balance:** ~1,200 images per letter
   - **Hand Detection Rate:** 97.5%
   - **Two-Handed Distribution:** Both hands detected in >=50% of frames for 15 classes. Several letters (H, J, Q, R) feature overlapping hands where single-camera detectors intermittently resolve one hand.

2. **ISL_Dataset_2 (External Generalization Dataset):**
   - **Total Images:** 702
   - **Image Dimensions:** 128x128
   - **Classes:** 23 (no H, J, or Y in dataset)
   - **Hand Detection Rate:** 72.1%
   - **Purpose:** Used strictly as an external test set to measure domain shift and generalization to unseen signers and backgrounds.

---

## Data Leakage in Video Datasets

Video frames collected sequentially within a single session contain minimal variance:
- Consecutive frames from the same 10-second recording share nearly identical lighting, background, clothing, and hand geometry.
- Performing a standard random train/test split leaks near-identical frames into the test partition, inflating apparent test accuracy to **100.0%**.
- When evaluated with a **stratified block split** (holding out consecutive 200-frame blocks per letter), the true held-out block accuracy is **97.1%**.
- When evaluated on **external unseen signers**, accuracy drops to **26.1%**.

---

## Preprocessing & Normalization
1. **Landmark Extraction:** MediaPipe Hands extracts 21 3D landmarks $(x, y, z)$ per detected hand.
2. **Wrist Centering:** Landmark coordinates are centered relative to the primary wrist landmark.
3. **Scale Normalization:** Coordinates are scaled by the maximum Euclidean span across palm landmarks.
4. **Mirror Augmentation:** Hand coordinates are mirrored ($x \to 1 - x$), handedness labels swapped (Left $\leftrightarrow$ Right), and hand index ordering inverted to achieve full symmetry invariance.
