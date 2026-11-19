# 📸 Phase 8: Automated Real Image Testing Report
**Project:** Indian Sign Language (ISL) Alphabet Recognition  
**Test Subject:** Actual Images in `samples/` Directory  
**Total Images:** 49  

---

## 1. Test Summary

| Metric | Measured Score | Explanation |
| :--- | :---: | :--- |
| **MediaPipe Hand Detection Rate** | **77.6%** (38/49) | Images where MediaPipe successfully located hands |
| **Indian Studio Samples Accuracy** | **100.0%** (26/26) | Clean studio samples across all 26 classes |
| **External Samples Accuracy** | **8.7%** (2/23) | Challenging unseen signers and varied backgrounds |
| **Overall Real Image Accuracy** | **57.1%** (28/49) | End-to-end performance across all available samples |

---

## 2. Complete Sample Prediction Table

| Image File | Dataset Source | Expected | Predicted | Confidence | Hand Detected | Result | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `External_A.jpg` | `External` | **A** | **J** | 65.6% | True | ❌ **FAIL (Wrong Label)** | 163.1 ms |
| `External_B.jpg` | `External` | **B** | **None** | 0.0% | False | ❌ **FAIL (No Hand)** | 90.9 ms |
| `External_C.jpg` | `External` | **C** | **None** | 0.0% | False | ❌ **FAIL (No Hand)** | 44.6 ms |
| `External_D.jpg` | `External` | **D** | **None** | 0.0% | False | ❌ **FAIL (No Hand)** | 42.8 ms |
| `External_E.jpg` | `External` | **E** | **A** | 39.3% | True | ❌ **FAIL (Wrong Label)** | 134.5 ms |
| `External_F.jpg` | `External` | **F** | **F** | 70.7% | True | ✅ **PASS** | 91.4 ms |
| `External_G.jpg` | `External` | **G** | **None** | 0.0% | False | ❌ **FAIL (No Hand)** | 50.1 ms |
| `External_I.jpg` | `External` | **I** | **None** | 0.0% | False | ❌ **FAIL (No Hand)** | 48.3 ms |
| `External_K.jpg` | `External` | **K** | **None** | 0.0% | False | ❌ **FAIL (No Hand)** | 45.3 ms |
| `External_L.jpg` | `External` | **L** | **None** | 0.0% | False | ❌ **FAIL (No Hand)** | 51.9 ms |
| `External_M.jpg` | `External` | **M** | **R** | 35.3% | True | ❌ **FAIL (Wrong Label)** | 109.3 ms |
| `External_N.jpg` | `External` | **N** | **F** | 58.1% | True | ❌ **FAIL (Wrong Label)** | 92.4 ms |
| `External_O.jpg` | `External` | **O** | **None** | 0.0% | False | ❌ **FAIL (No Hand)** | 64.5 ms |
| `External_P.jpg` | `External` | **P** | **D** | 52.1% | True | ❌ **FAIL (Wrong Label)** | 137.8 ms |
| `External_Q.jpg` | `External` | **Q** | **None** | 0.0% | False | ❌ **FAIL (No Hand)** | 48.6 ms |
| `External_R.jpg` | `External` | **R** | **H** | 98.2% | True | ❌ **FAIL (Wrong Label)** | 98.6 ms |
| `External_S.jpg` | `External` | **S** | **S** | 76.8% | True | ✅ **PASS** | 103.5 ms |
| `External_T.jpg` | `External` | **T** | **Y** | 94.9% | True | ❌ **FAIL (Wrong Label)** | 160.8 ms |
| `External_U.jpg` | `External` | **U** | **Y** | 82.3% | True | ❌ **FAIL (Wrong Label)** | 132.1 ms |
| `External_V.jpg` | `External` | **V** | **None** | 0.0% | False | ❌ **FAIL (No Hand)** | 94.8 ms |
| `External_W.jpg` | `External` | **W** | **R** | 34.2% | True | ❌ **FAIL (Wrong Label)** | 114.5 ms |
| `External_X.jpg` | `External` | **X** | **Y** | 38.7% | True | ❌ **FAIL (Wrong Label)** | 208.3 ms |
| `External_Z.jpg` | `External` | **Z** | **None** | 0.0% | False | ❌ **FAIL (No Hand)** | 61.2 ms |
| `Indian_A.jpg` | `Indian` | **A** | **A** | 97.9% | True | ✅ **PASS** | 151.2 ms |
| `Indian_B.jpg` | `Indian` | **B** | **B** | 90.4% | True | ✅ **PASS** | 139.1 ms |
| `Indian_C.jpg` | `Indian` | **C** | **C** | 97.9% | True | ✅ **PASS** | 94.8 ms |
| `Indian_D.jpg` | `Indian` | **D** | **D** | 97.7% | True | ✅ **PASS** | 133.9 ms |
| `Indian_E.jpg` | `Indian` | **E** | **E** | 95.8% | True | ✅ **PASS** | 144.7 ms |
| `Indian_F.jpg` | `Indian` | **F** | **F** | 97.7% | True | ✅ **PASS** | 134.5 ms |
| `Indian_G.jpg` | `Indian` | **G** | **G** | 97.7% | True | ✅ **PASS** | 142.3 ms |
| `Indian_H.jpg` | `Indian` | **H** | **H** | 98.4% | True | ✅ **PASS** | 95.7 ms |
| `Indian_I.jpg` | `Indian` | **I** | **I** | 98.3% | True | ✅ **PASS** | 90.1 ms |
| `Indian_J.jpg` | `Indian` | **J** | **J** | 98.1% | True | ✅ **PASS** | 84.3 ms |
| `Indian_K.jpg` | `Indian` | **K** | **K** | 97.7% | True | ✅ **PASS** | 135.2 ms |
| `Indian_L.jpg` | `Indian` | **L** | **L** | 98.6% | True | ✅ **PASS** | 98.4 ms |
| `Indian_M.jpg` | `Indian` | **M** | **M** | 98.9% | True | ✅ **PASS** | 151.7 ms |
| `Indian_N.jpg` | `Indian` | **N** | **N** | 98.2% | True | ✅ **PASS** | 173.8 ms |
| `Indian_O.jpg` | `Indian` | **O** | **O** | 99.4% | True | ✅ **PASS** | 148.9 ms |
| `Indian_P.jpg` | `Indian` | **P** | **P** | 98.0% | True | ✅ **PASS** | 158.7 ms |
| `Indian_Q.jpg` | `Indian` | **Q** | **Q** | 97.6% | True | ✅ **PASS** | 94.1 ms |
| `Indian_R.jpg` | `Indian` | **R** | **R** | 98.6% | True | ✅ **PASS** | 89.0 ms |
| `Indian_S.jpg` | `Indian` | **S** | **S** | 95.8% | True | ✅ **PASS** | 105.3 ms |
| `Indian_T.jpg` | `Indian` | **T** | **T** | 98.4% | True | ✅ **PASS** | 145.5 ms |
| `Indian_U.jpg` | `Indian` | **U** | **U** | 99.1% | True | ✅ **PASS** | 105.1 ms |
| `Indian_V.jpg` | `Indian` | **V** | **V** | 97.7% | True | ✅ **PASS** | 92.8 ms |
| `Indian_W.jpg` | `Indian` | **W** | **W** | 99.4% | True | ✅ **PASS** | 92.6 ms |
| `Indian_X.jpg` | `Indian` | **X** | **X** | 96.2% | True | ✅ **PASS** | 148.3 ms |
| `Indian_Y.jpg` | `Indian` | **Y** | **Y** | 95.6% | True | ✅ **PASS** | 132.5 ms |
| `Indian_Z.jpg` | `Indian` | **Z** | **Z** | 97.9% | True | ✅ **PASS** | 126.6 ms |

---

## 3. Failure Mode & Root-Cause Investigation

1. **Optical Occlusion & Multi-Hand Overlap:** Letters like H, J, and R involve overlapping fingers from both hands. MediaPipe can intermittently detect only one hand or lose finger joint depth in 2D perspective.
2. **Domain Shift in External Samples:** External samples have varied skin tones, background clutter, and differing camera distances compared to studio recordings.
3. **Annotated Debugging Images:** Debugging images with drawn skeletal landmarks have been rendered and saved in `reports/figures/sample_predictions/`.
