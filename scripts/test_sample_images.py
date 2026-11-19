"""
Phase 8: Automated Image-Based Testing Script.

Evaluates the trained ISL model using all actual images in samples/:
- Indian_A.jpg .. Indian_Z.jpg (26 letters)
- External_A.jpg .. External_Z.jpg (23 letters)

Outputs:
- reports/sample_image_test.csv
- reports/sample_image_test.md
- reports/figures/sample_predictions/ (annotated visual debugging images)
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Safe environment paths
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.feature_extraction import get_detector
from src.predictor import SignPredictor

SAMPLES_DIR = ROOT / "samples"
REPORTS_DIR = ROOT / "reports"
OUTPUT_DIR = REPORTS_DIR / "figures" / "sample_predictions"


def run_sample_image_tests():
    print("=" * 60)
    print("PHASE 8: Testing Trained Model on Real Sample Images...")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictor = SignPredictor()
    detector = get_detector()

    sample_images = sorted(list(SAMPLES_DIR.glob("*.jpg")) + list(SAMPLES_DIR.glob("*.png")))
    if not sample_images:
        print(f"No sample images found in {SAMPLES_DIR}!")
        return

    print(f"Found {len(sample_images)} sample images to evaluate.")

    results = []
    detector_failures = 0
    indian_correct = 0
    indian_total = 0
    external_correct = 0
    external_total = 0

    print("\n| Image File | Expected | Predicted | Confidence | Result | Time (ms) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for img_path in sample_images:
        name = img_path.name
        source = "Indian" if name.startswith("Indian_") else ("External" if name.startswith("External_") else "Other")
        # Extract expected label from filename: Indian_A.jpg -> A
        expected = name.split(".")[0].split("_")[-1].upper()

        if source == "Indian":
            indian_total += 1
        elif source == "External":
            external_total += 1

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            results.append({
                "image": name, "source": source, "expected": expected, "predicted": "ERROR",
                "confidence": 0.0, "hand_detected": False, "result": "FAIL (Read Error)", "time_ms": 0.0
            })
            continue

        t0 = time.perf_counter()
        pred_res = predictor.predict_image(bgr, detector=detector)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        hand_detected = pred_res["hand_detected"]
        pred_label = pred_res["predicted_letter"]
        conf = pred_res["confidence"]
        annotated = pred_res["annotated_image"]

        if not hand_detected:
            detector_failures += 1
            result_str = "FAIL (No Hand)"
        elif pred_label == expected:
            result_str = "PASS"
            if source == "Indian":
                indian_correct += 1
            elif source == "External":
                external_correct += 1
        else:
            result_str = "FAIL (Wrong Label)"

        # Overlay result text on annotated debugging image
        h, w = annotated.shape[:2]
        header_color = (40, 200, 80) if result_str == "PASS" else (60, 60, 240)
        cv2.rectangle(annotated, (0, 0), (w, 36), (20, 20, 20), -1)
        res_txt = f"{expected} -> {pred_label or 'None'} ({conf:.0%}) [{result_str}]"
        cv2.putText(annotated, res_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, header_color, 2, cv2.LINE_AA)

        # Save annotated image
        cv2.imwrite(str(OUTPUT_DIR / f"annotated_{name}"), annotated)

        row = {
            "image": name,
            "source": source,
            "expected": expected,
            "predicted": pred_label or "None",
            "confidence": round(conf, 4),
            "hand_detected": hand_detected,
            "result": result_str,
            "time_ms": round(elapsed_ms, 2)
        }
        results.append(row)

        print(f"| {name:<16} | {expected:^8} | {row['predicted']:^9} | {conf:>9.1%} | {result_str:<17} | {elapsed_ms:>8.1f} ms |")

    # Save reports/sample_image_test.csv
    csv_path = REPORTS_DIR / "sample_image_test.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "source", "expected", "predicted", "confidence", "hand_detected", "result", "time_ms"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved CSV test results to {csv_path}")

    # Summary Statistics
    ind_rate = (indian_correct / indian_total) if indian_total else 0.0
    ext_rate = (external_correct / external_total) if external_total else 0.0
    overall_correct = indian_correct + external_correct
    overall_rate = overall_correct / len(results) if results else 0.0
    detection_rate = (len(results) - detector_failures) / len(results) if results else 0.0

    print("\n" + "=" * 60)
    print("SAMPLE IMAGE TEST SUMMARY:")
    print(f"  Total Images Tested     : {len(results)}")
    print(f"  MediaPipe Detection Rate: {detection_rate:.1%} ({len(results) - detector_failures}/{len(results)})")
    print(f"  Indian Samples Accuracy : {ind_rate:.1%} ({indian_correct}/{indian_total})")
    print(f"  External Samples Accuracy: {ext_rate:.1%} ({external_correct}/{external_total})")
    print(f"  Overall Accuracy         : {overall_rate:.1%} ({overall_correct}/{len(results)})")
    print("=" * 60)

    # Save reports/sample_image_test.md
    md_rows = []
    for r in results:
        status_badge = "✅ **PASS**" if r["result"] == "PASS" else f"❌ **{r['result']}**"
        md_rows.append(f"| `{r['image']}` | `{r['source']}` | **{r['expected']}** | **{r['predicted']}** | {r['confidence']:.1%} | {r['hand_detected']} | {status_badge} | {r['time_ms']:.1f} ms |")
    md_table = "\n".join(md_rows)

    md_content = f"""# 📸 Phase 8: Automated Real Image Testing Report
**Project:** Indian Sign Language (ISL) Alphabet Recognition  
**Test Subject:** Actual Images in `samples/` Directory  
**Total Images:** {len(results)}  

---

## 1. Test Summary

| Metric | Measured Score | Explanation |
| :--- | :---: | :--- |
| **MediaPipe Hand Detection Rate** | **{detection_rate:.1%}** ({len(results) - detector_failures}/{len(results)}) | Images where MediaPipe successfully located hands |
| **Indian Studio Samples Accuracy** | **{ind_rate:.1%}** ({indian_correct}/{indian_total}) | Clean studio samples across all 26 classes |
| **External Samples Accuracy** | **{ext_rate:.1%}** ({external_correct}/{external_total}) | Challenging unseen signers and varied backgrounds |
| **Overall Real Image Accuracy** | **{overall_rate:.1%}** ({overall_correct}/{len(results)}) | End-to-end performance across all available samples |

---

## 2. Complete Sample Prediction Table

| Image File | Dataset Source | Expected | Predicted | Confidence | Hand Detected | Result | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{md_table}

---

## 3. Failure Mode & Root-Cause Investigation

1. **Optical Occlusion & Multi-Hand Overlap:** Letters like H, J, and R involve overlapping fingers from both hands. MediaPipe can intermittently detect only one hand or lose finger joint depth in 2D perspective.
2. **Domain Shift in External Samples:** External samples have varied skin tones, background clutter, and differing camera distances compared to studio recordings.
3. **Annotated Debugging Images:** Debugging images with drawn skeletal landmarks have been rendered and saved in `reports/figures/sample_predictions/`.
"""
    (REPORTS_DIR / "sample_image_test.md").write_text(md_content, encoding="utf-8")
    print(f"Saved Markdown test report to {REPORTS_DIR / 'sample_image_test.md'}")


if __name__ == "__main__":
    run_sample_image_tests()
