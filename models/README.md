# Model card — `alphabet_model.joblib`

| | |
|---|---|
| Task | classify a hand pose (one or two hands) into one of the 26 ISL fingerspelling letters |
| Model | scikit-learn `Pipeline(StandardScaler → Logistic Regression)`, selected by cross-validation (see `results/model_comparison.json`) |
| Input | 171 features from `isl/features.py` (mode `norm_eng`): normalised landmarks, fingertip distances, curl angles, inter-hand geometry |
| Output | letter + probability (the app ignores answers below the *minimum confidence*, default 0.60) |
| File | `models/alphabet_model.joblib` (21 KB) + `alphabet_meta.json` (feature mode, classes, training data, metrics, library version) |
| Training data | 31,130 studio images with a detected hand + 506 external images (each repeated ×10), all mirror-augmented |
| Speed | 0.002 ms per prediction (the MediaPipe hand detector dominates the cost) |

## Evaluation (honest numbers)

The numbers below come from an **evaluation copy** trained on studio training blocks only — the saved production model additionally saw the external images, so its own accuracy on them cannot be measured.

| Test | Accuracy |
|---|---:|
| Held-out blocks of the studio dataset (all 26 letters) | 96.7% (macro-F1 0.972) |
| External dataset, hand detected (506 images) | 32.2% |
| External dataset, missed hands count as errors | see Dashboard → Models |
| Random split (for reference – **not valid**, leaks neighbouring frames) | 100.0% |

Weakest letters on held-out blocks: I (F1 0.68), S (F1 0.81), D (F1 0.86).

## Intended use

Demonstration, teaching and research on landmark-based fingerspelling recognition; an assistive *aid* for typing letters.

## Limitations and risks

* Accuracy on people and rooms unlike the studio data is much lower; use *Teach the app your hands* (personal model) and verify with the user.
* Only static letters; no motion, no words, no sentences, no facial grammar. Not a replacement for a human interpreter. Not evaluated with deaf users.
* Confidence values of a linear model are not calibrated probabilities.

## Retrain

```bash
python scripts/train_alphabet.py --stage all      # ≈ 15 min on one CPU core; rewrites results/*.json and this model
```
