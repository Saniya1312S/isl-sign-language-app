# 🧠 Model Documentation & Checkpoint Card

## Production Shipped Model (`alphabet_model.joblib`)

- **Model Type:** Logistic Regression
- **Framework:** scikit-learn (saved with exact pinned version matching environment)
- **Feature Pipeline:** `norm_eng` (171 dimensions) with mirror augmentation
- **File Size:** ~0.04 MB (41 KB)
- **Held-out Block Accuracy:** 97.9%
- **Single-Sample Inference Latency:** ~0.22 ms on CPU
- **Dependencies:** Requires only `scikit-learn` and `numpy`. Operates completely without PyTorch.

### Selection Rationale
Selected among all benchmarked classifiers based on the pre-declared rule:
*Among models within 0.005 CV Macro-F1 of the best performing classifier, select the one with the smallest disk footprint and fastest single-sample inference.*

---

## Optional Deep Learning Model (`isl_dl_model.pth`)

- **Architecture:** `ISLLandmarkNet` (PyTorch Residual MLP)
- **Design:** Input projection (171 $\to$ 256) followed by 2 Residual Dense Blocks with LayerNorm, GELU activations, and Dropout (0.2), terminating in a 26-class classification head.
- **Training Protocol:** Fixed 25 epochs, AdamW optimizer (learning rate $10^{-3}$, weight decay $10^{-4}$), CrossEntropyLoss. Final weights are evaluated directly without cherry-picking best-validation epochs.
- **File Size:** ~1.41 MB
- **Dependencies:** Optional (`torch>=2.0` via `requirements-dl.txt`). The main app functions without PyTorch.
