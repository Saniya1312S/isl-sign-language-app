"""Page 4 - About: architecture, honest scope, how it differs from the original report."""
import streamlit as st

from isl import ui
from isl.runtime import read_result

DOT = """
digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#131C2E",
  fontcolor="#E6EDF7", color="#14B8A6", fontname="Helvetica"]; edge [color="#64748b"];
  cam [label="Browser camera\\n(WebRTC)"]; mp [label="MediaPipe Hands\\n21 landmarks / hand"];
  feat [label="Feature engineering\\nnormalise · order · angles"]; clf [label="Classifier\\n(scikit-learn Pipeline)"];
  spell [label="Spelling buffer\\nsmooth · hold · re-arm"]; tr [label="Translate\\nglossary + online"];
  voice [label="Voice\\nbrowser / gTTS"]; llm [label="SignAssistant\\n(Gemini)"];
  cam -> mp -> feat -> clf -> spell -> tr -> voice; spell -> llm -> voice; }
"""


def main() -> None:
    ui.inject_css()
    ui.hero("ℹ️ About this project",
            "A fingerspelling-to-speech system for Indian Sign Language (ISL): computer vision + classical machine learning "
            "+ translation + voice + an LLM assistant.")
    st.graphviz_chart(DOT)
    meta = read_result("eda.json")

    st.subheader("What is implemented — and what is not")
    st.markdown("""
| Area | Status in this repository |
|---|---|
| ISL alphabet A-Z recognition from hand landmarks | ✅ trained, evaluated three ways, live in the app |
| Browser-based real-time camera, snapshot and upload | ✅ |
| Fingerspelling → text with smoothing / hold / auto-complete | ✅ |
| English → Hindi / Marathi (offline glossary + online translation) | ✅ (online part needs internet) |
| Voice: browser speech + gTTS | ✅ (audio output itself must be checked on your device) |
| Gemini SignAssistant | ✅ code + tests with a fake client; **needs your own API key**, not verified against the live API here |
| Word-level and Hindi-letter recognition | ⚙️ pipeline included (`scripts/train_custom.py`); **models not shipped** — datasets were not part of the project files |
| MediaPipe *Holistic*, CNN-LSTM / Transformer, sequence windows, ONNX/TFLite | ❌ **not implemented** (the original report mentions them; the code never used them) |
""")
    st.subheader("Why classical ML and not deep learning?")
    st.markdown("""
Each sign is a **single frame** described by 21 landmarks per hand, i.e. a ~170-number vector. For small tabular
vectors like this, tree ensembles and small neural networks usually perform similarly (see **Dashboard → Models**,
where an MLP is compared), they train in seconds on a CPU and run in about a millisecond. Deep learning would become
the right tool for (a) raw images (a CNN) or (b) *moving* signs and whole sentences (LSTM / Transformer over
sequences of frames) — that is the natural next step and is listed under Future work in the README.
""")
    st.subheader("Privacy")
    st.markdown("Camera frames are processed in memory and **not stored**. Only text you choose to send is sent to "
                "Google (translation, text-to-speech, Gemini).")
    if meta:
        st.caption(f"Data summary available: {meta['total_images']:,} images processed.")


main()
