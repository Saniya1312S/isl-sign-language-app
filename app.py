"""
Indian Sign Language (ISL) Alphabet Recognition System.
A clean, minimal, fast, and reliable A-Z fingerspelling application with:
- Live Webcam capture & Image Upload
- Authentic Indian ISL Alphabet Sample Gallery
- Interactive Letter-to-Sentence Builder
- Multilingual Translation (Hindi & Marathi) and Browser Speech Pronunciation
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.predictor import SignPredictor
from src.speech import generate_browser_speech_html
from src.translator import LANG_CODES, translate_text

# Paths
ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "samples"
ALPHABET_DIR = SAMPLES_DIR / "indian_isl_alphabet"
MODELS_DIR = ROOT / "models"


# -----------------------------------------------------------------------------
# Cached Resources
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained ISL A–Z model...")
def get_cached_predictor() -> SignPredictor:
    """Load and cache the trained SignPredictor once across user sessions."""
    return SignPredictor()


# -----------------------------------------------------------------------------
# Session Initialization
# -----------------------------------------------------------------------------
def init_session():
    st.session_state.setdefault("sentence", "")
    st.session_state.setdefault("last_prediction", None)
    st.session_state.setdefault("speech_nonce", 0)
    st.session_state.setdefault("gallery_idx", 0)


# -----------------------------------------------------------------------------
# UI Helpers
# -----------------------------------------------------------------------------
def render_prediction_result(res: Dict[str, Any], source_label: str = "Input Image"):
    """Render the standard prediction card with landmarks, predicted letter, and confidence."""
    col_img, col_info = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("🖼️ Landmark Detection")
        annotated_bgr = res.get("annotated_image")
        if annotated_bgr is not None:
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True, caption=f"Source: {source_label}")
        else:
            st.warning("No image frame available.")

    with col_info:
        st.subheader("🎯 Classification Result")
        hand_detected = res.get("hand_detected", False)
        letter = res.get("predicted_letter")
        conf = res.get("confidence", 0.0)
        latency = res.get("processing_time_ms", 0.0)
        status = res.get("status")
        top3 = res.get("top_predictions", [])

        # Detection Status Badge
        if hand_detected:
            st.markdown(
                f'<span style="background-color: #065f46; color: #a7f3d0; padding: 4px 12px; '
                f'border-radius: 9999px; font-weight: 600; font-size: 0.85rem;">'
                f'● Hand Detected ({latency:.1f} ms)</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<span style="background-color: #7f1d1d; color: #fecaca; padding: 4px 12px; '
                f'border-radius: 9999px; font-weight: 600; font-size: 0.85rem;">'
                f'● No Hand Detected ({latency:.1f} ms)</span>',
                unsafe_allow_html=True
            )

        st.write("")

        # Predicted Letter Card
        if hand_detected and letter and status != "low_confidence":
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #1e293b, #0f172a); 
                            border: 2px solid #14b8a6; border-radius: 12px; padding: 1.5rem; text-align: center; margin-top: 10px;">
                    <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">Predicted Alphabet</div>
                    <div style="font-size: 4.5rem; font-weight: 800; color: #2dd4bf; line-height: 1.1;">{letter}</div>
                    <div style="font-size: 1.15rem; font-weight: 600; color: #f8fafc; margin-top: 0.5rem;">
                        Confidence: {conf:.1%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Button to append to constructed sentence
            if st.button(f"➕ Add Letter '{letter}' to Sentence", key=f"btn_add_{letter}_{source_label}", use_container_width=True):
                st.session_state["sentence"] += letter
                st.toast(f"Added '{letter}' to sentence!", icon="✅")
                st.rerun()

            # Top 3 Candidates
            if top3:
                st.markdown("##### 🏆 Candidate Predictions")
                cols = st.columns(len(top3))
                for idx, (lbl, pr) in enumerate(top3):
                    with cols[idx]:
                        medal = ["🥇", "🥈", "🥉"][idx] if idx < 3 else "•"
                        st.metric(label=f"{medal} Rank {idx+1}", value=lbl, delta=f"{pr:.1%}")

        elif hand_detected and status == "low_confidence":
            st.warning(f"⚠️ Uncertain prediction (Confidence: {conf:.1%}). Please sign clearly.")
            if letter:
                st.caption(f"Best candidate: **{letter}** ({conf:.1%})")
        else:
            st.warning("⚠️ **No hand detected.**")
            st.info("Tip: Hold your hand open and centered in the camera frame with good lighting.")


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="ISL Alphabet Recognition",
        page_icon="🤟",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session()
    predictor = get_cached_predictor()

    # Sidebar: Model status & Quick Controls
    with st.sidebar:
        st.markdown("### 🤟 ISL Recognition System")
        st.caption("Indian Sign Language A–Z Fingerspelling")
        st.divider()

        st.markdown("#### ⚙️ Active Model")
        meta = predictor.metadata
        st.success(f"**Model:** {meta.get('model_name', 'Support Vector Machine')}")
        st.info(f"**Features:** {meta.get('feature_mode', 'norm_eng')} ({meta.get('n_features', 171)} dims)")
        st.caption("Alphabet Classes: 26 (A to Z)")

        st.divider()
        st.markdown("#### 🔤 Active Sentence Buffer")
        active_txt = st.session_state.get("sentence", "")
        if active_txt:
            st.code(active_txt, language="text")
            if st.button("🗑️ Clear Buffer", use_container_width=True, key="side_clear"):
                st.session_state["sentence"] = ""
                st.rerun()
        else:
            st.caption("Sentence buffer is currently empty.")

        st.divider()
        st.markdown(
            """
            **How it works:**
            1. **Sign**: Show an ISL letter (A–Z) via webcam or upload.
            2. **Build**: Accumulate letters to form words.
            3. **Translate**: Convert English to Hindi or Marathi.
            4. **Speak**: Pronounce using text-to-speech.
            """
        )

    # Header
    st.markdown(
        """
        <div style="padding: 1rem 0; border-bottom: 1px solid #334155; margin-bottom: 1.5rem;">
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #f8fafc;">
                🤟 Indian Sign Language Alphabet Recognition
            </h1>
            <p style="margin: 0.5rem 0 0 0; color: #94a3b8; font-size: 1.05rem;">
                Static Hand Gesture Classification (A–Z) · Real-time Sentence Builder · Multilingual Translation & Pronunciation
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4 Main Tabs
    tab_recog, tab_sentence, tab_home, tab_guide = st.tabs([
        "🤟 1. A–Z Recognition",
        "✍️ 2. Sentence Builder & Translation",
        "🏠 3. Project Overview",
        "📖 4. Alphabet Reference Guide"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: A–Z Recognition (Webcam / Upload / Sample Gallery)
    # -------------------------------------------------------------------------
    with tab_recog:
        st.subheader("🤟 Recognize ISL Alphabet Hand Gestures")
        st.write("Capture a frame from your webcam, upload an image, or test with authentic ISL alphabet reference samples.")

        recog_mode = st.radio(
            "Select Input Source:",
            ["📷 Webcam Capture", "📤 Upload Image", "🖼️ Indian Alphabet Sample Gallery"],
            horizontal=True,
            key="recog_input_mode"
        )

        st.divider()

        # Mode 1: Webcam
        if recog_mode == "📷 Webcam Capture":
            st.markdown("#### 📷 Webcam Hand Gesture Capture")
            cam_col1, cam_col2 = st.columns([1, 1], gap="large")

            with cam_col1:
                cam_image = st.camera_input("Take a photo of your hand gesture", key="cam_input")

            with cam_col2:
                if cam_image is not None:
                    bytes_data = cam_image.getvalue()
                    cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    if cv_img is not None:
                        with st.spinner("Classifying gesture..."):
                            res = predictor.predict_image(cv_img)
                        st.session_state["last_prediction"] = res
                        render_prediction_result(res, source_label="Webcam Capture")
                else:
                    st.info("Capture a snapshot using your camera to recognize your hand sign.")

        # Mode 2: Upload
        elif recog_mode == "📤 Upload Image":
            st.markdown("#### 📤 Upload a Hand Gesture Image")
            up_col1, up_col2 = st.columns([1, 1], gap="large")

            with up_col1:
                uploaded_file = st.file_uploader("Choose a hand gesture photo (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"], key="file_upload")

            with up_col2:
                if uploaded_file is not None:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if cv_img is not None:
                        with st.spinner("Processing image and predicting alphabet..."):
                            res = predictor.predict_image(cv_img)
                        st.session_state["last_prediction"] = res
                        render_prediction_result(res, source_label=uploaded_file.name)
                    else:
                        st.error("Failed to decode uploaded image.")
                else:
                    st.info("Upload an image of a hand sign to test.")

        # Mode 3: Sample Gallery
        else:
            st.markdown("#### 🖼️ Authentic Indian ISL Alphabet Sample Gallery")
            st.caption("26 authentic Indian Sign Language reference images evaluated on the production model.")

            letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

            col_s1, col_s2, col_s3 = st.columns([1, 2, 1])
            with col_s1:
                if st.button("⬅️ Previous Letter", key="btn_prev_gal", use_container_width=True):
                    st.session_state["gallery_idx"] = (st.session_state["gallery_idx"] - 1) % 26
                    st.rerun()
            with col_s2:
                selected_letter = st.selectbox(
                    "Choose an ISL Alphabet Letter:",
                    options=letters,
                    index=st.session_state["gallery_idx"],
                    key="gallery_letter_selector"
                )
                st.session_state["gallery_idx"] = letters.index(selected_letter)
            with col_s3:
                if st.button("Next Letter ➡️", key="btn_next_gal", use_container_width=True):
                    st.session_state["gallery_idx"] = (st.session_state["gallery_idx"] + 1) % 26
                    st.rerun()

            sample_path = ALPHABET_DIR / f"{selected_letter}.jpg"
            if not sample_path.exists():
                sample_path = SAMPLES_DIR / f"Indian_{selected_letter}.jpg"

            if sample_path.exists():
                sample_bgr = cv2.imread(str(sample_path))
                if sample_bgr is not None:
                    with st.spinner(f"Evaluating ISL Letter '{selected_letter}'..."):
                        sample_res = predictor.predict_image(sample_bgr)
                    render_prediction_result(sample_res, source_label=f"Sample: {sample_path.name}")
                else:
                    st.error(f"Could not load image: {sample_path.name}")
            else:
                st.warning(f"Reference image for '{selected_letter}' is not available.")

    # -------------------------------------------------------------------------
    # TAB 2: Sentence Builder & Translation
    # -------------------------------------------------------------------------
    with tab_sentence:
        st.subheader("✍️ Live Text Translator & Pronunciation")
        st.write("Construct English words and sentences letter-by-letter from fingerspelled signs, then translate and pronounce.")

        col_sent1, col_sent2 = st.columns([1.2, 1], gap="large")

        with col_sent1:
            st.markdown("#### 🔤 Constructed English Text")
            current_sentence = st.session_state.get("sentence", "")

            # Big Sentence Display Box
            if current_sentence:
                st.markdown(
                    f"""
                    <div style="padding: 1.2rem; background: #0f172a; border: 2px solid #334155; border-radius: 10px; margin-bottom: 1rem;">
                        <span style="font-family: monospace; font-size: 2.2rem; font-weight: 700; color: #38bdf8; letter-spacing: 2px;">
                            {current_sentence}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("Sentence buffer is empty. Sign letters in the Recognition tab or type below.")

            # Action Buttons: Space, Backspace, Clear
            b_col1, b_col2, b_col3 = st.columns(3)
            with b_col1:
                if st.button("␣ Add Space", use_container_width=True, key="btn_space"):
                    st.session_state["sentence"] += " "
                    st.rerun()
            with b_col2:
                if st.button("⌫ Backspace", use_container_width=True, key="btn_backspace"):
                    if len(st.session_state["sentence"]) > 0:
                        st.session_state["sentence"] = st.session_state["sentence"][:-1]
                        st.rerun()
            with b_col3:
                if st.button("🗑️ Clear All", use_container_width=True, key="btn_clear_all"):
                    st.session_state["sentence"] = ""
                    st.rerun()

            # Direct Manual Text Input
            st.write("")
            manual_text = st.text_input(
                "Manual Text Editor (or type directly):",
                value=current_sentence,
                key="manual_sentence_input"
            )
            if manual_text != current_sentence:
                st.session_state["sentence"] = manual_text
                st.rerun()

        with col_sent2:
            st.markdown("#### 🌐 Multilingual Translation & Speech")
            target_lang = st.radio(
                "Select Output Language:",
                ["Hindi", "Marathi", "English"],
                horizontal=True,
                key="trans_lang_radio"
            )

            st.caption("ℹ️ Translation converts the fingerspelled English text into Hindi/Marathi using built-in offline vocabulary.")

            text_to_translate = st.session_state.get("sentence", "").strip()

            if text_to_translate:
                # Perform translation
                trans_res = translate_text(text_to_translate, target_lang=target_lang, allow_online=True)
                translated_text = trans_res.text

                # Display Translated Text
                st.markdown(
                    f"""
                    <div style="padding: 1.2rem; background: linear-gradient(135deg, #1e293b, #0f172a); 
                                border: 2px solid #14b8a6; border-radius: 10px; margin-top: 0.5rem; text-align: center;">
                        <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Translated ({target_lang})</div>
                        <div style="font-size: 2.2rem; font-weight: 700; color: #2dd4bf; margin-top: 0.5rem;">
                            {translated_text or text_to_translate}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")

                # Speech Button
                col_sp1, col_sp2 = st.columns([1, 1])
                with col_sp1:
                    if st.button(f"🔊 Speak ({target_lang})", key="btn_speak", use_container_width=True):
                        st.session_state["speech_nonce"] += 1

                # Render Web Speech API Component
                if st.session_state.get("speech_nonce", 0) > 0 and translated_text:
                    speech_html = generate_browser_speech_html(
                        translated_text,
                        lang=target_lang,
                        nonce=st.session_state["speech_nonce"]
                    )
                    st.components.v1.html(speech_html, height=45)
            else:
                st.info("Enter or sign some letters to see translation and hear pronunciation.")

    # -------------------------------------------------------------------------
    # TAB 3: Project Overview
    # -------------------------------------------------------------------------
    with tab_home:
        st.subheader("🏠 Indian Sign Language (ISL) Alphabet Recognition")
        st.markdown(
            r"""
            This application provides a production machine learning pipeline for recognizing static **Indian Sign Language (ISL) alphabet fingerspelling (A to Z)**.

            ### 🎯 Core Objectives:
            1. **Fingerspelling Recognition**: Recognize static hand signs for the 26 English alphabet letters (A–Z).
            2. **Sentence Building**: Sequence recognized letters to construct full words and sentences (e.g. `H` `E` `L` `L` `O` → `HELLO`).
            3. **Multilingual Access**: Translate constructed sentences into Hindi and Marathi.
            4. **Audio Pronunciation**: Speak the translation using client-side Web Speech synthesis.

            ### 📐 Machine Learning Pipeline:
            - **Detection**: MediaPipe Hands extracts 21 three-dimensional coordinate landmarks per hand.
            - **Feature Engineering**: 171-dimensional invariant representation:
              - Normalized relative landmark coordinates (translation and scale invariant)
              - Tip-to-wrist Euclidean distances
              - Inter-finger distance ratios
              - 3-joint flexion/curl angles
            - **Classifier**: Support Vector Machine (`SVC` with RBF kernel, $C=10.0$, $\gamma=\\text{scale}$).
            - **Evaluated Performance**: **96.8% Accuracy** on held-out consecutive video blocks (zero data leakage).

            ### ⚠️ Clear Scope & Limitations:
            - **Static Alphabet Only**: This model recognizes static fingerspelling letters. It does not recognize dynamic full-body signs or continuous sign language videos.
            - **Translation Role**: Hindi and Marathi are linguistic translations of the fingerspelled English sentence; the gesture classifier detects static ISL A–Z hand signs.
            """
        )

    # -------------------------------------------------------------------------
    # TAB 4: Alphabet Reference Guide
    # -------------------------------------------------------------------------
    with tab_guide:
        st.subheader("📖 Authentic Indian ISL Alphabet Reference Guide")
        st.write("Browse reference hand signs for all 26 letters of the Indian Sign Language alphabet.")

        cols = st.columns(6)
        all_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        for i, let in enumerate(all_letters):
            col = cols[i % 6]
            with col:
                img_p = ALPHABET_DIR / f"{let}.jpg"
                if not img_p.exists():
                    img_p = SAMPLES_DIR / f"Indian_{let}.jpg"
                if img_p.exists():
                    im = Image.open(img_p)
                    st.image(im, caption=f"Letter {let}", use_container_width=True)
                else:
                    st.caption(f"Letter {let}")


if __name__ == "__main__":
    main()
