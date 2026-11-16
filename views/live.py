"""
Page 1 - Live Translator: camera (live / snapshot / upload) -> spelling -> translation -> voice.
Default engine is the efficient scikit-learn model; PyTorch DL selectable when installed.
"""
from __future__ import annotations

import html
import io
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from isl import ui
from isl.config import ALPHABET, MODELS_DIR, RESULTS_DIR, SAMPLES_DIR
from isl.hands import HandDetector
from isl.live import SignVideoProcessor, analyse_image
from isl.personalise import CalibrationSet, base_training_set, personalise
from isl.predictor import SignPredictor
from isl.runtime import is_torch_available, load_bundles, read_result
from isl.spelling import SpellingBuffer
from isl.translate import LANGS, translate
from isl.ui import speak
from isl.vocab import suggest
from isl.voice import spell_out

RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


@st.cache_resource(show_spinner="Loading models…")
def get_bundles():
    return load_bundles()


@st.cache_resource
def get_static_detector():
    return HandDetector(static_image_mode=True)


def get_active_predictor(model_choice: str, min_conf: float):
    # Personalised model takes precedence if active
    if "personal_bundle" in st.session_state and model_choice.startswith("Personalised"):
        return SignPredictor(st.session_state["personal_bundle"], min_confidence=min_conf)

    dl_path = MODELS_DIR / "isl_dl_model.pth"
    if model_choice.startswith("Deep Learning") and dl_path.exists() and is_torch_available():
        return SignPredictor(dl_path, min_confidence=min_conf)

    bundles = get_bundles()
    if "alphabet" in bundles:
        return SignPredictor(bundles["alphabet"], min_confidence=min_conf)
    return None


def show_result(img_bgr: np.ndarray, predictor: SignPredictor, key: str, buffer: SpellingBuffer) -> None:
    annotated, pred = analyse_image(img_bgr, get_static_detector(), predictor)
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
    if pred.label:
        st.success(f"Recognized **{pred.label}** with {pred.confidence:.0%} confidence")
        if st.button(f"➕ Add “{pred.label}” to my text", key=f"add_{key}"):
            buffer.append(pred.label)
            st.rerun()
    elif pred.reason == "no_hand":
        st.warning("No hand detected. Ensure good lighting and keep hands fully in the camera view.")
    else:
        best_cand = pred.top_k[0] if pred.top_k else ("None", 0.0)
        st.info(f"Uncertain (best candidate {best_cand[0]} at {best_cand[1]:.0%}). Hold hand steady.")
    if pred.top_k:
        st.caption("Top candidates: " + "  ·  ".join(f"{l}: {p:.0%}" for l, p in pred.top_k))


@st.fragment(run_every=0.5)
def text_fragment(out_lang: str) -> None:
    """Refreshes twice a second: typed text, hold-progress, suggestions, translation."""
    buf: SpellingBuffer = st.session_state.buffer
    snap = buf.snapshot()
    shown = html.escape(snap.text) if snap.text else '<span style="opacity:.35">Your recognized letters will appear here…</span>'
    st.markdown(f'<div class="typed">{shown}</div>', unsafe_allow_html=True)
    if snap.candidate:
        st.progress(snap.progress, text=f"Holding sign “{snap.candidate}” … {snap.progress:.0%}")
    else:
        st.progress(0.0, text="Make an ISL sign and hold steady to type")

    for w in suggest(snap.current_word):
        if st.button(w, key=f"sg_{w}", help="Auto-complete"):
            buf.replace_current_word(w.upper())
            st.rerun()

    code = LANGS[out_lang]
    if code != "en" and snap.text.strip():
        tr = translate(snap.text.strip(), code)
        st.session_state["last_translation"] = tr
    tr = st.session_state.get("last_translation")
    if code != "en" and tr is not None:
        if tr.text:
            st.markdown(f"**{out_lang}:** <span style='font-size:1.4rem; font-weight:600; color:#5eead4'>{tr.text}</span>  <span class='chip'>{tr.source}</span>", unsafe_allow_html=True)
        else:
            st.caption(f"Translation unavailable ({tr.error})")


@st.fragment(run_every=1.0)
def stats_fragment(ctx) -> None:
    proc = ctx.video_processor if ctx else None
    if not proc:
        return
    s = proc.stats()
    last = proc.last
    c1, c2, c3 = st.columns(3)
    c1.metric("Live FPS", f"{s['fps']:.1f}")
    c2.metric("Latency", f"{s['latency_ms']:.1f} ms")
    c3.metric("Hands Detected", last.n_hands)


def main() -> None:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    ui.inject_css()
    ui.hero(
        "🤟 ISL Real-Time Gesture-to-Speech",
        "Sign Indian Sign Language letters in front of the camera. Hold steady to type, with instant translation to Hindi & Marathi and voiceover."
    )

    if "buffer" not in st.session_state:
        st.session_state.buffer = SpellingBuffer(hold_frames=6, hold_seconds=0.6)
    buffer: SpellingBuffer = st.session_state.buffer

    if "calibration_set" not in st.session_state:
        st.session_state.calibration_set = CalibrationSet()
    cal: CalibrationSet = st.session_state.calibration_set

    # Read genuine evaluation numbers for dropdown display
    final_rep = read_result("final_reports.json") or {}
    model_name = final_rep.get("model", "Logistic Regression")
    block_acc = 0.968
    if "block_test" in final_rep and "per_class" in final_rep["block_test"]:
        block_acc = float(np.mean([c["recall"] for c in final_rep["block_test"]["per_class"]]))

    model_options = [f"Classical ML ({model_name} - {block_acc:.1%} Block Acc)"]
    dl_path = MODELS_DIR / "isl_dl_model.pth"
    if is_torch_available() and dl_path.exists():
        dl_block_acc = block_acc
        if "dl_model" in final_rep and "block_test" in final_rep["dl_model"]:
            dl_block_acc = final_rep["dl_model"]["block_test"].get("accuracy", block_acc)
        model_options.append(f"Deep Learning (PyTorch ResNet-MLP - {dl_block_acc:.1%} Block Acc)")

    if "personal_bundle" in st.session_state:
        model_options.insert(0, "Personalised Model (Trained on your hands)")

    with st.expander("⚙️ Translation & Model Settings", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        model_choice = c1.selectbox(
            "AI Model Engine",
            model_options,
            help="Default is the fast Scikit-Learn model. PyTorch Deep Learning is optional."
        )
        hold_s = c2.slider(
            "Hold time to type (seconds)", 0.3, 1.5, 0.6, 0.1,
            help="How long a gesture must be held steadily before it commits. Lower = faster, Higher = fewer typos."
        )
        min_conf = c3.slider(
            "Minimum Confidence", 0.35, 0.90, 0.55, 0.05,
            help="Confidence threshold for accepting a detected sign."
        )
        mirror = c4.checkbox("Mirror (Selfie) View", True)

        v1, v2, v3 = st.columns(3)
        out_lang = v1.selectbox("Translate & Speak to", list(LANGS))
        engine = v2.radio(
            "Voice Engine", ["Browser voice (Native Web Speech)", "Server voice (gTTS)"],
            horizontal=True,
            help="Browser voice runs offline with natural accents. Server voice uses Google TTS."
        )
        auto_speak = v3.checkbox("Speak word on Space", False)

    active_predictor = get_active_predictor(model_choice, min_conf)
    if active_predictor is None:
        st.error("No model found. Please run `python scripts/train_alphabet.py --stage final` first.")
        st.stop()

    left, right = st.columns([3, 2], gap="large")
    voice_slot = right.container()

    with left:
        t_live, t_snap, t_up, t_teach, t_guide = st.tabs([
            "🎥 Live Camera", "📸 Instant Snapshot", "📁 Upload / Samples", "🎯 Teach Your Hands", "📖 ISL Gesture Guide"
        ])

        with t_live:
            st.caption("Click **START** to turn on your webcam. Ensure your hands are clearly visible in frame.")
            ctx = webrtc_streamer(
                key="isl-live",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
                video_processor_factory=SignVideoProcessor,
                async_processing=True
            )
            if ctx.video_processor:
                ctx.video_processor.buffer = buffer
                ctx.video_processor.configure(
                    {"alphabet": active_predictor},
                    "alphabet",
                    hold_frames=max(3, int(hold_s * 15)),
                    hold_seconds=hold_s,
                    min_confidence=min_conf,
                    mirror_view=mirror
                )
            stats_fragment(ctx)

        with t_snap:
            st.caption("Take a photo with your webcam. This mode is 100% reliable across all browsers.")
            shot = st.camera_input("Take a snapshot of your gesture", key="cam_snap")
            if shot is not None:
                img = cv2.imdecode(np.frombuffer(shot.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                show_result(img, active_predictor, "snap", buffer)

        with t_up:
            st.caption("Upload an image of an ISL gesture or test with sample images.")
            up = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"], key="file_up")
            samples = sorted(p.name for p in SAMPLES_DIR.glob("*.jpg")) if SAMPLES_DIR.exists() else []
            pick = st.selectbox("…or select a sample image", ["—"] + samples)
            img = None
            if up is not None:
                img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            elif pick != "—":
                img = cv2.imread(str(SAMPLES_DIR / pick))
            if img is not None:
                show_result(img, active_predictor, "up", buffer)

        with t_teach:
            st.subheader("🎯 Teach the App Your Hands (Few-Shot Personalisation)")
            st.caption(
                "A model trained on one dataset often drops accuracy on new people and backgrounds. "
                "Adding just 2–5 examples of your own hands per letter dramatically recovers recognition accuracy."
            )
            c_letter, c_btn = st.columns([2, 3])
            teach_letter = c_letter.selectbox("Select letter to record", ALPHABET, key="teach_let")

            teach_shot = st.camera_input("Capture calibration sample for selected letter", key="cal_snap")
            if teach_shot is not None:
                img_t = cv2.imdecode(np.frombuffer(teach_shot.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                hands = get_static_detector().detect_bgr(img_t)
                if hands:
                    from isl.features import hands_to_arrays
                    c_arr, nh_arr, _ = hands_to_arrays(hands)
                    cal.add(teach_letter, c_arr[0], nh_arr[0])
                    st.success(f"Recorded sample for letter '{teach_letter}'! Total: {cal.counts().get(teach_letter, 0)}")
                else:
                    st.warning("No hand detected in calibration snapshot. Please try again.")

            counts = cal.counts()
            if counts:
                st.write("**Recorded Calibration Samples:**")
                st.write("  ·  ".join(f"**{k}**: {v}" for k, v in counts.items()))
                c_p1, c_p2, c_p3 = st.columns(3)
                if c_p1.button("⚡ Train Personal Model", type="primary"):
                    bundles = get_bundles()
                    if "alphabet" in bundles:
                        with st.spinner("Adapting model with your calibration samples…"):
                            base_X, base_y = base_training_set(mode=bundles["alphabet"].feature_mode)
                            p_bundle = personalise(bundles["alphabet"], base_X, base_y, cal)
                            st.session_state["personal_bundle"] = p_bundle
                            st.success("Personal model trained successfully! Switch AI Model Engine to 'Personalised Model'.")
                            st.rerun()

                csv_bytes = cal.to_csv()
                c_p2.download_button("💾 Export Calibration CSV", csv_bytes, "isl_calibration.csv", "text/csv")
                if c_p3.button("🗑️ Clear Calibration"):
                    cal.clear()
                    st.session_state.pop("personal_bundle", None)
                    st.rerun()

            cal_up = st.file_uploader("Load previous calibration CSV", type=["csv"], key="cal_csv_up")
            if cal_up is not None:
                try:
                    n_loaded = cal.load_csv(cal_up.getvalue())
                    st.success(f"Loaded {n_loaded} calibration samples from CSV!")
                except Exception as e_cal:
                    st.error(f"Failed to load CSV: {e_cal}")

        with t_guide:
            st.subheader("📖 Indian Sign Language (ISL) Alphabet Reference")
            st.caption("Shown: gestures in our training dataset. Indian Sign Language has regional variations; check an ISL dictionary.")

            guide_cols = st.columns(4)
            for idx, letter in enumerate(ALPHABET):
                col = guide_cols[idx % 4]
                img_path = SAMPLES_DIR / f"Indian_{letter}.jpg"
                with col:
                    if img_path.exists():
                        st.image(str(img_path), caption=f"Letter {letter}", use_container_width=True)
                    else:
                        st.markdown(f"**Letter {letter}**")

    with right:
        text_fragment(out_lang)
        b1, b2, b3 = st.columns(3)
        if b1.button("␣ Space", use_container_width=True):
            word = buffer.current_word
            buffer.add_space()
            if auto_speak and word:
                speak(voice_slot, word, LANGS[out_lang] if out_lang != "English" else "en", engine)
            st.rerun()
        if b2.button("⌫ Delete", use_container_width=True):
            buffer.backspace()
            st.rerun()
        if b3.button("🗑 Clear", use_container_width=True):
            buffer.clear()
            st.session_state.pop("last_translation", None)
            st.rerun()

        s1, s2 = st.columns(2)
        if s1.button("🔊 Speak", use_container_width=True, type="primary"):
            text, lang = buffer.text.strip(), "en"
            if out_lang != "English":
                tr = translate(text, LANGS[out_lang])
                if tr and tr.text:
                    text, lang = tr.text, LANGS[out_lang]
            speak(voice_slot, text, lang, engine)

        if s2.button("🔤 Spell Aloud", use_container_width=True, help="Reads the letters one by one (e.g. C - A - T)"):
            speak(voice_slot, spell_out(buffer.text, "en" if out_lang == "English" else LANGS[out_lang]),
                  LANGS[out_lang], engine)

        if st.button("🤖 Ask SignAssistant with this text", use_container_width=True):
            st.session_state["assistant_draft"] = buffer.text.strip()
            st.switch_page("views/assistant_page.py")


if __name__ == "__main__":
    main()
