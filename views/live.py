"""
Page 1 - Live Translator: camera (live / snapshot / upload) -> spelling -> translation -> voice.
Supports both Classical Machine Learning (HistGradientBoosting) and Deep Learning (PyTorch ResNet-MLP).
"""
import html
from pathlib import Path
import cv2
import numpy as np
import streamlit as st

from isl import ui
from isl.ui import speak
from isl.config import ALPHABET, SAMPLES_DIR, MODELS_DIR
from isl.hands import HandDetector
from isl.live import SignVideoProcessor, analyse_image
from isl.runtime import MODE_LABELS, load_bundles
from isl.predictor import SignPredictor
from isl.spelling import SpellingBuffer
from isl.translate import LANGS, translate
from isl.vocab import suggest
from isl.voice import spell_out

RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

ISL_GESTURE_GUIDE = {
    "A": ("Two-handed", "Left thumb pointed up; Right index finger touches the tip of left thumb."),
    "B": ("Two-handed", "Both hands form circles with thumbs and index fingers, touching together (glasses shape)."),
    "C": ("One-handed", "Right hand forms an open curved 'C' shape facing left."),
    "D": ("Two-handed", "Left index finger held vertically; Right thumb and index form a curve touching left index."),
    "E": ("Two-handed", "Left hand fingers spread; Right index finger touches tip of left index finger."),
    "F": ("Two-handed", "Right index and middle fingers rest crossed horizontally over left index and middle."),
    "G": ("Two-handed", "Both hands formed into closed fists, placed one on top of the other."),
    "H": ("Two-handed", "Left palm held flat facing up; Right palm sweeps across the left palm horizontally."),
    "I": ("One-handed", "Left index finger vertical; or single right pinky held upright."),
    "J": ("One-handed", "Pinky finger held upright, then curved downwards in a 'J' hook motion."),
    "K": ("Two-handed", "Left index finger vertical; Right index finger bent and hooked onto the left knuckle."),
    "L": ("One-handed", "Right hand forms an 'L' shape with thumb horizontal and index finger vertical."),
    "M": ("Two-handed", "Three fingers of right hand (index, middle, ring) rest on the palm of left hand."),
    "N": ("Two-handed", "Two fingers of right hand (index, middle) rest on the palm of left hand."),
    "O": ("One-handed", "Fingers and thumb of right hand curved to touch and form a closed circle 'O'."),
    "P": ("Two-handed", "Left index finger vertical; Right thumb and index form loop touching top of left index."),
    "Q": ("Two-handed", "Left thumb and index form horizontal circle; Right index hooks through the circle."),
    "R": ("Two-handed", "Left palm open facing up; Right index and middle fingers crossed resting on left palm."),
    "S": ("Two-handed", "Left pinky finger extended; Right pinky hooks around left pinky finger."),
    "T": ("Two-handed", "Left hand flat, palm facing signer; Right index finger touches the edge of left hand."),
    "U": ("One-handed", "Right index and middle fingers held upright together, thumb holding down ring/pinky."),
    "V": ("One-handed", "Right index and middle fingers held upright spread apart in a 'V' peace shape."),
    "W": ("One-handed", "Right index, middle, and ring fingers held upright spread apart in a 'W' shape."),
    "X": ("Two-handed", "Both index fingers extended and crossed over each other to form an 'X'."),
    "Y": ("Two-handed", "Left index finger extended; Right hand forms 'V' between thumb and index touching left index."),
    "Z": ("Two-handed", "Right index finger touches the palm of left hand held vertically.")
}


@st.cache_resource(show_spinner="Loading models…")
def get_bundles():
    return load_bundles()


@st.cache_resource
def get_static_detector():
    return HandDetector(static_image_mode=True)


def get_active_predictor(model_choice: str, min_conf: float):
    dl_path = MODELS_DIR / "isl_dl_model.pth"
    if model_choice.startswith("Deep Learning") and dl_path.exists():
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
        st.info(f"Uncertain (best candidate {pred.top_k[0][0]} at {pred.top_k[0][1]:.0%}). Hold hand steady.")
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
    c1.metric("Live FPS", s["fps"])
    c2.metric("Latency", f"{s['latency_ms']:.1f} ms")
    c3.metric("Hands Detected", last.n_hands)


def main() -> None:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    ui.inject_css()
    ui.hero(
        "🤟 ISL Real-Time Gesture-to-Speech",
        "Sign Indian Sign Language letters in front of the camera. Hold steady to type, with instant translation to Hindi & Marathi and voiceover."
    )

    # Fast Launch Callout for Interview Demo
    st.info(
        "💡 **Interview Demo Tip:** For maximum 60+ FPS zero-latency performance without any browser/network overhead, "
        "run `python live_cam.py` in your terminal to launch the dedicated high-speed desktop camera window."
    )

    if "buffer" not in st.session_state:
        st.session_state.buffer = SpellingBuffer(hold_frames=6, hold_seconds=0.6)
    buffer: SpellingBuffer = st.session_state.buffer

    with st.expander("⚙️ Translation & Model Settings", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        model_choice = c1.selectbox(
            "AI Model Engine",
            ["Deep Learning (PyTorch ResNet-MLP - 99.6% Acc)", "Classical ML (HistGradientBoosting - 99.4% Acc)"],
            help="Switch between PyTorch Deep Neural Network and Scikit-Learn Gradient Boosted Trees."
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
        st.error("No model found. Please run `python scripts/train_production_models.py` first.")
        st.stop()

    left, right = st.columns([3, 2], gap="large")
    voice_slot = right.container()

    with left:
        t_live, t_snap, t_up, t_guide = st.tabs([
            "🎥 Live Camera", "📸 Instant Snapshot", "📁 Upload / Samples", "📖 ISL Gesture Guide"
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

        with t_guide:
            st.subheader("Indian Sign Language (ISL) Alphabet Reference")
            st.caption("ISL uses both 1-handed and 2-handed signs. Refer to this chart for accurate hand positioning.")
            
            guide_cols = st.columns(2)
            for idx, letter in enumerate(ALPHABET):
                col = guide_cols[idx % 2]
                kind, desc = ISL_GESTURE_GUIDE.get(letter, ("Two-handed", "ISL letter gesture."))
                badge_color = "#38bdf8" if kind == "One-handed" else "#a78bfa"
                with col:
                    st.markdown(
                        f"""
                        <div style="background:#1e293b; border:1px solid #334155; border-radius:10px; padding:10px 14px; margin-bottom:10px;">
                            <span style="font-size:1.5rem; font-weight:700; color:#5eead4; margin-right:8px;">{letter}</span>
                            <span style="background:{badge_color}; color:#0f172a; font-weight:600; font-size:0.75rem; padding:2px 8px; border-radius:999px;">{kind}</span>
                            <p style="font-size:0.88rem; color:#cbd5e1; margin:6px 0 0 0;">{desc}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

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
