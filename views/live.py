"""Page 1 - Live Translator: camera (live / snapshot / upload) -> spelling -> translation -> voice."""
import html

import cv2
import numpy as np
import streamlit as st

from isl import ui
from isl.ui import speak
from isl.config import ALPHABET, SAMPLES_DIR
from isl.hands import HandDetector
from isl.live import SignVideoProcessor, analyse_image
from isl.personalise import CalibrationSet, base_training_set, personalise
from isl.runtime import MODE_LABELS, build_predictors, load_bundles
from isl.spelling import SpellingBuffer
from isl.translate import LANGS, translate
from isl.vocab import suggest
from isl.voice import spell_out

# STUN server lets the browser find a route to the server. On very strict networks a TURN server is
# also needed (see README > Troubleshooting).
RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


@st.cache_resource(show_spinner="Loading models…")
def get_bundles():
    return load_bundles()


@st.cache_resource(show_spinner="Preparing training data (first time only)…")
def get_base_training_set():
    return base_training_set()


@st.fragment(run_every=0.5)
def calibration_status(ctx) -> None:
    cal: CalibrationSet = st.session_state.calibration
    proc = ctx.video_processor if ctx else None
    rec = proc.recording if proc else None
    if rec:
        st.warning(f"⏺ Recording {rec[0]} … {rec[1]} samples left. Hold the sign steady.")
    counts = cal.counts()
    st.caption("Samples so far: " + (", ".join(f"{k}×{v}" for k, v in counts.items()) if counts else "none yet"))


def teach_panel(ctx, bundles, buffer) -> None:
    with st.expander("🎯 Teach the app your hands (recommended — raises accuracy for you)"):
        st.caption("The shared model was trained on a studio dataset and is much less accurate for other people and rooms "
                   "(see Dashboard). Record ~12 samples of each letter you want to use, then train. Only landmark numbers are "
                   "kept - no pictures - and nothing leaves your browser session.")
        cal: CalibrationSet = st.session_state.calibration
        c1, c2, c3 = st.columns(3)
        letter = c1.selectbox("Letter to record", ALPHABET, key="cal_letter")
        if c2.button("⏺ Record 12 samples", disabled=not (ctx and ctx.video_processor), width="stretch"):
            ctx.video_processor.calibration = cal
            ctx.video_processor.start_recording(letter, 12)
        if c3.button("🗑 Forget all samples", width="stretch"):
            cal.clear()
            st.session_state.pop("personal_bundle", None)
        if ctx and ctx.video_processor:
            ctx.video_processor.calibration = cal
        calibration_status(ctx)
        t1, t2 = st.columns(2)
        # (not disabled when empty: recording finishes in the background, after this script run, so a disabled
        #  state computed here would be stale)
        if t1.button("🧠 Train my personal model", type="primary", width="stretch"):
            if cal.total() == 0:
                st.warning("Record at least one letter first.")
            else:
                with st.spinner("Training (a few seconds)…"):
                    X, y = get_base_training_set()
                    st.session_state["personal_bundle"] = personalise(bundles["alphabet"], X, y, cal)
                st.success(f"Personal model ready ({cal.total()} samples). Tick “Use my personal model” in Settings.")
        up = t2.file_uploader("Load saved samples (CSV)", type=["csv"], key="cal_up", label_visibility="collapsed")
        if up is not None and st.session_state.get("cal_up_done") != up.file_id:
            try:
                n = cal.load_csv(up.getvalue())
                st.session_state["cal_up_done"] = up.file_id
                st.success(f"Loaded {n} samples.")
            except Exception as exc:
                st.error(str(exc))
        if cal.total():
            st.download_button("⬇ Download my samples (CSV)", cal.to_csv(), "my_isl_samples.csv", "text/csv")


@st.cache_resource
def get_static_detector():
    return HandDetector(static_image_mode=True)


def show_result(img_bgr: np.ndarray, predictors: dict, mode: str, key: str, buffer: SpellingBuffer) -> None:
    annotated, pred = analyse_image(img_bgr, get_static_detector(), predictors[mode])
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
    if pred.label:
        st.success(f"Recognised **{pred.label}** with {pred.confidence:.0%} confidence")
        if st.button(f"➕ Add “{pred.label}” to my text", key=f"add_{key}"):
            buffer.append(pred.label)
            st.rerun()
    elif pred.reason == "no_hand":
        st.warning("No hand found. Use good light and keep your hand fully inside the picture.")
    else:
        st.info(f"Not sure (best guess {pred.top_k[0][0]} at {pred.top_k[0][1]:.0%}). Try a clearer picture.")
    if pred.top_k:
        st.caption("Top guesses: " + "  ·  ".join(f"{l} {p:.0%}" for l, p in pred.top_k))


@st.fragment(run_every=0.5)
def text_fragment(out_lang: str, mode: str) -> None:
    """Refreshes twice a second: typed text, hold-progress, suggestions, translation."""
    buf: SpellingBuffer = st.session_state.buffer
    snap = buf.snapshot()
    shown = html.escape(snap.text) if snap.text else '<span style="opacity:.35">your text appears here…</span>'
    st.markdown(f'<div class="typed">{shown}</div>', unsafe_allow_html=True)
    if snap.candidate:
        st.progress(snap.progress, text=f"Hold “{snap.candidate}” … {snap.progress:.0%}")
    else:
        st.progress(0.0, text="Show a sign and hold it steady")

    if mode != "hindi":
        for w in suggest(snap.current_word):
            if st.button(w, key=f"sg_{w}", help="Auto-complete"):
                buf.replace_current_word(w.upper())
    code = LANGS[out_lang]
    if code != "en" and snap.text.strip() and (snap.text.endswith(" ") or mode == "words"):
        tr = translate(snap.text.strip(), code)
        st.session_state["last_translation"] = tr
    tr = st.session_state.get("last_translation")
    if code != "en" and tr is not None:
        if tr.text:
            st.markdown(f"**{out_lang}:** {tr.text}  <span class='chip'>{tr.source}</span>", unsafe_allow_html=True)
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
    c1.metric("Frames / s", s["fps"])
    c2.metric("Processing (ms)", s["latency_ms"])
    c3.metric("Hands seen", last.n_hands)


def main() -> None:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    ui.inject_css()
    ui.hero("🤟 Live Translator",
            "Sign letters in front of your camera. Hold a sign steady to type it, then listen to it in English, "
            "Hindi or Marathi.")
    bundles = get_bundles()
    if "alphabet" not in bundles:
        st.error("Alphabet model not found. Run `python scripts/train_alphabet.py` first (see README).")
        st.stop()
    if "buffer" not in st.session_state:
        st.session_state.buffer = SpellingBuffer()
    if "calibration" not in st.session_state:
        st.session_state.calibration = CalibrationSet()
    buffer: SpellingBuffer = st.session_state.buffer

    with st.expander("⚙️ Settings", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        mode = c1.selectbox("Recognition mode", list(bundles), format_func=MODE_LABELS.get)
        hold_s = c2.slider("Hold a sign for (seconds)", 0.3, 2.5, 0.8, 0.1,
                           help="How long a sign must stay steady before it is typed. Longer = fewer mistakes.")
        min_conf = c3.slider("Minimum confidence", 0.30, 0.95, 0.60, 0.05,
                             help="Predictions below this are ignored. See Dashboard > Errors for the trade-off.")
        mirror = c4.checkbox("Mirror (selfie) view", True)
        v1, v2, v3 = st.columns(3)
        out_lang = v1.selectbox("Speak & translate to", list(LANGS))
        engine = v2.radio("Voice engine", ["Browser voice", "Server voice (gTTS)"], horizontal=True,
                          help="Browser voice works offline if your device has that language installed. "
                               "Server voice needs internet but works for Hindi/Marathi everywhere.")
        auto_speak = v3.checkbox("Speak each word when I press Space", False)
        has_personal = "personal_bundle" in st.session_state
        use_personal = st.checkbox("Use my personal model", value=has_personal, disabled=not has_personal,
                                   help="Available after you train it in “Teach the app your hands”.")
        missing = [MODE_LABELS[m] for m in MODE_LABELS if m not in bundles]
        if missing:
            st.caption("Not installed (optional): " + ", ".join(missing) + ". See README > “Add your own words / Hindi model”.")
    active = dict(bundles)
    if has_personal and use_personal:
        active["alphabet"] = st.session_state["personal_bundle"]
    predictors = build_predictors(active, min_conf)

    left, right = st.columns([3, 2], gap="large")
    voice_slot = right.container()

    with left:
        t_live, t_snap, t_up = st.tabs(["🎥 Live camera", "📸 Snapshot", "🖼️ Upload / samples"])
        with t_live:
            st.caption("Your browser asks permission to use the camera; video is processed by this app and not stored.")
            ctx = webrtc_streamer(
                key="isl-live", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIG,
                media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
                video_processor_factory=SignVideoProcessor, async_processing=True)
            if ctx.video_processor:
                ctx.video_processor.buffer = buffer
                ctx.video_processor.configure(predictors, mode, hold_frames=max(3, int(hold_s * 15)), hold_seconds=hold_s,
                                              min_confidence=min_conf, mirror_view=mirror)
            stats_fragment(ctx)
            teach_panel(ctx, bundles, buffer)
            st.caption("Tips: good light, plain background, hand fully visible, one letter at a time. "
                       "To type a double letter (LL) lower your hand briefly between the two signs.")
        with t_snap:
            shot = st.camera_input("Take a picture of your sign", help="Use this if live video does not connect on your network.")
            if shot is not None:
                img = cv2.imdecode(np.frombuffer(shot.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                show_result(img, predictors, mode, "snap", buffer)
        with t_up:
            up = st.file_uploader("Upload a picture of a sign", type=["jpg", "jpeg", "png"])
            samples = sorted(p.name for p in SAMPLES_DIR.glob("*.jpg"))
            pick = st.selectbox("…or try a sample image", ["—"] + samples,
                                help="Indian_* = studio dataset, External_* = other people/backgrounds")
            img = None
            if up is not None:
                img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            elif pick != "—":
                img = cv2.imread(str(SAMPLES_DIR / pick))
            if img is not None:
                show_result(img, predictors, mode, "up", buffer)

    with right:
        text_fragment(out_lang, mode)
        b1, b2, b3 = st.columns(3)
        if b1.button("␣ Space", width="stretch"):
            word = buffer.current_word
            buffer.add_space()
            if auto_speak and word:
                speak(voice_slot, word, LANGS[out_lang] if out_lang != "English" else "en", engine)
        if b2.button("⌫ Delete", width="stretch"):
            buffer.backspace()
        if b3.button("🗑 Clear", width="stretch"):
            buffer.clear()
            st.session_state.pop("last_translation", None)
        s1, s2 = st.columns(2)
        if s1.button("🔊 Speak", width="stretch", type="primary"):
            text, lang = buffer.text.strip(), "en"
            if out_lang != "English":
                tr = translate(text, LANGS[out_lang])
                if tr.text:
                    text, lang = tr.text, LANGS[out_lang]
                else:
                    st.warning(f"Could not translate ({tr.error}); speaking English.")
            speak(voice_slot, text, lang, engine)
        if s2.button("🔤 Spell aloud", width="stretch", help="Reads the letters one by one: C. A. T."):
            speak(voice_slot, spell_out(buffer.text, "en" if out_lang == "English" else LANGS[out_lang]),
                  LANGS[out_lang], engine)
        if st.button("🤖 Ask SignAssistant with this text", width="stretch"):
            st.session_state["assistant_draft"] = buffer.text.strip()
            st.switch_page("views/assistant_page.py")


main()
