import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import time
import webbrowser
import tempfile
import os
import google.generativeai as genai

# ============ GEMINI CHATBOT SETUP =============
GEMINI_API_KEY = "AIzaSyB9rag4XQdpolvgWDvwdmsb5Dl-pHeI-Yg"  # Paste your key here
genai.configure(api_key=GEMINI_API_KEY)
MODEL = genai.GenerativeModel("gemini-pro")

# =========== ISL MODEL SETUP =============
MODEL_PATH = "alphabet_model.pkl"
LE_PATH = "alphabet_label_encoder.pkl"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)
    return model, le

def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        st.warning(f"Audio error: {e}")

def extract_landmarks(frame, hands):
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    lh, rh = np.zeros(63), np.zeros(63)
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            xyz = []
            for lm in hand_landmarks.landmark:
                xyz.extend([lm.x, lm.y, lm.z])
            if idx == 0:
                lh = xyz
            else:
                rh = xyz
    return np.array(lh + rh)

# ========== SIDEBAR: THEME & CHATBOT ============

from google import genai

# ---- Sidebar Theme Switch & Gemini Chat ----
import streamlit as st
from google import genai

# -------- Sidebar Theme & SignAssistant Chat --------
with st.sidebar:
    st.markdown("### Theme")
    theme = st.radio("", ["Auto", "Light", "Dark"], index=2)
    st.markdown("---")

    # SignAssistant AI (Gemini) Chat
    st.markdown("### 🤖 SignAssistant ")
    if "signassistant_history" not in st.session_state:
        st.session_state.signassistant_history = []

    GEMINI_API_KEY = "AIzaSyB9rag4XQdpolvgWDvwdmsb5Dl-pHeI-Yg"

    with st.form(key="signassistant_chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask me anything…", key="signassistant_input", placeholder="Type your question and press Enter")
        send = st.form_submit_button("Send")

    if send and user_input.strip():
        st.session_state.signassistant_history.append({"role": "user", "text": user_input})
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_input,
            )
            bot_msg = response.text.strip()
        except Exception as e:
            bot_msg = f"SignAssistant Error: {e}"
        st.session_state.signassistant_history.append({"role": "bot", "text": bot_msg})

    st.markdown("---")
    for msg in st.session_state.signassistant_history[-8:]:
        if msg["role"] == "user":
            st.markdown(
                f"<div style='margin-bottom:4px; color:#3b82f6;'><b>You:</b> {msg['text']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='margin-bottom:8px; color:#10b981;'><b>SignAssistant:</b> {msg['text']}</div>", unsafe_allow_html=True)
    st.markdown("---")


# =========== MAIN APP ==============
st.set_page_config("ISL Alphabet Live Recognition", page_icon="🤟")
st.markdown(f"""
    <h1 style='color:#ffd700; font-size:2.4rem;'>🤟 ISL Alphabet Live Recognition</h1>
    <p style='font-size:1.15rem; color:#888;'>Show an Indian Sign Language (ISL) gesture.<br>
    Get real-time <b>word mode</b> or <b>alphabet</b> prediction. <br>
    Speak, search, or save your result. <br>
    </p>
""", unsafe_allow_html=True)

# ---- Mode selection ----
mode = st.radio("Mode", ["📝 Word (spell letters)", "🔤 Single Alphabet"], index=0, horizontal=True)
st.markdown("---")

model, le = load_model()
mp_hands = mp.solutions.hands
FRAME_WINDOW = st.empty()

# Prediction state
if 'pred_chain' not in st.session_state:
    st.session_state.pred_chain = []
if 'final_pred' not in st.session_state:
    st.session_state.final_pred = ""
if 'saved_words' not in st.session_state:
    st.session_state.saved_words = []

# ========== CAMERA or IMAGE UPLOAD ==========
col_cam, col_img = st.columns([2,1])
run_camera = col_cam.button("▶️ Start Camera", use_container_width=True)
stop_camera = col_cam.button("⏹️ Stop Camera", use_container_width=True)
img_upload = col_img.file_uploader("🖼️ Predict from Image (Upload)", type=["jpg","png","jpeg"])

def predict_from_frame(frame, hands):
    features = extract_landmarks(frame, hands)
    pred = "-"
    if features.shape[0] == 126:
        X = features.reshape(1, -1)
        idx = model.predict(X)[0]
        pred = le.inverse_transform([int(idx)])[0]
    return pred

if run_camera:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        prev_pred = "-"
        t_end = time.time() + (10 if mode == "📝 Word (spell letters)" else 4)
        st.info(f"⏰ Timer: {10 if mode == '📝 Word (spell letters)' else 4} seconds. Make your sign!")
        st.session_state.pred_chain = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: st.warning("Couldn't access camera."); break
            frame = cv2.flip(frame, 1)
            frame_disp = frame.copy()
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame_disp, handLms, mp_hands.HAND_CONNECTIONS)
                pred = predict_from_frame(frame, hands)
                if mode == "📝 Word (spell letters)":
                    st.session_state.pred_chain.append(pred)
                caption = f"Live prediction: {pred}"
            else:
                pred = "-"
                caption = "No hand detected"
            FRAME_WINDOW.image(frame_disp, caption=caption, width='stretch')
            if time.time() > t_end or stop_camera:
                break
        cap.release()
    st.session_state.final_pred = (
        "".join(st.session_state.pred_chain) if mode == "📝 Word (spell letters)" else
        (st.session_state.pred_chain[-1] if st.session_state.pred_chain else "")
    )
    st.success(f"⏰ Timer over! Prediction: **{st.session_state.final_pred or '-'}**")
else:
    FRAME_WINDOW.image(np.zeros((480, 640, 3), np.uint8), caption="Camera feed will appear here...", width='stretch')

# IMAGE UPLOAD
if img_upload is not None:
    file_bytes = np.asarray(bytearray(img_upload.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        pred = predict_from_frame(frame, hands)
    st.image(frame, caption=f"Prediction: {pred}", width='stretch')
    st.session_state.final_pred = pred

# ============ EDIT, SPEAK, SEARCH, SAVE ===========
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    edit_val = st.text_input("✏️ Edit/correct prediction:", value=st.session_state.final_pred or "")
    st.session_state.final_pred = edit_val

with col2:
    if st.button("🔊 Speak", use_container_width=True):
        if st.session_state.final_pred:
            speak(st.session_state.final_pred)

with col3:
    st.markdown("#### 🌐 Search")
    search_platform = st.selectbox(
        "Select platform",
        ["Google", "ChatGPT Web", "Gemini", "Perplexity", "Claude"],
        key="search_platform"
    )
    if st.button("🔎 Search", use_container_width=True):
        q = st.session_state.final_pred
        if q:
            url = {
                "Google": f"https://www.google.com/search?q={q}",
                "ChatGPT Web": f"https://chat.openai.com/?q={q}",
                "Gemini": f"https://gemini.google.com/?q={q}",
                "Perplexity": f"https://www.perplexity.ai/search/{q}",
                "Claude": f"https://claude.ai/?prompt={q}"
            }.get(search_platform, f"https://www.google.com/search?q={q}")
            webbrowser.open_new_tab(url)

with col4:
    if st.button("💾 Save", use_container_width=True):
        if st.session_state.final_pred:
            st.session_state.saved_words.append(st.session_state.final_pred)
            st.success(f"Saved: {st.session_state.final_pred}")

with col5:
    st.download_button(
        "⬇️ Download",
        "\n".join(st.session_state.saved_words),
        file_name="predictions.txt",
        mime="text/plain",
        use_container_width=True
    )

# =========== TRANSCRIPT & HISTORY ============
st.markdown("### 📝 Prediction Chain (Word Mode)")
st.write("".join(st.session_state.pred_chain[-30:]) if mode == "📝 Word (spell letters)" else "")
st.markdown("### 💾 Saved words:")
st.write(st.session_state.get('saved_words', []))
st.markdown("""
---
<div style='text-align: center; color: #aaa; font-size: 0.98rem;'>
    Made with 🤍 by <b>Saniya;|&nbsp; ISL AI Project 2025
</div>
""", unsafe_allow_html=True)
