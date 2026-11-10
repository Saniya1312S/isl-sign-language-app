"""Page 2 - SignAssistant: a Gemini assistant a deaf user can talk to by fingerspelling."""
import streamlit as st

from isl import ui
from isl.assistant import DEFAULT_MODEL, SignAssistant, resolve_api_key
from isl.translate import LANGS
from isl.ui import speak

QUICK = ["Where is the nearest hospital?", "How do I ask for help politely?", "What should I do if I feel dizzy?",
         "Explain this in simple words: appointment", "How do I say thank you to a teacher?"]


def _secrets() -> dict:
    try:
        return dict(st.secrets)
    except Exception:                      # no secrets file -> fine
        return {}


def main() -> None:
    ui.inject_css()
    ui.hero("🤖 SignAssistant",
            "Spell your question with your hands on the Live Translator page, send it here, and get a short, simple "
            "answer that is shown in large text and can be read aloud.")
    secrets = _secrets()

    with st.expander("🔑 Gemini connection", expanded=not resolve_api_key("", secrets)):
        st.caption("The key is used only for requests from this browser session. It is **never** saved to disk "
                   "or to the repository. Get a free key at Google AI Studio.")
        key_in = st.text_input("Gemini API key", type="password", key="gemini_key_input",
                               placeholder="(or set GEMINI_API_KEY in secrets / environment)")
        model = st.text_input("Model name", value=str(secrets.get("GEMINI_MODEL", DEFAULT_MODEL)), key="gemini_model",
                              help="Gemini model names change over time. Use the button below to list what your key can use.")
        api_key = resolve_api_key(key_in, secrets)
        if st.button("Find models available to my key", disabled=not api_key):
            try:
                st.session_state["found_models"] = SignAssistant(api_key).list_models()
            except Exception as exc:
                st.error(f"Could not list models: {exc}")
        if st.session_state.get("found_models"):
            st.write("Available:", ", ".join(st.session_state["found_models"][:15]))
    api_key = resolve_api_key(st.session_state.get("gemini_key_input", ""), secrets)

    st.session_state.setdefault("chat", [])
    c1, c2 = st.columns([3, 1])
    reply_lang = c2.selectbox("Reply language", list(LANGS))
    engine = c2.radio("Voice", ["Browser voice", "Server voice (gTTS)"])
    draft = st.session_state.pop("assistant_draft", "")
    question = c1.text_area("Your question", value=draft, height=90, key="q_box",
                            placeholder="Type, or send your signed text from the Live Translator…")
    q_cols = c1.columns(len(QUICK))
    for col, q in zip(q_cols, QUICK):
        if col.button(q[:22] + "…", key=f"q_{q}", help=q):
            question = q
    send = c1.button("Send", type="primary") or question in QUICK

    voice_slot = c2.container()
    if send and question.strip():
        bot = SignAssistant(api_key, model=st.session_state.get("gemini_model", DEFAULT_MODEL))
        with st.spinner("Thinking…"):
            reply = bot.ask(question, st.session_state["chat"], reply_lang)
        st.session_state["chat"] += [("user", question)]
        if reply.ok:
            st.session_state["chat"] += [("assistant", reply.text)]
        else:
            st.error(reply.error)

    for role, text in st.session_state["chat"][-10:]:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(f"<div style='font-size:1.25rem'>{text}</div>" if role != "user" else text, unsafe_allow_html=True)
    last = next((t for r, t in reversed(st.session_state["chat"]) if r == "assistant"), "")
    if last and c2.button("🔊 Read the last answer aloud"):
        speak(voice_slot, last, LANGS[reply_lang], engine)
    if st.session_state["chat"] and c2.button("Clear conversation"):
        st.session_state["chat"] = []
        st.rerun()
    ui.note("SignAssistant can make mistakes and is **not** a medical, legal or emergency service. "
            "In an emergency call 112. Your question text is sent to Google's Gemini API.")


main()
