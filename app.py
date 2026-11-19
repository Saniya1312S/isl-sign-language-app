"""
ISL Recognition System - Main Entry Point.

Run:
    streamlit run app.py
"""
import streamlit as st

pages = [
    st.Page("views/recognition_demo.py", title="ISL Recognition Demo", icon="🤟", default=True),
    st.Page("views/assistant_page.py", title="SignAssistant (Gemini)", icon="🤖"),
    st.Page("views/live.py", title="Live Continuous Translator", icon="🗣️"),
    st.Page("views/dashboard.py", title="Dataset & Audit Dashboard", icon="📊"),
    st.Page("views/about.py", title="About & Scope", icon="ℹ️"),
]

st.navigation(pages).run()
