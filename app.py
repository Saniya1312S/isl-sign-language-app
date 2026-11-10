"""
ISL Gesture-to-Speech - Streamlit entry point.

Run:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="ISL Gesture-to-Speech", page_icon="🤟", layout="wide")

pages = [
    st.Page("views/live.py", title="Live Translator", icon="🤟", default=True),
    st.Page("views/assistant_page.py", title="SignAssistant", icon="🤖"),
    st.Page("views/dashboard.py", title="Data & Model Dashboard", icon="📊"),
    st.Page("views/about.py", title="About & Scope", icon="ℹ️"),
]
st.navigation(pages).run()
