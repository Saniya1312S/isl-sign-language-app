"""Smoke tests for the Streamlit pages (no camera needed)."""
import pytest
from streamlit.testing.v1 import AppTest

from isl.config import RESULTS_DIR

ROOT_APP = str(__import__("pathlib").Path(__file__).resolve().parent.parent)


def run(page):
    at = AppTest.from_file(f"{ROOT_APP}/views/{page}", default_timeout=60)
    at.run()
    return at


@pytest.mark.skipif(not (RESULTS_DIR / "eda.json").exists(), reason="results not generated")
def test_dashboard_renders_all_tabs():
    at = run("dashboard.py")
    assert not at.exception, at.exception
    assert len(at.tabs) >= 5


def test_about_page_renders():
    at = run("about.py")
    assert not at.exception, at.exception


def test_assistant_without_key_shows_friendly_error():
    at = run("assistant_page.py")
    assert not at.exception, at.exception
    at.text_area(key="q_box").set_value("hello")
    at.button[[b.label for b in at.button].index("Send")].click().run()
    assert not at.exception
    assert any("API key" in e.value for e in at.error)


# NOTE: views/live.py contains the WebRTC camera component, which needs a real Streamlit server session and a browser.
# It cannot run inside AppTest, so it is covered by the end-to-end script tests/e2e_live_browser.py instead
# (starts the server, opens Chromium with a fake webcam and checks that a sign is typed).
