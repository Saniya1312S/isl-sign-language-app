"""
Open every page of the app in headless Chromium, check that it renders without errors and save the screenshots
used in the README (docs/screenshots/). Also exercises: sample-image recognition, the "Teach the app" flow with a
fake webcam, and the SignAssistant error message when no API key is set.

Setup (once):  pip install playwright && playwright install chromium
Run:           python scripts/make_screenshots.py
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from e2e_live_browser import PORT, URL, make_fake_webcam, wait_for_server    # noqa: E402

OUT = ROOT / "docs" / "screenshots"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="", help="use an already running app instead of starting one")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright
    app_url = args.url or URL
    OUT.mkdir(parents=True, exist_ok=True)
    cam = Path("/tmp/fake_cam.mjpeg") if Path("/tmp").exists() else ROOT / "fake_cam.mjpeg"
    make_fake_webcam(cam)
    server = None if args.url else subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(PORT), "--server.headless", "true"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = []
    check = lambda name, ok, extra="": (results.append(bool(ok)), print(("PASS " if ok else "FAIL ") + name, extra))
    try:
        if not args.url:
            wait_for_server()
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                                              f"--use-file-for-fake-video-capture={cam}", "--autoplay-policy=no-user-gesture-required"])
            ctx = browser.new_context(permissions=["camera"], viewport={"width": 1400, "height": 1000})
            page = ctx.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            exceptions = lambda: page.locator('[data-testid="stException"]').count()
            go = lambda label: (page.get_by_role("link", name=label).first.click(timeout=60000), time.sleep(3))
            tab = lambda name: (page.get_by_role("tab", name=name).click(timeout=60000), time.sleep(1))

            page.goto(app_url)
            page.get_by_text("Live Translator").first.wait_for(timeout=60000)
            time.sleep(5)                                          # let the first render settle

            # ---- upload / samples tab ------------------------------------------------------------------------
            tab("Upload / samples")
            pick = lambda label, value: (page.get_by_test_id("stSelectbox").filter(has_text=label).first.click(),
                                         page.keyboard.type(value), time.sleep(0.5), page.keyboard.press("Enter"), time.sleep(1.5))
            pick("try a sample", "Indian_V")
            page.get_by_text("Recognised").first.wait_for(timeout=60000)
            check("sample image recognised", "Recognised" in page.inner_text("body"), "(Indian_V.jpg)")
            page.screenshot(path=str(OUT / "snapshot-upload.png"))

            # ---- teach flow (fake webcam shows the sign L) -----------------------------------------------------
            tab("Live camera")
            started = False
            for _ in range(40):
                for fr in page.frames:
                    b = fr.get_by_role("button", name="Start")
                    if b.count():
                        b.first.click()
                        started = True
                        break
                if started:
                    break
                time.sleep(0.5)
            page.get_by_text("Teach the app your hands").click()
            time.sleep(8)                                          # let the stream connect
            pick("Letter to record", "L")
            page.get_by_role("button", name="⏺ Record 12 samples").click()
            t0 = time.time()
            while time.time() - t0 < 60 and "L×12" not in page.inner_text("body"):
                time.sleep(1)
            check("12 samples recorded from the live camera", "L×12" in page.inner_text("body"))
            page.get_by_role("button", name="🧠 Train my personal model").click()
            page.get_by_text("Personal model ready").wait_for(timeout=120000)
            check("personal model trained", True)
            page.screenshot(path=str(OUT / "teach-panel.png"))

            # ---- other pages -----------------------------------------------------------------------------------
            go("SignAssistant")
            page.get_by_role("button", name="Send").wait_for(timeout=20000)
            page.get_by_role("textbox", name="Your question").fill("where is the hospital")
            page.keyboard.press("Control+Enter")
            page.get_by_role("button", name="Send").click()
            time.sleep(3)
            check("assistant explains the missing API key", "API key" in page.inner_text("body"))
            page.screenshot(path=str(OUT / "assistant.png"))

            go("Data & Model Dashboard")
            page.get_by_role("tab", name="Dataset").wait_for(timeout=30000)
            for i, (tab_name, name) in enumerate([("Dataset", "dashboard-dataset"), ("Evaluation honesty", "dashboard-evaluation"),
                                             ("Models", "dashboard-models"), ("Errors & confidence", "dashboard-errors"),
                                             ("Speed & old-vs-new", "dashboard-speed")]):
                page.get_by_role("tab", name=tab_name).click(timeout=60000)
                time.sleep(4)
                check(f"dashboard tab renders: {tab_name}", exceptions() == 0 and page.locator(".js-plotly-plot").count() > 0,
                      f"charts={page.locator('.js-plotly-plot').count()}")
                page.screenshot(path=str(OUT / f"{name}.png"), full_page=False)

            go("About & Scope")
            time.sleep(3)
            check("about page renders", exceptions() == 0 and "What is implemented" in page.inner_text("body"))
            page.screenshot(path=str(OUT / "about.png"))
            check("no JavaScript errors", not errors, str(errors[:2]))
            browser.close()
    finally:
        if server:
            server.terminate()
    print("\nALL PASSED" if all(results) else "\nSOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
