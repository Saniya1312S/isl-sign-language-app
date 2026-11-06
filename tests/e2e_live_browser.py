"""
End-to-end test of the LIVE camera page in a real browser (not run by pytest).

It starts the Streamlit server, opens headless Chromium with a FAKE WEBCAM (a video file made from a sample image:
the sign "L" for 5 s, an empty scene for 2 s, then "L" again) and checks that the app really
    * connects the browser camera through WebRTC,
    * recognises the sign and types it,
    * types the letter once (not once per frame) and again after the hand was lowered,
    * keeps working when buttons are pressed.

Setup (once):   pip install playwright && playwright install chromium
Run:            python tests/e2e_live_browser.py [--screenshots docs/screenshots]
"""
import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PORT = 8601
URL = f"http://localhost:{PORT}"


def make_fake_webcam(path: Path) -> None:
    src = cv2.imread(str(ROOT / "samples" / "Indian_L.jpg"))
    canvas = np.zeros((480, 640, 3), np.uint8)
    canvas[:, 80:560] = cv2.resize(src, (480, 480), interpolation=cv2.INTER_CUBIC)
    empty = np.full((480, 640, 3), (40, 30, 25), np.uint8)
    enc = lambda f: cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
    path.write_bytes(b"".join([enc(canvas)] * 75 + [enc(empty)] * 30 + [enc(canvas)] * 75))      # 15 fps: 5 s, 2 s, 5 s


def wait_for_server(timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if urllib.request.urlopen(f"{URL}/_stcore/health", timeout=2).read() == b"ok":
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError("Streamlit did not start")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screenshots", default="")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    cam = Path("/tmp/fake_cam.mjpeg") if Path("/tmp").exists() else ROOT / "fake_cam.mjpeg"
    make_fake_webcam(cam)
    server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(PORT),
                               "--server.headless", "true"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = []
    check = lambda name, ok, extra="": (results.append(bool(ok)), print(("PASS " if ok else "FAIL ") + name, extra))
    try:
        wait_for_server()
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                                              f"--use-file-for-fake-video-capture={cam}", "--autoplay-policy=no-user-gesture-required"])
            ctx = browser.new_context(permissions=["camera"], viewport={"width": 1400, "height": 1100})
            page = ctx.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(URL)
            page.get_by_text("Live Translator").first.wait_for(timeout=60000)
            check("page loads", True)

            # start the camera: the START button lives inside the streamlit-webrtc iframe
            started = False
            for _ in range(40):
                for fr in page.frames:
                    btn = fr.get_by_role("button", name="Start")
                    if btn.count():
                        btn.first.click()
                        started = True
                        break
                if started:
                    break
                time.sleep(0.5)
            check("START button found and clicked", started)

            typed = lambda: page.locator(".typed").first.inner_text().strip()
            t0 = time.time()
            while time.time() - t0 < 40 and "L" not in typed():
                time.sleep(0.5)
            first = typed()
            check("sign recognised and typed", "L" in first, f"text={first!r} after {time.time() - t0:.1f}s")
            time.sleep(1.5)
            check("typed ONCE while the hand is held (no per-frame spam)", typed().replace("your text appears here…", "").count("L") == 1,
                  f"text={typed()!r}")
            if args.screenshots:
                page.screenshot(path=str(Path(args.screenshots) / "live-translator.png"))

            t0 = time.time()                      # wait for the 2nd "L" (hand lowered, then raised)
            while time.time() - t0 < 30 and typed().count("L") < 2:
                time.sleep(0.5)
            check("same letter typed again after the hand was lowered", typed().count("L") >= 2, f"text={typed()!r}")

            def wait_until(cond, seconds=20):
                t_ = time.time()
                while time.time() - t_ < seconds:
                    if cond():
                        return True
                    time.sleep(0.4)
                return False

            before = typed()
            space = page.get_by_role("button", name="␣ Space")
            space.wait_for(timeout=90000)                 # the page may be re-rendering on a slow machine
            space.click(timeout=90000)
            page.get_by_role("button", name="⌫ Delete").click(timeout=90000)
            check("buttons respond while streaming (page stays alive)", wait_until(lambda: page.get_by_role("button", name="🗑 Clear").count() == 1),
                  f"text before={before!r} now={typed()!r}")
            page.get_by_role("button", name="🗑 Clear").click()
            # the (looping) fake webcam may type one new "L" right after clearing, so allow at most a single letter
            check("clear empties the text", wait_until(lambda: len(typed().replace("your text appears here…", "")) <= 1),
                  f"text={typed()!r}")
            check("no JavaScript errors", not errors, str(errors[:2]))
            browser.close()
    finally:
        server.terminate()
    print("\nALL PASSED" if all(results) else "\nSOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
