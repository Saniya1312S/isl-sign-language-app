/**
 * Page 1: ISL A-Z Recognition Logic
 * Implements 5-second countdown capture, manual snapshot, image upload, and sample gallery.
 */

const RecognitionPage = {
  countdownTimer: null,
  countdownSeconds: 5,
  isRecognizing: false,
  lastPrediction: null,

  init() {
    this.bindEvents();
    this.loadSampleGallery();
  },

  bindEvents() {
    // Mode Switching
    document.querySelectorAll('input[name="p1-input-mode"]').forEach((el) => {
      el.addEventListener("change", (e) => {
        this.switchInputMode(e.target.value);
      });
    });

    // 5-Second Countdown Start
    const btnStart5s = document.getElementById("p1-btn-start-5s");
    if (btnStart5s) {
      btnStart5s.addEventListener("click", () => this.startFiveSecondCapture());
    }

    // Manual Capture Now
    const btnCaptureNow = document.getElementById("p1-btn-capture-now");
    if (btnCaptureNow) {
      btnCaptureNow.addEventListener("click", () => this.captureAndPredict(true));
    }

    // Stop / Reset
    const btnReset = document.getElementById("p1-btn-reset");
    if (btnReset) {
      btnReset.addEventListener("click", () => this.resetRecognition());
    }

    // File Upload
    const fileInput = document.getElementById("p1-file-input");
    if (fileInput) {
      fileInput.addEventListener("change", (e) => this.handleFileUpload(e));
    }

    // Add to Sentence Button
    const btnAddToSentence = document.getElementById("p1-btn-add-sentence");
    if (btnAddToSentence) {
      btnAddToSentence.addEventListener("click", () => {
        if (this.lastPrediction && this.lastPrediction.predicted_letter) {
          TranslatorPage.appendLetter(this.lastPrediction.predicted_letter);
          App.showToast(`Added '${this.lastPrediction.predicted_letter}' to Sentence!`, "success");
        }
      });
    }
  },

  switchInputMode(mode) {
    const webcamSec = document.getElementById("p1-webcam-section");
    const uploadSec = document.getElementById("p1-upload-section");
    const gallerySec = document.getElementById("p1-gallery-section");

    if (mode === "webcam") {
      webcamSec.style.display = "block";
      uploadSec.style.display = "none";
      gallerySec.style.display = "none";
      page1Camera.startCamera().catch((err) => {
        App.showToast("Could not access camera. Please allow permission.", "error");
      });
    } else if (mode === "upload") {
      webcamSec.style.display = "none";
      uploadSec.style.display = "block";
      gallerySec.style.display = "none";
      page1Camera.stopCamera();
    } else if (mode === "gallery") {
      webcamSec.style.display = "none";
      uploadSec.style.display = "none";
      gallerySec.style.display = "block";
      page1Camera.stopCamera();
    }
  },

  async startFiveSecondCapture() {
    if (this.isRecognizing) return;

    try {
      await page1Camera.startCamera();
    } catch (e) {
      App.showToast("Webcam is required for 5-second capture.", "error");
      return;
    }

    this.isRecognizing = true;
    this.countdownSeconds = 5;

    const overlay = document.getElementById("p1-countdown-overlay");
    const numberEl = document.getElementById("p1-countdown-number");
    const labelEl = document.getElementById("p1-countdown-label");

    if (overlay) overlay.classList.add("active");
    if (numberEl) numberEl.textContent = this.countdownSeconds;
    if (labelEl) labelEl.textContent = "Position your hand gesture...";

    if (this.countdownTimer) clearInterval(this.countdownTimer);

    this.countdownTimer = setInterval(() => {
      this.countdownSeconds -= 1;
      if (numberEl) numberEl.textContent = this.countdownSeconds;

      if (this.countdownSeconds <= 0) {
        clearInterval(this.countdownTimer);
        this.countdownTimer = null;
        if (overlay) overlay.classList.remove("active");
        this.isRecognizing = false;
        // Capture frame at 0
        this.captureAndPredict(false);
      }
    }, 1000);
  },

  async captureAndPredict(isManual = false) {
    const frameB64 = page1Camera.captureFrameAsBase64();
    if (!frameB64) {
      App.showToast("No camera frame available. Please start camera first.", "warning");
      return;
    }

    this.displayLoadingState();

    try {
      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: frameB64, source: isManual ? "manual" : "countdown_5s" })
      });

      if (!resp.ok) {
        throw new Error(`Prediction API returned status ${resp.status}`);
      }

      const data = await resp.json();
      this.lastPrediction = data;
      this.renderPredictionResult(data);
    } catch (err) {
      console.error("Prediction failed:", err);
      App.showToast("Failed to process gesture: " + err.message, "error");
    }
  },

  async handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    this.displayLoadingState();

    try {
      const resp = await fetch("/api/predict-upload", {
        method: "POST",
        body: formData
      });

      if (!resp.ok) {
        throw new Error("File upload prediction failed.");
      }

      const data = await resp.json();
      this.lastPrediction = data;
      this.renderPredictionResult(data);
    } catch (err) {
      console.error("Upload error:", err);
      App.showToast("Error predicting uploaded image: " + err.message, "error");
    }
  },

  async loadSampleGallery() {
    const gridEl = document.getElementById("p1-alphabet-grid");
    if (!gridEl) return;

    gridEl.innerHTML = "";
    const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

    letters.forEach((letter) => {
      const btn = document.createElement("button");
      btn.className = "letter-btn";
      btn.textContent = letter;
      btn.title = `Test authentic sample for letter ${letter}`;
      btn.addEventListener("click", () => this.testSampleLetter(letter, btn));
      gridEl.appendChild(btn);
    });
  },

  async testSampleLetter(letter, buttonElement) {
    document.querySelectorAll(".letter-btn").forEach((b) => b.classList.remove("selected"));
    if (buttonElement) buttonElement.classList.add("selected");

    this.displayLoadingState();

    try {
      const resp = await fetch(`/api/samples/${letter}`);
      if (!resp.ok) {
        throw new Error(`Sample for letter ${letter} could not be loaded.`);
      }

      const data = await resp.json();
      this.lastPrediction = data.prediction;
      this.renderPredictionResult(data.prediction, data.raw_image);
    } catch (err) {
      App.showToast(err.message, "error");
    }
  },

  displayLoadingState() {
    const letterBig = document.getElementById("p1-letter-big");
    const metaEl = document.getElementById("p1-letter-meta");
    const badgeEl = document.getElementById("p1-status-badge");

    if (letterBig) letterBig.textContent = "...";
    if (metaEl) metaEl.textContent = "Analyzing landmarks with MediaPipe & SVM...";
    if (badgeEl) {
      badgeEl.className = "badge badge-neutral";
      badgeEl.textContent = "Processing...";
    }
  },

  renderPredictionResult(result, overrideRawImage = null) {
    const letterBig = document.getElementById("p1-letter-big");
    const metaEl = document.getElementById("p1-letter-meta");
    const badgeEl = document.getElementById("p1-status-badge");
    const previewImg = document.getElementById("p1-preview-img");
    const btnAdd = document.getElementById("p1-btn-add-sentence");
    const top3El = document.getElementById("p1-top3-container");

    const handDetected = result.hand_detected;
    const letter = result.predicted_letter;
    const conf = (result.confidence * 100).toFixed(1);
    const latency = result.processing_time_ms.toFixed(1);
    const nHands = result.n_hands || 0;

    // Show image overlay
    if (previewImg) {
      const imgSrc = result.annotated_image || overrideRawImage;
      if (imgSrc) {
        previewImg.src = imgSrc;
        previewImg.style.display = "block";
      } else {
        previewImg.style.display = "none";
      }
    }

    if (handDetected && letter && result.status !== "low_confidence") {
      if (letterBig) letterBig.textContent = letter;
      if (metaEl) metaEl.textContent = `Confidence: ${conf}% · Latency: ${latency} ms`;
      if (badgeEl) {
        badgeEl.className = "badge badge-success";
        const handText = nHands === 2 ? "2 Hands" : "1 Hand";
        badgeEl.textContent = `● Hand Detected (${handText})`;
      }
      if (btnAdd) {
        btnAdd.style.display = "inline-flex";
        btnAdd.textContent = `➕ Add '${letter}' to Sentence`;
      }
    } else if (handDetected && result.status === "low_confidence") {
      if (letterBig) letterBig.textContent = letter || "?";
      if (metaEl) metaEl.textContent = `Low Confidence: ${conf}% · Please sign clearly`;
      if (badgeEl) {
        badgeEl.className = "badge badge-warning";
        badgeEl.textContent = "● Uncertain Prediction";
      }
      if (btnAdd) btnAdd.style.display = "none";
    } else {
      if (letterBig) letterBig.textContent = "—";
      if (metaEl) metaEl.textContent = "No hand landmarks detected. Hold hand inside frame with good lighting.";
      if (badgeEl) {
        badgeEl.className = "badge badge-danger";
        badgeEl.textContent = "● No Hand Detected";
      }
      if (btnAdd) btnAdd.style.display = "none";
    }

    // Render candidate list
    if (top3El && result.top_predictions && result.top_predictions.length > 0) {
      top3El.innerHTML = "";
      result.top_predictions.forEach(([candLetter, candProb], i) => {
        const medal = ["🥇", "🥈", "🥉"][i] || "•";
        const pill = document.createElement("span");
        pill.className = "badge badge-neutral";
        pill.textContent = `${medal} ${candLetter}: ${(candProb * 100).toFixed(1)}%`;
        top3El.appendChild(pill);
      });
    }
  },

  resetRecognition() {
    if (this.countdownTimer) {
      clearInterval(this.countdownTimer);
      this.countdownTimer = null;
    }
    this.isRecognizing = false;
    const overlay = document.getElementById("p1-countdown-overlay");
    if (overlay) overlay.classList.remove("active");

    const letterBig = document.getElementById("p1-letter-big");
    const metaEl = document.getElementById("p1-letter-meta");
    const badgeEl = document.getElementById("p1-status-badge");
    const btnAdd = document.getElementById("p1-btn-add-sentence");

    if (letterBig) letterBig.textContent = "—";
    if (metaEl) metaEl.textContent = "Ready for capture";
    if (badgeEl) {
      badgeEl.className = "badge badge-neutral";
      badgeEl.textContent = "● Ready";
    }
    if (btnAdd) btnAdd.style.display = "none";
  }
};
