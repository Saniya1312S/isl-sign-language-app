/**
 * Page 2: Continuous Fingerspelling Sentence Translator
 * Implements 400ms frame sampling, 5-second consistent sign detection,
 * duplicate letter prevention, sentence builder, translation, and speech.
 */

const TranslatorPage = {
  // State
  sentence: "",
  isStreaming: false,
  isPaused: false,
  sampleTimer: null,
  sampleIntervalMs: 400,

  // Stability detection state
  currentCandidate: null,
  candidateStreakCount: 0,
  stabilityElapsedMs: 0,
  requiredStabilityMs: 5000, // 5.0 seconds
  lastAddedLetter: null,
  waitingForSignChange: false,

  init() {
    this.bindEvents();
    this.renderSentence();
  },

  bindEvents() {
    // Camera Controls
    const btnStart = document.getElementById("p2-btn-start-camera");
    if (btnStart) {
      btnStart.addEventListener("click", () => this.startContinuousRecognition());
    }

    const btnPause = document.getElementById("p2-btn-pause-camera");
    if (btnPause) {
      btnPause.addEventListener("click", () => this.togglePause());
    }

    const btnStop = document.getElementById("p2-btn-stop-camera");
    if (btnStop) {
      btnStop.addEventListener("click", () => this.stopContinuousRecognition());
    }

    const btnManualAdd = document.getElementById("p2-btn-manual-add");
    if (btnManualAdd) {
      btnManualAdd.addEventListener("click", () => this.manualAddCurrentLetter());
    }

    // Sentence Controls
    const btnSpace = document.getElementById("p2-btn-space");
    if (btnSpace) {
      btnSpace.addEventListener("click", () => {
        this.sentence += " ";
        this.renderSentence();
        this.waitingForSignChange = false;
      });
    }

    const btnBackspace = document.getElementById("p2-btn-backspace");
    if (btnBackspace) {
      btnBackspace.addEventListener("click", () => {
        if (this.sentence.length > 0) {
          this.sentence = this.sentence.slice(0, -1);
          this.renderSentence();
        }
      });
    }

    const btnClear = document.getElementById("p2-btn-clear");
    if (btnClear) {
      btnClear.addEventListener("click", () => {
        this.sentence = "";
        this.renderSentence();
        this.lastAddedLetter = null;
        this.waitingForSignChange = false;
      });
    }

    const btnCopy = document.getElementById("p2-btn-copy");
    if (btnCopy) {
      btnCopy.addEventListener("click", () => {
        if (this.sentence) {
          navigator.clipboard.writeText(this.sentence);
          App.showToast("Sentence copied to clipboard!", "success");
        }
      });
    }

    // Direct Manual Text Editing
    const sentenceDisplay = document.getElementById("p2-sentence-display");
    if (sentenceDisplay) {
      sentenceDisplay.addEventListener("input", (e) => {
        this.sentence = e.target.innerText;
        this.triggerTranslation();
      });
    }

    // Translation Triggers
    const btnTranslateHi = document.getElementById("p2-btn-trans-hi");
    if (btnTranslateHi) {
      btnTranslateHi.addEventListener("click", () => this.triggerTranslation("Hindi"));
    }

    const btnTranslateMr = document.getElementById("p2-btn-trans-mr");
    if (btnTranslateMr) {
      btnTranslateMr.addEventListener("click", () => this.triggerTranslation("Marathi"));
    }

    // Speech Pronunciation Triggers
    const btnSpeakEn = document.getElementById("p2-btn-speak-en");
    if (btnSpeakEn) {
      btnSpeakEn.addEventListener("click", () => App.speak(this.sentence, "en-IN"));
    }

    const btnSpeakHi = document.getElementById("p2-btn-speak-hi");
    if (btnSpeakHi) {
      btnSpeakHi.addEventListener("click", () => {
        const text = document.getElementById("p2-trans-hi-text")?.textContent;
        if (text && text !== "—") App.speak(text, "hi-IN");
      });
    }

    const btnSpeakMr = document.getElementById("p2-btn-speak-mr");
    if (btnSpeakMr) {
      btnSpeakMr.addEventListener("click", () => {
        const text = document.getElementById("p2-trans-mr-text")?.textContent;
        if (text && text !== "—") App.speak(text, "mr-IN");
      });
    }

    // Send Sentence to Gemini Page
    const btnSendGemini = document.getElementById("p2-btn-send-gemini");
    if (btnSendGemini) {
      btnSendGemini.addEventListener("click", () => {
        if (!this.sentence.trim()) {
          App.showToast("Construct a sentence before sending to Gemini.", "warning");
          return;
        }
        GeminiPage.receiveSentence(this.sentence.trim());
        App.navigateTo("gemini");
        App.showToast("Sentence transferred to Gemini AI Assistant!", "success");
      });
    }
  },

  async startContinuousRecognition() {
    try {
      await page2Camera.startCamera();
      this.isStreaming = true;
      this.isPaused = false;
      this.updateCameraButtons(true);

      const statusBadge = document.getElementById("p2-camera-status");
      if (statusBadge) {
        statusBadge.className = "badge badge-success";
        statusBadge.textContent = "● Camera Active · Sampling Gestures";
      }

      this.resetStability();
      this.startSamplingLoop();
      App.showToast("Webcam recognition active. Hold an ISL sign for 5 seconds.", "info");
    } catch (err) {
      App.showToast("Failed to access camera: " + err.message, "error");
    }
  },

  togglePause() {
    this.isPaused = !this.isPaused;
    const btnPause = document.getElementById("p2-btn-pause-camera");
    const statusBadge = document.getElementById("p2-camera-status");

    if (this.isPaused) {
      if (btnPause) btnPause.textContent = "▶️ Resume";
      if (statusBadge) {
        statusBadge.className = "badge badge-warning";
        statusBadge.textContent = "⏸️ Paused";
      }
      this.resetStability();
    } else {
      if (btnPause) btnPause.textContent = "⏸️ Pause";
      if (statusBadge) {
        statusBadge.className = "badge badge-success";
        statusBadge.textContent = "● Camera Active · Sampling Gestures";
      }
    }
  },

  stopContinuousRecognition() {
    if (this.sampleTimer) {
      clearInterval(this.sampleTimer);
      this.sampleTimer = null;
    }
    page2Camera.stopCamera();
    this.isStreaming = false;
    this.isPaused = false;
    this.updateCameraButtons(false);
    this.resetStability();

    const statusBadge = document.getElementById("p2-camera-status");
    if (statusBadge) {
      statusBadge.className = "badge badge-neutral";
      statusBadge.textContent = "● Camera Offline";
    }
  },

  updateCameraButtons(isActive) {
    const btnStart = document.getElementById("p2-btn-start-camera");
    const btnPause = document.getElementById("p2-btn-pause-camera");
    const btnStop = document.getElementById("p2-btn-stop-camera");

    if (btnStart) btnStart.style.display = isActive ? "none" : "inline-flex";
    if (btnPause) btnPause.style.display = isActive ? "inline-flex" : "none";
    if (btnStop) btnStop.style.display = isActive ? "inline-flex" : "none";
  },

  startSamplingLoop() {
    if (this.sampleTimer) clearInterval(this.sampleTimer);

    this.sampleTimer = setInterval(async () => {
      if (!this.isStreaming || this.isPaused) return;

      const frameB64 = page2Camera.captureFrameAsBase64();
      if (!frameB64) return;

      try {
        const resp = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: frameB64, source: "sampling_loop" })
        });

        if (!resp.ok) return;
        const res = await resp.json();
        this.processSamplingResult(res);
      } catch (err) {
        // Ignore single frame network glitches to keep stream smooth
      }
    }, this.sampleIntervalMs);
  },

  processSamplingResult(res) {
    const liveLetterEl = document.getElementById("p2-live-letter");
    const liveStatusEl = document.getElementById("p2-live-status");
    const timerBar = document.getElementById("p2-stability-progress");
    const timerText = document.getElementById("p2-timer-text");

    const handDetected = res.hand_detected;
    const letter = res.predicted_letter;
    const conf = res.confidence;

    // 1. If no hand is detected
    if (!handDetected) {
      if (liveLetterEl) liveLetterEl.textContent = "—";
      if (liveStatusEl) liveStatusEl.textContent = "No hand visible · Please hold your sign in frame";
      // Hand was removed, clear duplicate suppression flag!
      this.waitingForSignChange = false;
      this.resetStability();
      return;
    }

    // 2. If hand is detected
    if (liveLetterEl) liveLetterEl.textContent = letter || "?";

    // 3. Duplicate letter protection: If user is still holding the same letter they just added
    if (this.waitingForSignChange && letter === this.lastAddedLetter) {
      if (liveStatusEl) {
        liveStatusEl.textContent = `✅ Added '${letter}'! Change sign or relax hand for next letter.`;
      }
      this.resetStability();
      return;
    }

    // If letter changed, clear duplicate suppression flag
    if (letter !== this.lastAddedLetter) {
      this.waitingForSignChange = false;
    }

    // 4. Track Stability
    if (letter && conf >= 0.50) {
      if (letter === this.currentCandidate) {
        this.stabilityElapsedMs += this.sampleIntervalMs;
      } else {
        // Sign changed! Reset stability timer for new sign
        this.currentCandidate = letter;
        this.stabilityElapsedMs = this.sampleIntervalMs;
      }

      const remainingSec = Math.max(0, ((this.requiredStabilityMs - this.stabilityElapsedMs) / 1000)).toFixed(1);
      const progressPct = Math.min(100, (this.stabilityElapsedMs / this.requiredStabilityMs) * 100);

      if (timerBar) timerBar.style.width = `${progressPct}%`;
      if (timerText) timerText.textContent = `Holding '${letter}' stable: ${remainingSec}s remaining`;
      if (liveStatusEl) liveStatusEl.textContent = `Consistency: ${(conf * 100).toFixed(0)}%`;

      // 5. STABILITY REACHED (5 Seconds Held)!
      if (this.stabilityElapsedMs >= this.requiredStabilityMs) {
        this.onStableSignConfirmed(letter);
      }
    } else {
      this.resetStability();
    }
  },

  onStableSignConfirmed(letter) {
    // Append letter
    this.appendLetter(letter);

    // Suppress duplicates until user changes or releases sign
    this.lastAddedLetter = letter;
    this.waitingForSignChange = true;
    this.resetStability();

    // Visual confirmation toast
    App.showToast(`✨ Captured verified letter '${letter}'!`, "success");

    // Audio chime if speech available
    App.playLetterChime();

    const liveStatusEl = document.getElementById("p2-live-status");
    if (liveStatusEl) {
      liveStatusEl.textContent = `✅ Successfully added '${letter}'! Ready for next letter.`;
    }
  },

  resetStability() {
    this.stabilityElapsedMs = 0;
    this.currentCandidate = null;

    const timerBar = document.getElementById("p2-stability-progress");
    const timerText = document.getElementById("p2-timer-text");
    if (timerBar) timerBar.style.width = "0%";
    if (timerText) timerText.textContent = "Hold an ISL sign stable for 5 seconds to add";
  },

  manualAddCurrentLetter() {
    const liveLetterEl = document.getElementById("p2-live-letter");
    const letter = liveLetterEl ? liveLetterEl.textContent.trim() : "";
    if (letter && letter !== "—" && letter !== "?") {
      this.appendLetter(letter);
      App.showToast(`Manually added '${letter}' to sentence!`, "success");
      this.waitingForSignChange = true;
      this.lastAddedLetter = letter;
    } else {
      App.showToast("No valid letter currently detected.", "warning");
    }
  },

  appendLetter(char) {
    this.sentence += char;
    this.renderSentence();
  },

  renderSentence() {
    const displayEl = document.getElementById("p2-sentence-display");
    if (displayEl) {
      displayEl.textContent = this.sentence || "[Fingerspell letters above...]";
    }

    // Auto-update translations whenever sentence changes
    if (this.sentence.trim()) {
      this.triggerTranslation("Hindi");
      this.triggerTranslation("Marathi");
    } else {
      const hiEl = document.getElementById("p2-trans-hi-text");
      const mrEl = document.getElementById("p2-trans-mr-text");
      if (hiEl) hiEl.textContent = "—";
      if (mrEl) mrEl.textContent = "—";
    }
  },

  async triggerTranslation(targetLang = "Hindi") {
    const cleanText = this.sentence.trim();
    if (!cleanText) return;

    try {
      const resp = await fetch("/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: cleanText, target_language: targetLang })
      });

      if (!resp.ok) return;
      const data = await resp.json();

      if (targetLang === "Hindi" || targetLang === "hi") {
        const hiEl = document.getElementById("p2-trans-hi-text");
        const hiSrcEl = document.getElementById("p2-trans-hi-src");
        if (hiEl) hiEl.textContent = data.translated_text;
        if (hiSrcEl) hiSrcEl.textContent = `Source: ${data.source}`;
      } else if (targetLang === "Marathi" || targetLang === "mr") {
        const mrEl = document.getElementById("p2-trans-mr-text");
        const mrSrcEl = document.getElementById("p2-trans-mr-src");
        if (mrEl) mrEl.textContent = data.translated_text;
        if (mrSrcEl) mrSrcEl.textContent = `Source: ${data.source}`;
      }
    } catch (err) {
      console.error("Translation error:", err);
    }
  }
};
