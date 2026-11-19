/**
 * Global App Controller
 * Manages 3-page tab switching, notifications, and client-side speech synthesis.
 */

const App = {
  activePage: "home",

  init() {
    this.bindNavigation();
    RecognitionPage.init();
    TranslatorPage.init();
    GeminiPage.init();
    this.checkBackendHealth();
  },

  bindNavigation() {
    document.querySelectorAll(".nav-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        const targetPage = btn.getAttribute("data-page");
        if (targetPage) {
          this.navigateTo(targetPage);
        }
      });
    });
  },

  navigateTo(pageId) {
    this.activePage = pageId;

    // Update nav buttons
    document.querySelectorAll(".nav-btn").forEach((btn) => {
      if (btn.getAttribute("data-page") === pageId) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });

    // Update page sections
    document.querySelectorAll(".page-section").forEach((sec) => {
      if (sec.id === `page-${pageId}`) {
        sec.classList.add("active");
      } else {
        sec.classList.remove("active");
      }
    });

    // Handle camera transitions when switching pages
    if (pageId === "home") {
      page2Camera.stopCamera();
    } else if (pageId === "translator") {
      page1Camera.stopCamera();
    } else if (pageId === "gemini") {
      page1Camera.stopCamera();
      page2Camera.stopCamera();
      GeminiPage.checkServerStatus();
    } else if (pageId === "about") {
      page1Camera.stopCamera();
      page2Camera.stopCamera();
    }

    window.scrollTo({ top: 0, behavior: "smooth" });
  },

  showToast(message, type = "info") {
    let container = document.getElementById("toast-container");
    if (!container) {
      container = document.createElement("div");
      container.id = "toast-container";
      container.className = "toast-container";
      document.body.appendChild(container);
    }

    const toast = document.createElement("div");
    toast.className = `toast ${type}`;

    let icon = "ℹ️";
    if (type === "success") icon = "✅";
    if (type === "warning") icon = "⚠️";
    if (type === "error") icon = "❌";

    toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(10px)";
      toast.style.transition = "all 0.3s ease";
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  },

  speak(text, lang = "en-IN") {
    const cleanText = (text || "").trim();
    if (!cleanText) return;

    if (!("speechSynthesis" in window)) {
      this.showToast("Speech synthesis is not supported in this browser.", "warning");
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.lang = lang;
    utterance.rate = 0.95;

    const voices = window.speechSynthesis.getVoices();
    const voice = voices.find((v) => v.lang === lang || v.lang.startsWith(lang.slice(0, 2)));
    if (voice) {
      utterance.voice = voice;
    }

    window.speechSynthesis.speak(utterance);
  },

  playLetterChime() {
    try {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(587.33, audioCtx.currentTime); // D5
      osc.frequency.exponentialRampToValueAtTime(880, audioCtx.currentTime + 0.15); // A5
      gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.25);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.25);
    } catch (e) {
      // AudioContext not allowed or unsupported
    }
  },

  async checkBackendHealth() {
    try {
      const resp = await fetch("/api/health");
      if (resp.ok) {
        const data = await resp.json();
        console.log("Backend healthy:", data);
      }
    } catch (e) {
      console.warn("Backend health check failed:", e);
    }
  }
};

// Initialize application on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  App.init();
});
