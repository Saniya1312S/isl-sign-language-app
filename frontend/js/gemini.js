/**
 * Page 3: Gemini AI Assistant
 * Manages conversational AI, server environment key discovery, session-memory keys,
 * sentence ingestion from Translator, multilingual generation (English, Hindi, Marathi),
 * and speech pronunciation.
 */

const GeminiPage = {
  history: [],
  selectedLanguage: "English",
  transferredSentence: "",
  isConfigured: false,

  async init() {
    this.bindEvents();
    await this.checkServerStatus();
  },

  bindEvents() {
    // API Key Save / Connect
    const btnSaveKey = document.getElementById("p3-btn-save-key");
    if (btnSaveKey) {
      btnSaveKey.addEventListener("click", () => this.handleSaveApiKey());
    }

    // Clear / Disconnect Key
    const btnClearKey = document.getElementById("p3-btn-clear-key");
    if (btnClearKey) {
      btnClearKey.addEventListener("click", () => this.handleClearApiKey());
    }

    // Language Selector
    const langSelect = document.getElementById("p3-lang-select");
    if (langSelect) {
      langSelect.addEventListener("change", (e) => {
        this.selectedLanguage = e.target.value;
      });
    }

    // Send Message Button
    const btnSend = document.getElementById("p3-btn-send");
    if (btnSend) {
      btnSend.addEventListener("click", () => this.handleSendMessage());
    }

    // Enter Key to Send
    const chatInput = document.getElementById("p3-chat-input");
    if (chatInput) {
      chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          this.handleSendMessage();
        }
      });
    }

    // Clear Conversation
    const btnClearChat = document.getElementById("p3-btn-clear-chat");
    if (btnClearChat) {
      btnClearChat.addEventListener("click", () => {
        this.history = [];
        this.renderChatHistory();
        App.showToast("Conversation cleared.", "info");
      });
    }

    // Send Transferred Sentence Shortcut
    const btnSendTransferred = document.getElementById("p3-btn-send-transferred");
    if (btnSendTransferred) {
      btnSendTransferred.addEventListener("click", () => {
        if (this.transferredSentence) {
          const sent = this.transferredSentence;
          this.transferredSentence = "";
          this.updateTransferredBanner();
          this.sendUserMessage(sent);
        }
      });
    }
  },

  async checkServerStatus() {
    try {
      const resp = await fetch("/api/gemini/status");
      if (!resp.ok) return;
      const data = await resp.json();

      if (data.key_configured) {
        this.isConfigured = true;
        const srcText = data.key_source === "environment" ? "Server .env Configured" : "Connected";
        this.updateConnectionStatus(true, `● Gemini Ready (${srcText})`);
        
        const keyInput = document.getElementById("p3-api-key-input");
        if (keyInput && data.masked_key) {
          keyInput.placeholder = `Configured (${data.masked_key})`;
        }
      } else {
        this.isConfigured = false;
        this.updateConnectionStatus(false, "● No Key Configured");
      }
    } catch (err) {
      console.warn("Could not query Gemini status:", err);
    }
  },

  async handleSaveApiKey() {
    const keyInput = document.getElementById("p3-api-key-input");
    const val = keyInput ? keyInput.value.trim() : "";
    if (!val) {
      App.showToast("Please enter a valid Gemini API key.", "warning");
      return;
    }

    const btnSave = document.getElementById("p3-btn-save-key");
    if (btnSave) {
      btnSave.disabled = true;
      btnSave.textContent = "Connecting...";
    }

    try {
      const resp = await fetch("/api/gemini/set-key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: val })
      });
      const data = await resp.json();

      if (data.connection_success) {
        this.isConfigured = true;
        this.updateConnectionStatus(true, "● Gemini Connected");
        App.showToast("Gemini API key successfully verified and connected!", "success");
        if (keyInput) keyInput.value = "";
      } else {
        this.isConfigured = false;
        this.updateConnectionStatus(false, data.message || "Key Check Failed");
        App.showToast(data.message || "Failed to connect to Gemini API.", "warning");
      }
    } catch (e) {
      App.showToast("Network error testing Gemini key.", "error");
    } finally {
      if (btnSave) {
        btnSave.disabled = false;
        btnSave.textContent = "🔑 Connect API Key";
      }
      await this.checkServerStatus();
    }
  },

  async handleClearApiKey() {
    try {
      await fetch("/api/gemini/clear-key", { method: "POST" });
    } catch (e) {
      // Ignore
    }
    this.isConfigured = false;
    const keyInput = document.getElementById("p3-api-key-input");
    if (keyInput) {
      keyInput.value = "";
      keyInput.placeholder = "AIzaSy...";
    }
    this.updateConnectionStatus(false, "● Key Disconnected");
    App.showToast("Gemini API session disconnected.", "info");
    await this.checkServerStatus();
  },

  updateConnectionStatus(isConnected, message = "") {
    const badge = document.getElementById("p3-connection-badge");
    if (!badge) return;

    if (isConnected) {
      badge.className = "badge badge-success";
      badge.textContent = message || "● Gemini Connected";
    } else {
      badge.className = "badge badge-neutral";
      badge.textContent = message || "● No Key Configured";
    }
  },

  receiveSentence(sentence) {
    this.transferredSentence = sentence;
    this.updateTransferredBanner();

    const chatInput = document.getElementById("p3-chat-input");
    if (chatInput) {
      chatInput.value = sentence;
      chatInput.focus();
    }
  },

  updateTransferredBanner() {
    const banner = document.getElementById("p3-transferred-banner");
    const textEl = document.getElementById("p3-transferred-text");
    if (!banner || !textEl) return;

    if (this.transferredSentence) {
      banner.style.display = "flex";
      textEl.textContent = `"${this.transferredSentence}"`;
    } else {
      banner.style.display = "none";
    }
  },

  async handleSendMessage() {
    const chatInput = document.getElementById("p3-chat-input");
    const text = chatInput ? chatInput.value.trim() : "";
    if (!text) return;

    if (chatInput) chatInput.value = "";
    this.transferredSentence = "";
    this.updateTransferredBanner();
    await this.sendUserMessage(text);
  },

  async sendUserMessage(text) {
    // Add user message to local history
    this.history.push({ role: "user", text });
    this.renderChatHistory();

    // Show typing bubble
    this.renderLoadingBubble();

    try {
      const resp = await fetch("/api/gemini/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          history: this.history.slice(0, -1),
          language: this.selectedLanguage
        })
      });

      this.removeLoadingBubble();

      if (!resp.ok) {
        throw new Error(`Server returned HTTP ${resp.status}`);
      }

      const data = await resp.json();

      if (data.status === "no_key") {
        this.history.push({
          role: "model",
          text: data.reply || "⚠️ Gemini API key is not configured. Please enter your key in the setup panel above or configure GEMINI_API_KEY in .env.",
          isWarning: true
        });
        App.showToast("Gemini key required.", "warning");
      } else if (data.status === "api_error") {
        this.history.push({
          role: "model",
          text: data.reply || "⚠️ Gemini encountered an error communicating with the API.",
          isError: true
        });
        App.showToast("Gemini API communication issue.", "warning");
      } else {
        const replyText = data.reply || "No response received.";
        this.history.push({
          role: "model",
          text: replyText,
          language: data.language || this.selectedLanguage,
          model: data.model
        });
      }

      this.renderChatHistory();
    } catch (err) {
      this.removeLoadingBubble();
      this.history.push({
        role: "model",
        text: `⚠️ Network error: ${err.message}. Please check if the FastAPI server is running.`,
        isError: true
      });
      this.renderChatHistory();
    }
  },

  renderChatHistory() {
    const historyContainer = document.getElementById("p3-chat-history");
    if (!historyContainer) return;

    historyContainer.innerHTML = "";

    if (this.history.length === 0) {
      historyContainer.innerHTML = `
        <div style="text-align: center; color: var(--text-muted); margin: auto; padding: 2rem 1rem;">
          <p style="font-size: 1.4rem; margin-bottom: 0.5rem;">🤖</p>
          <p style="font-size: 1.05rem; font-weight: 600; color: var(--text-primary);">Gemini AI Assistant</p>
          <p style="font-size: 0.88rem; margin-top: 0.25rem; max-width: 380px; margin-left: auto; margin-right: auto;">
            Send your recognized Indian Sign Language sentence from the Translator or type a message below.
          </p>
        </div>
      `;
      return;
    }

    this.history.forEach((msg) => {
      const bubble = document.createElement("div");
      bubble.className = `chat-msg ${msg.role}`;
      if (msg.isError) {
        bubble.style.borderColor = "var(--danger)";
        bubble.style.background = "#fff1f2";
        bubble.style.color = "#9f1239";
      } else if (msg.isWarning) {
        bubble.style.borderColor = "#f59e0b";
        bubble.style.background = "#fffbeb";
        bubble.style.color = "#92400e";
      }

      const textPara = document.createElement("div");
      textPara.style.whiteSpace = "pre-wrap";
      textPara.textContent = msg.text;
      bubble.appendChild(textPara);

      // Add speak button for successful assistant responses
      if (msg.role === "model" && !msg.isError && !msg.isWarning) {
        const actionRow = document.createElement("div");
        actionRow.style.display = "flex";
        actionRow.style.alignItems = "center";
        actionRow.style.gap = "0.75rem";
        actionRow.style.marginTop = "0.5rem";

        const speakBtn = document.createElement("button");
        speakBtn.className = "btn btn-secondary btn-sm";
        speakBtn.style.padding = "3px 8px";
        speakBtn.style.fontSize = "0.8rem";
        speakBtn.innerHTML = "🔊 Listen";
        speakBtn.addEventListener("click", () => {
          const lang = msg.language || this.selectedLanguage;
          const locale = lang === "Hindi" ? "hi-IN" : (lang === "Marathi" ? "mr-IN" : "en-IN");
          App.speak(msg.text, locale);
        });
        actionRow.appendChild(speakBtn);

        if (msg.model) {
          const modelBadge = document.createElement("span");
          modelBadge.style.fontSize = "0.72rem";
          modelBadge.style.color = "var(--text-muted)";
          modelBadge.textContent = `via ${msg.model}`;
          actionRow.appendChild(modelBadge);
        }

        bubble.appendChild(actionRow);
      }

      historyContainer.appendChild(bubble);
    });

    historyContainer.scrollTop = historyContainer.scrollHeight;
  },

  renderLoadingBubble() {
    const historyContainer = document.getElementById("p3-chat-history");
    if (!historyContainer) return;

    const loadingDiv = document.createElement("div");
    loadingDiv.id = "p3-loading-bubble";
    loadingDiv.className = "chat-msg model";
    loadingDiv.innerHTML = "<em>✨ Gemini is generating a response...</em>";
    historyContainer.appendChild(loadingDiv);
    historyContainer.scrollTop = historyContainer.scrollHeight;
  },

  removeLoadingBubble() {
    const loadingDiv = document.getElementById("p3-loading-bubble");
    if (loadingDiv) loadingDiv.remove();
  }
};
