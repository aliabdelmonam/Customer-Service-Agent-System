const chatArea = document.getElementById("chat-area");
const input = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const micBtn = document.getElementById("mic-btn");
const settingsBtn = document.getElementById("settings-btn");
const settingsPanel = document.getElementById("settings-panel");
const settingsSave = document.getElementById("settings-save");
const settingsCancel = document.getElementById("settings-cancel");
const clearBtn = document.getElementById("clear-btn");
const apiUrlInput = document.getElementById("api-url");
const apiEndpointInput = document.getElementById("api-endpoint");

let sessionId = null;
let mediaRecorder = null;
let microphoneStream = null;
let recordedChunks = [];

const GREETING_MESSAGES = [
  "Hello, this is Lol from Lolo Company Customer Service.",
  "How can I help you today?",
];

const STORAGE_KEY = "serenity_settings";
const defaults = { apiUrl: "http://localhost:8000", endpoint: "/api/v1/chat" };

function loadSettings() {
  try {
    return { ...defaults, ...JSON.parse(localStorage.getItem(STORAGE_KEY)) };
  } catch {
    return { ...defaults };
  }
}

function saveSettings(s) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
}

let settings = loadSettings();

// ── Settings panel ──

settingsBtn.addEventListener("click", () => {
  const open = settingsPanel.classList.toggle("open");
  if (open) {
    apiUrlInput.value = settings.apiUrl;
    apiEndpointInput.value = settings.endpoint;
    apiUrlInput.focus();
  }
});

settingsCancel.addEventListener("click", () =>
  settingsPanel.classList.remove("open")
);

settingsSave.addEventListener("click", () => {
  settings.apiUrl = apiUrlInput.value.replace(/\/+$/, "") || defaults.apiUrl;
  settings.endpoint = apiEndpointInput.value || defaults.endpoint;
  saveSettings(settings);
  settingsPanel.classList.remove("open");
});

// ── Input handling ──

input.addEventListener("input", () => {
  sendBtn.disabled = !input.value.trim();
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 120) + "px";
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (input.value.trim()) send();
  }
});

sendBtn.addEventListener("click", send);
micBtn.addEventListener("click", toggleRecording);

// ── Quick prompts ──

document.addEventListener("click", (e) => {
  if (e.target.classList.contains("prompt-chip")) {
    input.value = e.target.dataset.prompt;
    input.dispatchEvent(new Event("input"));
    send();
  }
});

// ── Clear chat ──

clearBtn.addEventListener("click", () => {
  sessionId = null;
  showGreeting();
});

// ── Chat logic ──

function removeWelcome() {
  const welcome = chatArea.querySelector(".welcome");
  if (welcome) welcome.remove();
}

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = `message ${role}`;

  const avatarLabel = role === "user" ? "You" : "S";
  div.innerHTML = `
    <div class="avatar">${avatarLabel}</div>
    <div class="bubble">${escapeHtml(text)}</div>`;

  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
  return div;
}

function showGreeting() {
  chatArea.innerHTML = "";
  GREETING_MESSAGES.forEach((message) => addMessage("bot", message));
}

// A greeting is a UI-only message: it does not call the API or create a ticket.
showGreeting();

function setRecordingState(recording) {
  micBtn.classList.toggle("recording", recording);
  micBtn.title = recording ? "Stop and send voice message" : "Start voice message";
  micBtn.setAttribute("aria-label", micBtn.title);
}

async function toggleRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    addError("Voice messages are not supported by this browser.");
    return;
  }

  try {
    microphoneStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(microphoneStream);
    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) recordedChunks.push(event.data);
    });
    mediaRecorder.addEventListener("stop", transcribeRecording, { once: true });
    mediaRecorder.start();
    setRecordingState(true);
  } catch (error) {
    addError("Microphone access was not granted. Please allow microphone access and try again.");
  }
}

async function transcribeRecording() {
  setRecordingState(false);
  microphoneStream?.getTracks().forEach((track) => track.stop());
  microphoneStream = null;

  const audio = new Blob(recordedChunks, { type: mediaRecorder?.mimeType || "audio/webm" });
  mediaRecorder = null;
  if (!audio.size) {
    addError("No audio was recorded. Please try again.");
    return;
  }

  showTyping();
  try {
    const formData = new FormData();
    formData.append("audio", audio, "voice-message.webm");
    const response = await fetch(`${settings.apiUrl}/api/v1/asr/transcribe`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(`Speech recognition returned ${response.status}.`);
    }
    const { text } = await response.json();
    if (!text?.trim()) throw new Error("No speech was recognized.");
    input.value = text.trim();
    input.dispatchEvent(new Event("input"));
    hideTyping();
    await send();
  } catch (error) {
    hideTyping();
    addError(error.message || "Voice transcription failed. Please try again.");
  }
}

// `var` is intentionally used here so an early UI callback cannot hit the
// temporal-dead-zone error produced by a `let` declaration.
var messageId = 0;

function addBotMessage(text, userMessage) {
  const id = ++messageId;
  const div = document.createElement("div");
  div.className = "message bot";

  div.innerHTML = `
    <div class="avatar">S</div>
    <div class="bubble-wrap">
      <div class="bubble markdown">${marked.parse(text)}</div>
      <div class="feedback-btns" data-id="${id}">
        <button class="fb-btn" data-vote="up" title="Helpful">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>
        </button>
        <button class="fb-btn" data-vote="down" title="Not helpful">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z"/><path d="M17 2h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"/></svg>
        </button>
      </div>
    </div>`;

  div.querySelectorAll(".fb-btn").forEach((btn) => {
    btn.addEventListener("click", () => sendFeedback(btn, userMessage, text));
  });

  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
}

async function sendFeedback(btn, userMessage, botResponse) {
  const wrap = btn.closest(".feedback-btns");
  if (wrap.classList.contains("voted")) return;

  const vote = btn.dataset.vote;
  wrap.classList.add("voted");
  btn.classList.add("selected");

  try {
    await fetch(settings.apiUrl + "/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        vote,
        user_message: userMessage,
        bot_response: botResponse,
      }),
    });
  } catch {}
}

function addError(text) {
  const div = document.createElement("div");
  div.className = "message bot";
  div.innerHTML = `
    <div class="avatar">S</div>
    <div class="bubble error-bubble">${escapeHtml(text)}</div>`;
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
}

function showTyping() {
  const div = document.createElement("div");
  div.className = "message bot";
  div.id = "typing";
  div.innerHTML = `
    <div class="avatar">S</div>
    <div class="bubble">
      <div class="typing-indicator"><span></span><span></span><span></span></div>
    </div>`;
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
}

function hideTyping() {
  const el = document.getElementById("typing");
  if (el) el.remove();
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

async function send() {
  const text = input.value.trim();
  if (!text) return;

  removeWelcome();
  addMessage("user", text);

  input.value = "";
  input.style.height = "auto";
  sendBtn.disabled = true;
  input.focus();

  showTyping();

  try {
    const url = settings.apiUrl + settings.endpoint;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: text,
        session_id: sessionId,
      }),
    });

    hideTyping();

    if (!res.ok) {
      if (res.status === 429) {
        throw new Error(
          "You're sending messages too quickly. Please wait a moment and try again."
        );
      }
      const errText = await res.text().catch(() => "");
      throw new Error(
        `Server returned ${res.status}${errText ? ": " + errText : ""}`
      );
    }

    const data = await res.json();

    if (data.session_id) {
      sessionId = data.session_id;
    }

    const replies = Array.isArray(data.messages) && data.messages.length
      ? data.messages
      : [data.response || data.answer || data.message || data.reply || data.text];

    replies
      .filter((reply) => typeof reply === "string" && reply.trim())
      .forEach((reply) => addBotMessage(reply, text));
  } catch (err) {
    hideTyping();
    if (err.name === "TypeError" && err.message === "Failed to fetch") {
      addError(
        `Could not connect to ${settings.apiUrl}. Make sure your backend is running and CORS is enabled.`
      );
    } else {
      addError(err.message);
    }
  }
}
