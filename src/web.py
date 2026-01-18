from __future__ import annotations

import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import AppConfig
from .graph import run_agent

load_dotenv()

app = FastAPI(title="FutureBase Chat")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    todos: list[str]
    trace_id: str | None = None


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(HTML_PAGE)


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    session_id = payload.session_id or str(uuid.uuid4())
    config = AppConfig.from_env()
    try:
        state = run_agent(payload.message, session_id=session_id, config=config)
    except Exception as exc:  # pragma: no cover - surface agent errors to UI
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc
    return ChatResponse(
        session_id=session_id,
        reply=str(state.get("result", "")),
        todos=list(state.get("todos", [])),
        trace_id=state.get("trace_id") or None,
    )


HTML_PAGE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>FutureBase Multi-turn Chat</title>
    <style>
      @import url("https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Space+Grotesk:wght@400;500;600;700&display=swap");

      :root {
        color-scheme: light;
        --ink: #1c1a14;
        --muted: #5b5b52;
        --paper: #f4efe6;
        --paper-strong: #f0e7d9;
        --accent: #e2725b;
        --accent-strong: #ba4b34;
        --accent-cool: #2f4858;
        --bubble-user: #f1d3a7;
        --bubble-agent: #e8f0ef;
        --stroke: #d7cdbd;
        --shadow: 0 24px 45px rgba(20, 18, 12, 0.12);
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        font-family: "Space Grotesk", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at 12% 18%, rgba(226, 114, 91, 0.22), transparent 52%),
          radial-gradient(circle at 86% 12%, rgba(47, 72, 88, 0.18), transparent 48%),
          linear-gradient(145deg, #f8f2e6, #dde4d0);
        min-height: 100vh;
      }

      body::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
          linear-gradient(transparent 95%, rgba(0, 0, 0, 0.05) 100%),
          linear-gradient(90deg, transparent 95%, rgba(0, 0, 0, 0.04) 100%);
        background-size: 22px 22px;
        opacity: 0.25;
      }

      .shell {
        max-width: 980px;
        margin: 0 auto;
        padding: 32px 20px 36px;
        display: flex;
        flex-direction: column;
        gap: 24px;
      }

      header {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
      }

      .title-block {
        display: grid;
        gap: 8px;
      }

      .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.24em;
        font-size: 11px;
        font-weight: 600;
        color: var(--accent-strong);
      }

      h1 {
        font-family: "Fraunces", serif;
        font-size: clamp(2rem, 2.5vw, 2.6rem);
        margin: 0;
      }

      .sub {
        margin: 0;
        color: var(--muted);
        max-width: 520px;
        line-height: 1.5;
      }

      .session-panel {
        display: grid;
        gap: 12px;
        padding: 14px 18px;
        border-radius: 16px;
        background: rgba(244, 239, 230, 0.85);
        border: 1px solid var(--stroke);
        box-shadow: var(--shadow);
        min-width: 220px;
      }

      .session-meta {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        font-size: 14px;
        color: var(--muted);
      }

      .session-code {
        font-weight: 600;
        color: var(--ink);
      }

      .button {
        appearance: none;
        border: none;
        background: var(--accent);
        color: #fff;
        padding: 10px 16px;
        border-radius: 999px;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
      }

      .button.secondary {
        background: var(--accent-cool);
      }

      .button:active {
        transform: translateY(1px);
      }

      main {
        display: grid;
        gap: 18px;
      }

      .chat-window {
        min-height: 46vh;
        max-height: 62vh;
        overflow-y: auto;
        padding: 22px;
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid var(--stroke);
        box-shadow: var(--shadow);
        backdrop-filter: blur(12px);
        display: flex;
        flex-direction: column;
        gap: 16px;
      }

      .empty-state {
        text-align: center;
        color: var(--muted);
        padding: 26px 14px;
        border: 1px dashed var(--stroke);
        border-radius: 16px;
        background: rgba(240, 231, 217, 0.6);
      }

      .message {
        display: flex;
        flex-direction: column;
        gap: 6px;
        animation: rise 0.3s ease;
      }

      .message.user {
        align-items: flex-end;
      }

      .message.assistant {
        align-items: flex-start;
      }

      .label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: var(--muted);
      }

      .bubble {
        padding: 14px 16px;
        border-radius: 18px;
        max-width: min(520px, 90%);
        line-height: 1.55;
        font-size: 15px;
        box-shadow: 0 10px 24px rgba(20, 18, 12, 0.08);
      }

      .message.user .bubble {
        background: var(--bubble-user);
        border-top-right-radius: 6px;
      }

      .message.assistant .bubble {
        background: var(--bubble-agent);
        border-top-left-radius: 6px;
      }

      .message.pending .bubble {
        opacity: 0.7;
        font-style: italic;
      }

      .todos {
        margin: 0;
        padding: 0 0 0 18px;
        color: var(--muted);
      }

      .composer {
        display: grid;
        gap: 12px;
        padding: 18px;
        border-radius: 18px;
        background: rgba(244, 239, 230, 0.85);
        border: 1px solid var(--stroke);
        box-shadow: var(--shadow);
      }

      textarea {
        width: 100%;
        min-height: 96px;
        border-radius: 14px;
        border: 1px solid var(--stroke);
        padding: 12px 14px;
        font-family: inherit;
        font-size: 15px;
        resize: vertical;
        background: #fffdf8;
      }

      .actions {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
      }

      .hint {
        font-size: 12px;
        color: var(--muted);
      }

      @keyframes rise {
        from {
          opacity: 0;
          transform: translateY(8px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      @media (max-width: 720px) {
        header {
          flex-direction: column;
          align-items: flex-start;
        }

        .session-panel {
          width: 100%;
        }

        .chat-window {
          max-height: none;
        }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <header>
        <div class="title-block">
          <div class="eyebrow">FutureBase</div>
          <h1>Multi-turn Commerce Chat</h1>
          <p class="sub">Ask for product options, pricing, cart changes, or checkout steps.</p>
        </div>
        <div class="session-panel">
          <div class="session-meta">
            <span>Session</span>
            <span class="session-code" id="session-id">new</span>
          </div>
          <button class="button secondary" id="new-session" type="button">New session</button>
        </div>
      </header>

      <main>
        <section class="chat-window" id="chat" aria-live="polite"></section>
        <form class="composer" id="composer">
          <textarea
            id="message"
            placeholder="Ask me to find a product under 5000, compare options, or prep a checkout."
          ></textarea>
          <div class="actions">
            <div class="hint">Shift+Enter for newline</div>
            <button class="button" id="send" type="submit">Send</button>
          </div>
        </form>
      </main>
    </div>

    <script>
      const chat = document.getElementById("chat");
      const form = document.getElementById("composer");
      const input = document.getElementById("message");
      const sendButton = document.getElementById("send");
      const sessionLabel = document.getElementById("session-id");
      const newSessionButton = document.getElementById("new-session");

      const SESSION_KEY = "futurebase.session";
      const MESSAGES_KEY = "futurebase.messages";
      let sessionId = localStorage.getItem(SESSION_KEY) || "";

      function updateSessionLabel(id) {
        if (!id) {
          sessionLabel.textContent = "new";
          return;
        }
        const short = id.length > 8 ? id.slice(-8) : id;
        sessionLabel.textContent = short;
      }

      function saveSession(id) {
        sessionId = id;
        if (id) {
          localStorage.setItem(SESSION_KEY, id);
        } else {
          localStorage.removeItem(SESSION_KEY);
        }
        updateSessionLabel(id);
      }

      function loadMessages() {
        try {
          return JSON.parse(localStorage.getItem(MESSAGES_KEY) || "[]");
        } catch (error) {
          return [];
        }
      }

      function saveMessages(messages) {
        localStorage.setItem(MESSAGES_KEY, JSON.stringify(messages));
      }

      function appendMessage(role, text, todos, options = {}) {
        const wrapper = document.createElement("div");
        wrapper.className = `message ${role}`;
        if (options.pending) {
          wrapper.classList.add("pending");
        }

        const label = document.createElement("div");
        label.className = "label";
        label.textContent = role === "user" ? "You" : "Agent";

        const bubble = document.createElement("div");
        bubble.className = "bubble";
        bubble.textContent = text;

        wrapper.appendChild(label);
        wrapper.appendChild(bubble);

        if (Array.isArray(todos) && todos.length) {
          const list = document.createElement("ul");
          list.className = "todos";
          todos.forEach((item) => {
            const li = document.createElement("li");
            li.textContent = item;
            list.appendChild(li);
          });
          wrapper.appendChild(list);
        }

        chat.appendChild(wrapper);
        chat.scrollTop = chat.scrollHeight;
        return { wrapper, bubble };
      }

      function renderStoredMessages() {
        const stored = loadMessages();
        if (!stored.length) {
          const empty = document.createElement("div");
          empty.className = "empty-state";
          empty.textContent = "Start a conversation to build a multi-turn thread.";
          chat.appendChild(empty);
          return;
        }
        stored.forEach((msg) => {
          appendMessage(msg.role, msg.text, msg.todos || []);
        });
      }

      function addStoredMessage(role, text, todos) {
        const stored = loadMessages();
        stored.push({ role, text, todos: todos || [] });
        saveMessages(stored);
      }

      function clearMessages() {
        localStorage.removeItem(MESSAGES_KEY);
        chat.innerHTML = "";
        renderStoredMessages();
      }

      async function sendMessage(message) {
        if (!message.trim()) {
          return;
        }
        if (chat.querySelector(".empty-state")) {
          chat.innerHTML = "";
        }

        appendMessage("user", message);
        addStoredMessage("user", message, []);
        input.value = "";
        sendButton.disabled = true;
        sendButton.textContent = "Working...";

        const pending = appendMessage("assistant", "Thinking...", [], { pending: true });

        try {
          const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, session_id: sessionId || null })
          });
          if (!response.ok) {
            const errorPayload = await response.json().catch(() => ({}));
            throw new Error(errorPayload.detail || "Server error");
          }
          const data = await response.json();
          saveSession(data.session_id);
          pending.bubble.textContent = data.reply || "No response returned.";
          pending.wrapper.classList.remove("pending");
          if (Array.isArray(data.todos) && data.todos.length) {
            const list = document.createElement("ul");
            list.className = "todos";
            data.todos.forEach((item) => {
              const li = document.createElement("li");
              li.textContent = item;
              list.appendChild(li);
            });
            pending.wrapper.appendChild(list);
          }
          addStoredMessage("assistant", data.reply || "", data.todos || []);
        } catch (error) {
          pending.bubble.textContent = `Error: ${error.message}`;
          pending.wrapper.classList.remove("pending");
        } finally {
          sendButton.disabled = false;
          sendButton.textContent = "Send";
        }
      }

      form.addEventListener("submit", (event) => {
        event.preventDefault();
        sendMessage(input.value);
      });

      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          sendMessage(input.value);
        }
      });

      newSessionButton.addEventListener("click", () => {
        saveSession("");
        clearMessages();
      });

      updateSessionLabel(sessionId);
      renderStoredMessages();
    </script>
  </body>
</html>
"""
