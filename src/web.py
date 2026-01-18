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
        color-scheme: dark;
        --ink: #f8fafc;
        --muted: #94a3b8;
        --paper: #0f1115;
        --paper-strong: #1f232b;
        --accent: #60a5fa;
        --accent-strong: #2563eb;
        --accent-cool: #22d3ee;
        --bubble-user: #2f343e;
        --bubble-agent: #1f232b;
        --stroke: rgba(148, 163, 184, 0.15);
        --shadow: 0 24px 60px rgba(2, 6, 23, 0.6);
        --shadow-strong: 0 30px 70px rgba(2, 6, 23, 0.7);
        --glow: 0 0 0 1px rgba(96, 165, 250, 0.2);
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        font-family: "Space Grotesk", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at 20% 20%, rgba(59, 130, 246, 0.15), transparent 45%),
          radial-gradient(circle at 80% 10%, rgba(34, 211, 238, 0.1), transparent 45%),
          linear-gradient(160deg, #0b0d12 0%, #151922 60%, #1b1f29 100%);
        min-height: 100vh;
      }

      body::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
          linear-gradient(transparent 96%, rgba(255, 255, 255, 0.05) 100%),
          linear-gradient(90deg, transparent 96%, rgba(255, 255, 255, 0.04) 100%);
        background-size: 28px 28px;
        opacity: 0.15;
      }

      .ambient-orbs {
        position: fixed;
        inset: 0;
        pointer-events: none;
        overflow: hidden;
        z-index: 0;
      }

      .ambient-orbs span {
        position: absolute;
        width: 320px;
        height: 320px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(96, 165, 250, 0.25), transparent 70%);
        filter: blur(12px);
        animation: float 18s ease-in-out infinite;
      }

      .ambient-orbs span:nth-child(1) {
        top: -80px;
        left: -60px;
      }

      .ambient-orbs span:nth-child(2) {
        width: 420px;
        height: 420px;
        bottom: -120px;
        right: -80px;
        background: radial-gradient(circle, rgba(34, 211, 238, 0.18), transparent 70%);
        animation-delay: -8s;
      }

      .ambient-orbs span:nth-child(3) {
        width: 220px;
        height: 220px;
        top: 35%;
        right: 12%;
        background: radial-gradient(circle, rgba(79, 70, 229, 0.18), transparent 70%);
        animation-delay: -12s;
      }

      .shell {
        max-width: 1040px;
        margin: 0 auto;
        padding: 36px 22px 40px;
        display: flex;
        flex-direction: column;
        gap: 24px;
        position: relative;
        z-index: 1;
        min-height: 100vh;
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
        font-size: clamp(2rem, 2.6vw, 2.8rem);
        margin: 0;
      }

      .subtle {
        color: var(--muted);
        font-size: 14px;
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
        border-radius: 18px;
        background: rgba(31, 35, 43, 0.8);
        border: 1px solid var(--stroke);
        box-shadow: var(--shadow);
        min-width: 220px;
        backdrop-filter: blur(12px);
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
        background: linear-gradient(135deg, var(--accent-strong), var(--accent-cool));
        color: #fff;
        padding: 10px 18px;
        border-radius: 999px;
        font-weight: 600;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        box-shadow: 0 12px 24px rgba(79, 70, 229, 0.25);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
      }

      .button.secondary {
        background: linear-gradient(135deg, #111827, #1f2937);
      }

      .button:active {
        transform: translateY(1px);
      }

      .button.is-sending::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(120deg, transparent, rgba(255, 255, 255, 0.4), transparent);
        transform: translateX(-100%);
        animation: shimmer 1s ease-in-out infinite;
      }

      main {
        display: grid;
        gap: 18px;
        flex: 1;
      }

      .chat-window {
        min-height: 60vh;
        border-radius: 0;
        background: transparent;
        border: none;
        box-shadow: none;
        backdrop-filter: none;
        position: relative;
        display: flex;
        flex-direction: column;
        overflow: visible;
      }

      .chat-messages {
        display: flex;
        flex-direction: column;
        gap: 18px;
        padding: 26px clamp(16px, 6vw, 96px) 32px;
        overflow-y: visible;
        flex: 1;
        align-items: center;
      }

      .chat-overlay {
        position: absolute;
        inset: 0;
        display: grid;
        place-items: center;
        background: rgba(15, 23, 42, 0.6);
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.3s ease;
        z-index: 2;
      }

      body.is-sending .chat-overlay {
        opacity: 1;
      }

      .overlay-card {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 16px 20px;
        border-radius: 18px;
        background: rgba(17, 24, 39, 0.95);
        border: 1px solid rgba(148, 163, 184, 0.3);
        box-shadow: var(--shadow);
        backdrop-filter: blur(12px);
      }

      .overlay-orbit {
        width: 46px;
        height: 46px;
        border-radius: 50%;
        border: 2px solid rgba(79, 70, 229, 0.2);
        position: relative;
      }

      .overlay-orbit::before {
        content: "";
        position: absolute;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: var(--accent);
        top: -5px;
        left: 18px;
        animation: spin 1.4s linear infinite;
      }

      .overlay-text {
        display: grid;
        gap: 4px;
      }

      .overlay-text strong {
        font-size: 15px;
      }

      .overlay-text span {
        font-size: 12px;
        color: var(--muted);
      }

      .overlay-dots {
        display: flex;
        gap: 6px;
      }

      .overlay-dots span {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--accent-cool);
        animation: bounce 1s ease-in-out infinite;
      }

      .overlay-dots span:nth-child(2) {
        animation-delay: 0.2s;
      }

      .overlay-dots span:nth-child(3) {
        animation-delay: 0.4s;
      }

      .empty-state {
        text-align: center;
        color: var(--muted);
        padding: 28px 16px;
        border: 1px dashed rgba(148, 163, 184, 0.4);
        border-radius: 18px;
        background: rgba(30, 41, 59, 0.35);
      }

      .message {
        display: flex;
        flex-direction: column;
        gap: 6px;
        animation: rise 0.3s ease;
        width: 100%;
      }

      .message.user {
        align-items: flex-end;
      }

      .message.assistant {
        align-items: center;
      }

      .label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.22em;
        color: rgba(148, 163, 184, 0.6);
      }

      .bubble {
        padding: 14px 18px;
        border-radius: 18px;
        max-width: min(540px, 92%);
        line-height: 1.6;
        font-size: 15px;
        box-shadow: 0 12px 24px rgba(2, 6, 23, 0.25);
        border: 1px solid rgba(148, 163, 184, 0.08);
      }

      .message.assistant .bubble {
        max-width: 100%;
        width: 100%;
      }

      .bubble p {
        margin: 0 0 10px;
      }

      .bubble p:last-child {
        margin-bottom: 0;
      }

      .bubble ul,
      .bubble ol {
        margin: 8px 0 8px 18px;
        padding: 0;
      }

      .bubble code {
        font-family: "Space Grotesk", sans-serif;
        background: rgba(148, 163, 184, 0.12);
        padding: 2px 6px;
        border-radius: 6px;
      }

      .bubble pre {
        background: rgba(2, 6, 23, 0.5);
        padding: 10px 12px;
        border-radius: 12px;
        overflow-x: auto;
      }

      .bubble pre code {
        background: transparent;
        padding: 0;
      }

      .bubble table {
        width: 100%;
        border-collapse: collapse;
        margin: 12px 0;
        font-size: 14px;
      }

      .bubble th,
      .bubble td {
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 8px 10px;
        text-align: left;
      }

      .bubble thead {
        background: rgba(15, 23, 42, 0.6);
      }

      .message.user .bubble {
        background: var(--bubble-user);
        border-top-right-radius: 8px;
      }

      .message.assistant .bubble {
        background: var(--bubble-agent);
        border-top-left-radius: 8px;
      }

      .message.pending .bubble {
        opacity: 0.9;
        font-style: italic;
      }

      .thinking {
        display: inline-flex;
        align-items: center;
        gap: 8px;
      }

      .thinking-dots {
        display: inline-flex;
        gap: 4px;
      }

      .thinking-dots span {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--accent);
        animation: bounce 1s ease-in-out infinite;
      }

      .thinking-dots span:nth-child(2) {
        animation-delay: 0.2s;
      }

      .thinking-dots span:nth-child(3) {
        animation-delay: 0.4s;
      }

      .todos {
        margin: 0;
        padding: 0 0 0 18px;
        color: var(--muted);
      }

      .composer {
        display: grid;
        gap: 12px;
        padding: 16px 20px 18px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        background: rgba(17, 24, 39, 0.9);
        backdrop-filter: blur(10px);
        margin: 0 0 32px;
        width: 100%;
        border-radius: 16px;
        box-shadow: var(--shadow);
        position: sticky;
        bottom: 24px;
      }

      textarea {
        width: 100%;
        min-height: 78px;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        padding: 12px 14px;
        font-family: inherit;
        font-size: 15px;
        resize: vertical;
        background: rgba(15, 23, 42, 0.7);
        color: var(--ink);
        box-shadow: inset 0 1px 2px rgba(2, 6, 23, 0.4);
        transition: border 0.2s ease, box-shadow 0.2s ease;
      }

      textarea:focus {
        outline: none;
        border-color: rgba(96, 165, 250, 0.6);
        box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.15);
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

      @keyframes float {
        0%,
        100% {
          transform: translateY(0);
        }
        50% {
          transform: translateY(18px);
        }
      }

      @keyframes spin {
        0% {
          transform: rotate(0deg) translateY(-1px);
        }
        100% {
          transform: rotate(360deg) translateY(-1px);
        }
      }

      @keyframes bounce {
        0%,
        100% {
          transform: translateY(0);
          opacity: 0.5;
        }
        50% {
          transform: translateY(-4px);
          opacity: 1;
        }
      }

      @keyframes shimmer {
        0% {
          transform: translateX(-120%);
        }
        100% {
          transform: translateX(120%);
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
    <div class="ambient-orbs" aria-hidden="true">
      <span></span>
      <span></span>
      <span></span>
    </div>
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
        <section class="chat-window" id="chat" aria-live="polite">
          <div class="chat-overlay" id="chat-overlay" aria-hidden="true">
            <div class="overlay-card">
              <div class="overlay-orbit" aria-hidden="true"></div>
              <div class="overlay-text">
                <strong>Agent is thinking</strong>
                <span>Reviewing the catalog and drafting a response.</span>
              </div>
              <div class="overlay-dots" aria-hidden="true">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
          <div class="chat-messages" id="chat-messages"></div>
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
        </section>
      </main>
    </div>

    <script>
      const chat = document.getElementById("chat");
      const chatMessages = document.getElementById("chat-messages");
      const form = document.getElementById("composer");
      const input = document.getElementById("message");
      const sendButton = document.getElementById("send");
      const sessionLabel = document.getElementById("session-id");
      const newSessionButton = document.getElementById("new-session");
      const overlay = document.getElementById("chat-overlay");

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

      function setSendingState(isSending) {
        document.body.classList.toggle("is-sending", isSending);
        sendButton.classList.toggle("is-sending", isSending);
        input.disabled = isSending;
        if (overlay) {
          overlay.setAttribute("aria-hidden", String(!isSending));
        }
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

      function escapeHTML(value) {
        return value
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#39;");
      }

      function formatInline(text) {
        return text
          .replace(/`([^`]+)`/g, "<code>$1</code>")
          .replace(/\\*\\*([^*]+)\\*\\*/g, "<strong>$1</strong>")
          .replace(/\\*([^*]+)\\*/g, "<em>$1</em>");
      }

      function splitTableRow(line) {
        const trimmed = line.trim().replace(/^\\|/, "").replace(/\\|$/, "");
        return trimmed.split("|").map((cell) => cell.trim());
      }

      function isTableSeparator(line) {
        const cells = splitTableRow(line);
        if (cells.length < 2) {
          return false;
        }
        return cells.every((cell) => /^:?-{3,}:?$/.test(cell));
      }

      function isTableRow(line) {
        return line.includes("|");
      }

      function buildTable(lines, startIndex) {
        const headerCells = splitTableRow(lines[startIndex]);
        let index = startIndex + 2;
        let html = "<table><thead><tr>";
        headerCells.forEach((cell) => {
          html += `<th>${formatInline(cell)}</th>`;
        });
        html += "</tr></thead><tbody>";

        while (index < lines.length && lines[index].trim()) {
          if (!isTableRow(lines[index]) || isTableSeparator(lines[index])) {
            break;
          }
          const rowCells = splitTableRow(lines[index]);
          html += "<tr>";
          rowCells.forEach((cell) => {
            html += `<td>${formatInline(cell)}</td>`;
          });
          html += "</tr>";
          index += 1;
        }

        html += "</tbody></table>";
        return { html, nextIndex: index };
      }

      function formatMessage(text) {
        const escaped = escapeHTML(text);
        const lines = escaped.split(/\\n/);
        let html = "";
        let listType = null;

        for (let i = 0; i < lines.length; i += 1) {
          const line = lines[i];
          const trimmed = line.trim();
          const bulletMatch = trimmed.match(/^[-*]\\s+(.*)/);
          const orderedMatch = trimmed.match(/^\\d+\\.\\s+(.*)/);
          const hasTable =
            isTableRow(trimmed) && lines[i + 1] && isTableSeparator(lines[i + 1].trim());

          if (hasTable) {
            if (listType) {
              html += `</${listType}>`;
              listType = null;
            }
            const table = buildTable(lines, i);
            html += table.html;
            i = table.nextIndex - 1;
            continue;
          }

          if (bulletMatch) {
            if (listType !== "ul") {
              if (listType) {
                html += `</${listType}>`;
              }
              listType = "ul";
              html += "<ul>";
            }
            html += `<li>${formatInline(bulletMatch[1])}</li>`;
            return;
          }

          if (orderedMatch) {
            if (listType !== "ol") {
              if (listType) {
                html += `</${listType}>`;
              }
              listType = "ol";
              html += "<ol>";
            }
            html += `<li>${formatInline(orderedMatch[1])}</li>`;
            return;
          }

          if (listType) {
            html += `</${listType}>`;
            listType = null;
          }

          if (!trimmed) {
            continue;
          }

          html += `<p>${formatInline(trimmed)}</p>`;
        }

        if (listType) {
          html += `</${listType}>`;
        }

        return html || "<p>…</p>";
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
        if (options.pending) {
          bubble.innerHTML = `
            <span class="thinking">
              <span>Thinking</span>
              <span class="thinking-dots">
                <span></span>
                <span></span>
                <span></span>
              </span>
            </span>
          `;
        } else {
          bubble.innerHTML = role === "assistant" ? formatMessage(text) : escapeHTML(text);
        }

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

        chatMessages.appendChild(wrapper);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return { wrapper, bubble };
      }

      function renderStoredMessages() {
        const stored = loadMessages();
        if (!stored.length) {
          const empty = document.createElement("div");
          empty.className = "empty-state";
          empty.textContent = "Start a conversation to build a multi-turn thread.";
          chatMessages.appendChild(empty);
          return;
        }
        stored.forEach((msg) => {
          appendMessage(msg.role, msg.text, msg.todos || []);
        });
        chatMessages.scrollTop = chatMessages.scrollHeight;
      }

      function addStoredMessage(role, text, todos) {
        const stored = loadMessages();
        stored.push({ role, text, todos: todos || [] });
        saveMessages(stored);
      }

      function clearMessages() {
        localStorage.removeItem(MESSAGES_KEY);
        chatMessages.innerHTML = "";
        renderStoredMessages();
      }

      async function sendMessage(message) {
        if (!message.trim()) {
          return;
        }
        if (chatMessages.querySelector(".empty-state")) {
          chatMessages.innerHTML = "";
        }

        appendMessage("user", message);
        addStoredMessage("user", message, []);
        input.value = "";
        sendButton.disabled = true;
        sendButton.textContent = "Sending...";
        setSendingState(true);

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
          pending.bubble.innerHTML = formatMessage(data.reply || "No response returned.");
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
          setSendingState(false);
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
