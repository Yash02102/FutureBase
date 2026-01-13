from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .commerce_agent import run


app = FastAPI(title="Commerce Agent Chat UI")


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    address: Optional[str] = None
    payment_method: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _HTML


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    initial_entities = {}
    if request.user_id:
        initial_entities["user_id"] = request.user_id
    if request.address:
        initial_entities["address"] = request.address
    if request.payment_method:
        initial_entities["payment_method"] = request.payment_method
    state = run(
        request.message,
        session_id=request.session_id,
        initial_entities=initial_entities or None,
    )
    return {
        "reply": state.get("result", ""),
        "session_id": state.get("session_id"),
        "intent": state.get("intent"),
    }


_HTML = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Commerce Agent</title>
    <style>
      :root {
        --bg: #f5f2ec;
        --panel: #ffffff;
        --ink: #1d1d1f;
        --accent: #0f766e;
        --muted: #6b7280;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Space Grotesk", "Segoe UI", sans-serif;
        background: radial-gradient(circle at top, #fef3c7, var(--bg));
        color: var(--ink);
      }
      .shell {
        max-width: 980px;
        margin: 0 auto;
        padding: 32px 20px 64px;
      }
      header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 24px;
      }
      header h1 {
        font-size: 28px;
        margin: 0;
      }
      header p {
        margin: 6px 0 0;
        color: var(--muted);
      }
      .panel {
        background: var(--panel);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 20px 60px rgba(15, 23, 42, 0.08);
      }
      .messages {
        display: grid;
        gap: 12px;
        max-height: 420px;
        overflow-y: auto;
        padding-right: 6px;
      }
      .bubble {
        padding: 12px 16px;
        border-radius: 14px;
        line-height: 1.5;
        white-space: pre-wrap;
      }
      .user {
        background: #f0fdfa;
        justify-self: end;
        border: 1px solid #99f6e4;
      }
      .bot {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
      }
      form {
        display: grid;
        gap: 12px;
        margin-top: 16px;
      }
      .row {
        display: grid;
        gap: 10px;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      }
      input, textarea {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        padding: 10px 12px;
        font: inherit;
      }
      button {
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 999px;
        padding: 10px 18px;
        font-weight: 600;
        cursor: pointer;
      }
      button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
      }
      .hint {
        color: var(--muted);
        font-size: 13px;
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <header>
        <div>
          <h1>Commerce Agent</h1>
          <p>Ask to buy, compare, track, return, or reorder.</p>
        </div>
      </header>
      <div class="panel">
        <div class="messages" id="messages"></div>
        <form id="chat-form">
          <textarea id="message" rows="3" placeholder="e.g. Buy me a wireless headset under 5000"></textarea>
          <div class="row">
            <input id="user_id" placeholder="User ID (optional)" />
            <input id="payment_method" placeholder="Payment method (optional)" />
            <input id="address" placeholder="Address (optional)" />
          </div>
          <button id="send" type="submit">Send</button>
          <span class="hint">Approvals run via console when APPROVAL_MODE=manual.</span>
        </form>
      </div>
    </div>
    <script>
      const messages = document.getElementById("messages");
      const form = document.getElementById("chat-form");
      const messageInput = document.getElementById("message");
      const userIdInput = document.getElementById("user_id");
      const paymentInput = document.getElementById("payment_method");
      const addressInput = document.getElementById("address");
      const sendButton = document.getElementById("send");
      let sessionId = localStorage.getItem("commerce_session_id");

      function addBubble(text, cls) {
        const div = document.createElement("div");
        div.className = `bubble ${cls}`;
        div.textContent = text;
        messages.appendChild(div);
        messages.scrollTop = messages.scrollHeight;
      }

      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const message = messageInput.value.trim();
        if (!message) return;
        addBubble(message, "user");
        messageInput.value = "";
        sendButton.disabled = true;
        try {
          const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              message,
              session_id: sessionId,
              user_id: userIdInput.value.trim() || null,
              payment_method: paymentInput.value.trim() || null,
              address: addressInput.value.trim() || null,
            }),
          });
          const data = await response.json();
          sessionId = data.session_id;
          localStorage.setItem("commerce_session_id", sessionId);
          addBubble(data.reply || "No response.", "bot");
        } catch (err) {
          addBubble("Request failed. Check the server logs.", "bot");
        } finally {
          sendButton.disabled = false;
        }
      });
    </script>
  </body>
</html>
"""
