"""HTML frontend template for the chatbot."""

from __future__ import annotations


def render_index_html() -> str:
    """Return the static HTML for the chat interface."""

    return """
    <!doctype html>
    <html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
        <title>LangChain Chatbot</title>
        <style>
            :root { --bg: #0f172a; --panel: #111827; --accent: #22d3ee; --text: #e5e7eb; --muted: #94a3b8; }
            * { box-sizing: border-box; }
            body { margin: 0; background: var(--bg); color: var(--text); font-family: system-ui, -apple-system, sans-serif; }
            header { padding: 16px 20px; border-bottom: 1px solid #1f2937; display: flex; align-items: center; gap: 12px; }
            header h1 { margin: 0; font-size: 20px; letter-spacing: 0.4px; }
            main { display: flex; justify-content: center; padding: 24px; }
            .chat { width: min(820px, 100%); background: var(--panel); border: 1px solid #1f2937; border-radius: 14px; overflow: hidden; display: grid; grid-template-rows: auto 1fr auto; min-height: 72vh; }
            .messages { padding: 16px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }
            .settings { padding: 12px 16px; border-bottom: 1px solid #1f2937; display: flex; align-items: center; gap: 10px; background: #0b1220; }
            .settings label { color: var(--muted); font-size: 14px; }
            .settings input { flex: 1; padding: 10px; border-radius: 10px; border: 1px solid #1f2937; background: #0f172a; color: var(--text); }
            .bubble { padding: 12px 14px; border-radius: 12px; line-height: 1.5; max-width: 92%; white-space: pre-wrap; }
            .user { align-self: flex-end; background: #1f2937; border: 1px solid #22d3ee44; }
            .bot { align-self: flex-start; background: #0b2530; border: 1px solid #0ea5e9; }
            form { display: flex; gap: 10px; padding: 12px; border-top: 1px solid #1f2937; background: #0b1220; }
            textarea { flex: 1; resize: none; padding: 12px; border-radius: 10px; border: 1px solid #1f2937; background: #0f172a; color: var(--text); min-height: 56px; font-size: 15px; }
            button { background: linear-gradient(135deg, #06b6d4, #38bdf8); border: none; color: #0b1220; font-weight: 700; border-radius: 10px; padding: 0 18px; cursor: pointer; min-width: 88px; }
            button:disabled { opacity: 0.55; cursor: not-allowed; }
            .meta { color: var(--muted); font-size: 13px; }
            .spinner { width: 14px; height: 14px; border-radius: 50%; border: 2px solid #38bdf8; border-top-color: transparent;display: inline-block; animation: spin 1s linear infinite; margin-right: 8px; vertical-align: middle; }
            .status { color: #fbbf24; font-size: 13px; }
            @keyframes spin { to { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <header>
            <svg width=\"28\" height=\"28\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"#22d3ee\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><circle cx=\"12\" cy=\"12\" r=\"10\"/><path d=\"M8 12h8M12 8v8\"/></svg>
            <div>
                <h1>LangChain Chatbot</h1>
                <div class=\"meta\">Ask about LangChain, Chroma, or containers.</div>
            </div>
        </header>
        <main>
            <section class=\"chat\">
                <div class=\"settings\">
                    <label for=\"persona\">Persona</label>
                    <input id=\"persona\" name=\"persona\" placeholder=\"Optional tone or role...\" />
                </div>
                <div id=\"messages\" class=\"messages\"></div>
                <form id=\"chat-form\">
                    <textarea id=\"message\" placeholder=\"Ask me anything...\" required></textarea>
                    <button type=\"submit\">Send</button>
                </form>
            </section>
        </main>
        <script>
            const form = document.getElementById('chat-form');
            const textarea = document.getElementById('message');
            const messages = document.getElementById('messages');
            const personaInput = document.getElementById('persona');

            const SESSION_KEY = 'langchain-agent-session';
            const PERSONA_KEY = 'langchain-agent-persona';

            function getSessionId() {
                let sid = localStorage.getItem(SESSION_KEY);
                if (!sid) {
                    sid = crypto.randomUUID();
                    localStorage.setItem(SESSION_KEY, sid);
                }
                return sid;
            }

            function getPersona() {
                return localStorage.getItem(PERSONA_KEY) || '';
            }

            function persistPersona(value) {
                localStorage.setItem(PERSONA_KEY, value.trim());
            }

            function addBubble(text, type = 'bot') {
                const bubble = document.createElement('div');
                bubble.className = `bubble ${type}`;
                bubble.textContent = text;
                messages.appendChild(bubble);
                messages.scrollTop = messages.scrollHeight;
                return bubble;
            }

            function createStreamBubble() {
                const bubble = document.createElement('div');
                bubble.className = 'bubble bot';
                const spinner = document.createElement('span');
                spinner.className = 'spinner';
                const textSpan = document.createElement('span');
                textSpan.className = 'stream-text';
                bubble.appendChild(spinner);
                bubble.appendChild(textSpan);
                messages.appendChild(bubble);
                messages.scrollTop = messages.scrollHeight;
                return { bubble, spinner, textSpan };
            }

            function setStatus(message) {
                const status = document.createElement('div');
                status.className = 'status';
                status.textContent = message;
                messages.appendChild(status);
                messages.scrollTop = messages.scrollHeight;
            }

            async function sendMessage(evt) {
                evt.preventDefault();
                const content = textarea.value.trim();
                if (!content) return;
                addBubble(content, 'user');
                textarea.value = '';

                const { bubble, spinner, textSpan } = createStreamBubble();
                const controller = new AbortController();
                let accumulated = '';
                let buffer = '';
                const decoder = new TextDecoder();
                const timeout = setTimeout(() => controller.abort(), 90000);

                try {
                    const res = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Accept': 'text/event-stream'
                        },
                        body: JSON.stringify({
                            message: content,
                            session_id: getSessionId(),
                            persona: getPersona()
                        }),
                        signal: controller.signal
                    });

                    if (!res.ok || !res.body) {
                        throw new Error('Network response was not OK');
                    }

                    const reader = res.body.getReader();
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        buffer += decoder.decode(value, { stream: true });
                        const events = buffer.split('\n\n');
                        buffer = events.pop();

                        for (const evt of events) {
                            const line = evt.trim();
                            if (!line.startsWith('data: ')) continue;
                            const payload = line.slice(6);
                            if (payload === '[DONE]') {
                                spinner.remove();
                                return;
                            }
                            accumulated += payload;
                            textSpan.textContent = accumulated;
                            messages.scrollTop = messages.scrollHeight;
                        }
                    }

                    if (buffer) {
                        const payload = buffer.replace(/^data: /, '').trim();
                        if (payload && payload !== '[DONE]') {
                            accumulated += payload;
                            textSpan.textContent = accumulated;
                        }
                    }
                } catch (err) {
                    if (spinner.isConnected) spinner.remove();
                    bubble.classList.add('bot');
                    textSpan.textContent = accumulated || 'Connection dropped. Please try again.';
                    setStatus('Streaming interrupted: ' + (err?.message || 'unknown error'));
                } finally {
                    clearTimeout(timeout);
                    if (spinner.isConnected) spinner.remove();
                    messages.scrollTop = messages.scrollHeight;
                }
            }

            form.addEventListener('submit', sendMessage);
            personaInput.value = getPersona();
            personaInput.addEventListener('input', (evt) => {
                persistPersona(evt.target.value);
            });
            addBubble('Hi! I am a tiny LangChain agent. Ask me something about the demo context.');
        </script>
    </body>
    </html>
    """
