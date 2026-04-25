"""PocketFlow Multi-Model Debate with Real-Time SSE Streaming"""

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer, BaseHTTPRequestHandler
from queue import Empty, Queue
from urllib.parse import urlparse

import gradio as gr

from flow import create_flow

# ---- Streaming infrastructure ----

_active_stream_queue: Queue | None = None
_active_lock = threading.Lock()

chatflow_thread_pool = ThreadPoolExecutor(
    max_workers=5, thread_name_prefix="chatflow_worker"
)


class SSEHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if urlparse(self.path).path != "/stream":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        while True:
            with _active_lock:
                q = _active_stream_queue
            if q is None:
                time.sleep(0.1)
                continue
            try:
                token = q.get(timeout=60)
                if token is None:
                    self.wfile.write(b"event: done\ndata: \n\n")
                    self.wfile.flush()
                    break
                data = json.dumps({"token": token})
                self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()
            except Exception:
                break

    def log_message(self, format, *args):
        pass


def start_sse_server(port=7861):
    server = HTTPServer(("127.0.0.1", port), SSEHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ---- Gradio callbacks ----

def add_user_message(message: str, history: list):
    history.append({"role": "user", "content": message})
    return "", history


def run_debate(history: list, uuid_state: uuid.UUID):
    global _active_stream_queue
    message = history[-1]["content"]
    conversation_id = str(uuid_state)

    chat_queue = Queue()
    flow_queue = Queue()
    stream_queue = Queue()

    with _active_lock:
        _active_stream_queue = stream_queue

    shared = {
        "conversation_id": conversation_id,
        "query": message,
        "history": history[:-1],
        "queue": chat_queue,
        "flow_queue": flow_queue,
        "stream_queue": stream_queue,
    }

    chat_flow = create_flow()
    chatflow_thread_pool.submit(chat_flow.run, shared)

    history.append({"role": "assistant", "content": "... *Debating* ..."})
    yield history

    accumulated = ""
    flow_done = False

    while not flow_done:
        changed = False
        try:
            while True:
                token = stream_queue.get_nowait()
                if token is not None:
                    accumulated += token
                    changed = True
                stream_queue.task_done()
        except Empty:
            pass
        try:
            while True:
                thought = flow_queue.get_nowait()
                if thought is None:
                    flow_done = True
                    break
                changed = True
                flow_queue.task_done()
        except Empty:
            pass
        if changed:
            content = f"... *Debating* ...\n\n{accumulated}" if accumulated else "... *Debating* ..."
            history[-1] = {"role": "assistant", "content": content}
            yield history
        else:
            time.sleep(0.05)

    try:
        while True:
            token = stream_queue.get_nowait()
            if token is None:
                break
            accumulated += token
            stream_queue.task_done()
    except Empty:
        pass

    stream_queue.put(None)
    with _active_lock:
        if _active_stream_queue is stream_queue:
            _active_stream_queue = None

    while True:
        final = chat_queue.get()
        if final is None:
            break
        history.append({"role": "assistant", "content": f"### Final Answer\n\n{final}"})
        yield history
        chat_queue.task_done()


# ---- Launch ----

start_sse_server(7861)

# JavaScript injected via head: waits for DOM elements, then connects SSE
SSE_JS = """
<script>
(function() {
  var source = null;
  var out = null;
  var status = null;

  function connect() {
    if (!out || !status) return;
    if (source) return;
    source = new EventSource('http://localhost:7861/stream');
    out.textContent = '';
    status.textContent = 'Connecting...';
    status.style.color = '#ff9800';

    source.onmessage = function(e) {
      var d = JSON.parse(e.data);
      out.textContent += d.token;
      status.textContent = '[LIVE] Streaming';
      status.style.color = '#4caf50';
    };
    source.addEventListener('done', function() {
      status.textContent = '[DONE] Complete';
      status.style.color = '#4caf50';
      source.close();
      source = null;
    });
    source.onerror = function() {
      if (source) { source.close(); source = null; }
      status.textContent = 'Reconnecting...';
      status.style.color = '#ff9800';
      setTimeout(connect, 2000);
    };
  }

  // Wait for DOM elements to appear
  var obs = new MutationObserver(function() {
    out = document.getElementById('stream-output');
    status = document.getElementById('stream-status');
    if (out && status) {
      obs.disconnect();
      connect();
    }
  });
  obs.observe(document.body, { childList: true, subtree: true });
})();
</script>
"""

with gr.Blocks(
    fill_height=True,
    title="PocketFlow Multi-Model Debate",
    head=SSE_JS
) as demo:
    uuid_state = gr.State(uuid.uuid4())
    gr.Markdown("# PocketFlow Multi-Model Debate (Streaming)")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(scale=1, render_markdown=True, height=500, layout="panel")
            with gr.Row():
                msg = gr.Textbox(scale=9, placeholder="Ask a question...", container=False, lines=1)
                submit_btn = gr.Button("Send", scale=1, variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Live Stream")
            gr.HTML("""
            <div style="border:1px solid #333; border-radius:8px; padding:12px;
                 min-height:400px; max-height:500px; overflow-y:auto;
                 background:#0d1117; font-family:monospace; font-size:14px;
                 color:#c9d1d9; line-height:1.6;">
              <div id="stream-output" style="white-space:pre-wrap; word-break:break-word;">
                <em style="color:#666;">Send a query to begin...</em>
              </div>
              <div id="stream-status" style="margin-top:8px; font-size:12px; color:#888;"></div>
            </div>
            """)

    clear_btn = gr.Button("Clear Conversation")

    msg.submit(add_user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        run_debate, [chatbot, uuid_state], [chatbot]
    )
    submit_btn.click(add_user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        run_debate, [chatbot, uuid_state], [chatbot]
    )
    clear_btn.click(lambda: ([], uuid.uuid4()), None, [chatbot, uuid_state])

demo.queue(default_concurrency_limit=5)
demo.launch(theme="ocean")
