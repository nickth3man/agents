"""PocketFlow Multi-Model Debate with Real-Time SSE Streaming — Editorial UI."""

import asyncio
import json
import os
import threading
import time
import traceback
import uuid
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from queue import Empty, Queue
from urllib.parse import urlparse

_FRONTEND = os.path.join(os.path.dirname(__file__), "frontend")


def _read(name: str) -> str:
    with open(os.path.join(_FRONTEND, name), encoding="utf-8") as f:
        return f.read()

from dotenv import load_dotenv
load_dotenv()

from utils.observability import (
    bind_request,
    get_tracer,
    logger as log,
)

import gradio as gr

from flow import create_flow
from utils.conversation import delete_conversation

# ---- Streaming infrastructure ----

_active_stream_queues: dict[str, Queue] = {}
_active_lock = threading.Lock()
_active_lock = threading.Lock()


# Map flow_queue thoughts → phase ids the front-end uses to route tokens
PHASE_MAP = {
    "🎯 Preparing debate context...": "prepare",
    "✅ Model A (Direct & Practical) has responded": "after_a",
    "✅ Model B (Cautious & Analytical) has responded": "after_b",
    "✅ Model C (Creative & Alternative) has responded": "after_c",
    "🔄 Debate round completed - models critiqued each other": "after_round",
    "⚖️ Judge has synthesized the final answer": "after_judge",
}

# When phase X completes, the NEXT phase that will start streaming tokens
NEXT_PHASE = {
    "prepare": "model_a",
    "after_a": "model_b",
    "after_b": "model_c",
    "after_c": "round",
    "after_round": "judge",
    "after_judge": "done",
}

PHASE_SENTINEL_PREFIX = "\x00PHASE:"
PHASE_SENTINEL_SUFFIX = "\x00"


class SSEHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        tracer = get_tracer()
        if urlparse(self.path).path != "/stream":
            self.send_error(404)
            return
        client_addr = self.client_address[0] if isinstance(self.client_address, tuple) else self.client_address
        with tracer.start_as_current_span("sse_connection") as span:
            span.set_attribute("client_address", client_addr)
            log.info("sse_connect", client=client_addr)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            token_count = 0
            while True:
                with _active_lock:
                    q = _active_stream_queues.get(client_addr)
                if q is None:
                    time.sleep(0.1)
                    continue
                try:
                    token = q.get(timeout=60)
                    if token is None:
                        log.info("sse_done", sent_tokens=token_count)
                        self.wfile.write(b"event: done\ndata: \n\n")
                        self.wfile.flush()
                        break
                    token_count += 1
                    # Phase sentinels: "\x00PHASE:model_b\x00"
                    if isinstance(token, str) and token.startswith(PHASE_SENTINEL_PREFIX):
                        phase = token[len(PHASE_SENTINEL_PREFIX):].rstrip(PHASE_SENTINEL_SUFFIX)
                        payload = json.dumps({"phase": phase})
                        self.wfile.write(f"event: phase\ndata: {payload}\n\n".encode())
                        self.wfile.flush()
                        continue
                    data = json.dumps({"token": token})
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                except Empty:
                    continue
                except Exception:
                    break

    def log_message(self, format, *args):
        pass


def start_sse_server(port=7861):
    server = HTTPServer(("127.0.0.1", port), SSEHandler)
    log.info("sse_server_start port=%d", port)
    threading.Thread(target=server.serve_forever, daemon=False, name="sse-server").start()
    return server


# ---- Gradio callbacks ----
MAX_INPUT_CHARS = 2000


def add_user_message(message: str, history: list):
    if not message or not message.strip():
        return "", history
    trimmed = message.strip()[:MAX_INPUT_CHARS]
    history.append({"role": "user", "content": trimmed})
    return "", history


async def _flow_runner(chat_flow, shared, flow_queue):
    """Run the async flow; capture and report failures."""
    tracer = get_tracer()
    conversation_id = shared.get("conversation_id")
    try:
        with tracer.start_as_current_span("flow_runner") as span:
            span.set_attribute("conversation_id", conversation_id)
            bind_request(conversation_id)
            log.info("flow_run_start")
            await chat_flow.run_async(shared)
            log.info("flow_run_done")
    except Exception as exc:
        tb = traceback.format_exc()
        log.error("flow_run_failed", exc_info=exc, traceback=tb)
        try:
            flow_queue.put(f"💥 Flow error: {type(exc).__name__}: {exc}")
        finally:
            flow_queue.put(None)
            chat_q = shared.get("queue")
            if chat_q is not None:
                chat_q.put(f"**Error:** {type(exc).__name__}: {exc}\n\n```\n{tb}\n```")
                chat_q.put(None)
            sq = shared.get("stream_queue")
            if sq is not None:
                sq.put(None)


async def run_debate(history: list, request: gr.Request):
    tracer = get_tracer()
    if not history or history[-1]["role"] != "user":
        yield history
        return

    message = history[-1]["content"]
    conversation_id = str(uuid.uuid4())
    bind_request(conversation_id)

    with tracer.start_as_current_span("debate_request") as span:
        span.set_attribute("conversation_id", conversation_id)
        log.info("run_debate_start", message_preview=message[:80])

        chat_queue = Queue()
        flow_queue = Queue()
        stream_queue = Queue()

        client_host = request.client.host
        with _active_lock:
            _active_stream_queues[client_host] = stream_queue

        # Reset floor then kick off with the first phase marker
        stream_queue.put(f"{PHASE_SENTINEL_PREFIX}reset{PHASE_SENTINEL_SUFFIX}")
        stream_queue.put(f"{PHASE_SENTINEL_PREFIX}model_a{PHASE_SENTINEL_SUFFIX}")

        shared = {
            "conversation_id": conversation_id,
            "query": message,
            "history": history[:-1],
            "queue": chat_queue,
            "flow_queue": flow_queue,
            "stream_queue": stream_queue,
        }

        chat_flow = create_flow()
        # Run async flow in a background thread with its own event loop
        def run_async_flow():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_flow_runner(chat_flow, shared, flow_queue))
            finally:
                loop.close()
        
        threading.Thread(target=run_async_flow, daemon=False, name="flow-runner").start()

        history.append({"role": "assistant", "content": "_The floor is open. Three models will deliberate._"})
        yield history

        flow_done = False
        thoughts: list[str] = []
        loop_iter = 0

        while not flow_done:
            loop_iter += 1
            if loop_iter % 200 == 0:
                log.info(
                    "run_debate_loop",
                    iter=loop_iter,
                    thoughts=len(thoughts),
                    flow_qsize=flow_queue.qsize(),
                )
            changed = False
            # Drain flow events first → translate to phase sentinels
            try:
                while True:
                    thought = flow_queue.get_nowait()
                    if thought is None:
                        flow_done = True
                        log.info("run_debate_flow_done")
                        break
                    thoughts.append(thought)
                    log.info("run_debate_flow_event", thought=thought)
                    phase_after = PHASE_MAP.get(thought)
                    if phase_after:
                        next_phase = NEXT_PHASE.get(phase_after)
                        if next_phase and next_phase != "done":
                            stream_queue.put(
                                f"{PHASE_SENTINEL_PREFIX}{next_phase}{PHASE_SENTINEL_SUFFIX}"
                            )
                    changed = True
                    flow_queue.task_done()
            except Empty:
                pass

            if changed:
                transcript = "\n".join(f"- {t}" for t in thoughts)
                history[-1] = {
                    "role": "assistant",
                    "content": f"_Deliberating…_\n\n{transcript}",
                }
                yield history
            else:
                time.sleep(0.05)

        stream_queue.put(None)
        with _active_lock:
            _active_stream_queues.pop(client_host, None)

        log.info("run_debate_awaiting_final_answer")
        with tracer.start_as_current_span("final_answer"):
            while True:
                final = chat_queue.get()
                if final is None:
                    break
                log.info("run_debate_final_answer", answer_length=len(final) if final else 0)
                history[-1] = {"role": "assistant", "content": final}
                yield history
                chat_queue.task_done()
        log.info("run_debate_done")
    delete_conversation(conversation_id)

# ---- Launch ----

start_sse_server(7861)

ISSUE_DATE = datetime.now().strftime("%A, %B %d, %Y").upper()

_FONT_LINKS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1'
    '&family=Geist:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">\n'
)
SSE_JS = _FONT_LINKS + "<style>\n" + _read("styles.css") + "</style>\n\n<script>\n" + _read("sse.js") + "</script>\n"
MASTHEAD_HTML = _read("masthead.html").replace("{ISSUE_DATE}", ISSUE_DATE)
FLOOR_HTML = _read("floor.html")
CHAT_HEADER_HTML = _read("chat_header.html")

with gr.Blocks(
    fill_height=True,
    title="The Debate · PocketFlow",
) as demo:

    gr.HTML(MASTHEAD_HTML)

    with gr.Row(elem_id="main-row", equal_height=False):
        with gr.Column(scale=3, elem_id="chat-col"):
            gr.HTML(CHAT_HEADER_HTML)
            chatbot = gr.Chatbot(
                render_markdown=True,
                height=520,
                show_label=False,
                avatar_images=None,
                elem_id="chatbot",
                value=[{
                    "role": "assistant",
                    "content": "_The bench is empty. Pose a question and three models will take the floor._",
                }],
            )
            with gr.Row(elem_id="input-row"):
                msg = gr.Textbox(
                    scale=8,
                    placeholder="Pose your question to the court…",
                    container=False,
                    lines=1,
                    elem_id="msg-input",
                    show_label=False,
                )
                submit_btn = gr.Button("Submit", scale=1, variant="primary", elem_id="send-btn")
            clear_btn = gr.Button("New Session — Clear Bench", elem_id="clear-btn")

        with gr.Column(scale=2):
            gr.HTML(FLOOR_HTML)

    msg.submit(add_user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        run_debate, [chatbot], [chatbot]
    )
    submit_btn.click(add_user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        run_debate, [chatbot], [chatbot]
    )
    clear_btn.click(
        lambda: [{"role": "assistant", "content": "_The bench is empty. Pose a question and three models will take the floor._"}],
        None,
        [chatbot],
    )

demo.queue(default_concurrency_limit=5)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(), head=SSE_JS)
    try:
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        pass
