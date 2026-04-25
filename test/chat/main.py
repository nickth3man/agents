"""PocketFlow Multi-Model Debate with Real-Time SSE Streaming — Editorial UI."""

import json
import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from queue import Empty, Queue
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [%(threadName)s] %(name)s: %(message)s",
)
log = logging.getLogger("debate")

import gradio as gr

from flow import create_flow

# ---- Streaming infrastructure ----

_active_stream_queue: Queue | None = None
_active_lock = threading.Lock()

chatflow_thread_pool = ThreadPoolExecutor(
    max_workers=5, thread_name_prefix="chatflow_worker"
)

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
        if urlparse(self.path).path != "/stream":
            self.send_error(404)
            return
        log.info("sse_connect from=%s", self.client_address)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        token_count = 0
        while True:
            with _active_lock:
                q = _active_stream_queue
            if q is None:
                time.sleep(0.1)
                continue
            try:
                token = q.get(timeout=60)
                if token is None:
                    log.info("sse_done sent_tokens=%d", token_count)
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

def add_user_message(message: str, history: list):
    if not message or not message.strip():
        return "", history
    history.append({"role": "user", "content": message})
    return "", history


def _flow_runner(chat_flow, shared, flow_queue):
    """Run the flow in the worker thread; capture and report failures."""
    try:
        log.info("flow_run_start conversation_id=%s", shared.get("conversation_id"))
        chat_flow.run(shared)
        log.info("flow_run_done conversation_id=%s", shared.get("conversation_id"))
    except Exception as exc:
        tb = traceback.format_exc()
        log.error("flow_run_failed: %s\n%s", exc, tb)
        # Surface the failure to the UI so we don't hang forever.
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


def run_debate(history: list, uuid_state: uuid.UUID):
    global _active_stream_queue
    if not history or history[-1]["role"] != "user":
        yield history
        return

    message = history[-1]["content"]
    conversation_id = str(uuid_state)
    log.info("run_debate_start conv=%s msg=%r", conversation_id, message[:80])

    chat_queue = Queue()
    flow_queue = Queue()
    stream_queue = Queue()

    with _active_lock:
        _active_stream_queue = stream_queue

    # Kick off with the first phase marker so the UI lights up Model A
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
    chatflow_thread_pool.submit(_flow_runner, chat_flow, shared, flow_queue)

    history.append({"role": "assistant", "content": "_The floor is open. Three models will deliberate._"})
    yield history

    flow_done = False
    thoughts: list[str] = []
    loop_iter = 0

    while not flow_done:
        loop_iter += 1
        if loop_iter % 200 == 0:
            log.info(
                "run_debate_loop conv=%s iter=%d thoughts=%d flow_qsize=%d",
                conversation_id, loop_iter, len(thoughts), flow_queue.qsize(),
            )
        changed = False
        # Drain flow events first → translate to phase sentinels
        try:
            while True:
                thought = flow_queue.get_nowait()
                if thought is None:
                    flow_done = True
                    log.info("run_debate flow_done conv=%s", conversation_id)
                    break
                thoughts.append(thought)
                log.info("run_debate flow_event conv=%s thought=%r", conversation_id, thought)
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
        if _active_stream_queue is stream_queue:
            _active_stream_queue = None

    log.info("run_debate awaiting final answer conv=%s", conversation_id)
    while True:
        final = chat_queue.get()
        if final is None:
            break
        log.info("run_debate final answer conv=%s len=%d", conversation_id, len(final) if final else 0)
        history[-1] = {"role": "assistant", "content": final}
        yield history
        chat_queue.task_done()
    log.info("run_debate_done conv=%s", conversation_id)


# ---- Launch ----

start_sse_server(7861)

ISSUE_DATE = datetime.now().strftime("%A, %B %d, %Y").upper()

# All custom styling + JS lives here. The right-side floor is fully custom HTML;
# Gradio supplies the chat + input only.
SSE_JS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --paper:        #F4EFE2;
    --paper-2:      #EBE4D2;
    --ink:          #1A1814;
    --ink-soft:     #4A463E;
    --rule:         #1A1814;
    --rule-soft:    rgba(26,24,20,0.18);
    --gold:         oklch(0.62 0.13 65);    /* Model A — rust/amber */
    --teal:         oklch(0.48 0.09 200);   /* Model B — deep teal */
    --olive:        oklch(0.55 0.10 110);   /* Model C — mustard-olive */
    --gavel:        oklch(0.45 0.16 25);    /* Judge — court red */
    --serif:        "Instrument Serif", "Times New Roman", serif;
    --sans:         "Geist", "Helvetica Neue", system-ui, sans-serif;
    --mono:         "JetBrains Mono", ui-monospace, monospace;
  }

  /* ---- Global page chrome ---- */
  html, body, gradio-app, .gradio-container {
    background: var(--paper) !important;
    color: var(--ink) !important;
    font-family: var(--sans) !important;
  }
  .gradio-container { max-width: 1400px !important; padding: 0 28px 32px !important; }
  footer, .footer, .show-api, .built-with { display: none !important; }

  /* Subtle paper grain */
  body::before {
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
      radial-gradient(circle at 20% 30%, rgba(0,0,0,0.025) 0, transparent 40%),
      radial-gradient(circle at 80% 70%, rgba(0,0,0,0.02) 0, transparent 40%);
    mix-blend-mode: multiply;
  }

  /* ---- Masthead ---- */
  .masthead {
    border-top: 4px solid var(--ink);
    border-bottom: 1px solid var(--ink);
    padding: 18px 0 12px;
    margin-bottom: 18px;
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: end;
    gap: 24px;
  }
  .masthead .meta {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink-soft);
    line-height: 1.6;
  }
  .masthead .meta.right { text-align: right; }
  .masthead h1 {
    font-family: var(--serif);
    font-style: italic;
    font-weight: 400;
    font-size: 78px;
    line-height: 0.95;
    letter-spacing: -0.02em;
    margin: 0;
    text-align: center;
    color: var(--ink);
  }
  .masthead h1 .amp { font-style: italic; opacity: 0.5; }
  .masthead .strap {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    text-align: center;
    margin-top: 6px;
    color: var(--ink-soft);
  }
  .masthead-rule {
    height: 4px;
    border-top: 1px solid var(--ink);
    border-bottom: 1px solid var(--ink);
    margin-bottom: 28px;
  }

  /* ---- Two-col grid via Gradio Row ---- */
  #main-row { gap: 32px !important; align-items: stretch; }

  /* Section headers — small caps editorial */
  .sect-h {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--ink-soft);
    border-bottom: 1px solid var(--rule);
    padding-bottom: 8px;
    margin: 0 0 14px;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }
  .sect-h .num { font-style: italic; font-family: var(--serif); font-size: 14px; letter-spacing: 0; color: var(--ink); }

  /* ---- Chat column ---- */
  #chat-col { background: transparent; }

  /* Gradio 6 Chatbot reset — targets actual DOM: #chatbot.block */
  #chatbot, #chatbot > div, #chatbot .wrapper, .bubble-wrap {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--ink) !important;
  }
  /* Gradio 6 message bubbles */
  #chatbot .message-wrap, #chatbot .prose, #chatbot [class*="message"] {
    background: transparent !important;
    color: var(--ink) !important;
  }
  .message-row { padding: 0 !important; }
  .message, #chatbot .message {
    background: transparent !important;
    border: none !important;
    padding: 14px 0 !important;
    border-bottom: 1px solid var(--rule-soft) !important;
    border-radius: 0 !important;
    color: var(--ink) !important;
    font-family: var(--sans) !important;
    font-size: 16px !important;
    line-height: 1.55 !important;
    max-width: 100% !important;
  }
  .message.user, #chatbot [data-testid="user"] {
    font-family: var(--serif) !important;
    font-size: 22px !important;
    font-style: italic !important;
    line-height: 1.3 !important;
    color: var(--ink) !important;
    background: transparent !important;
  }
  .message.user::before {
    content: "Q.";
    font-family: var(--mono); font-style: normal;
    font-size: 11px; letter-spacing: 0.2em;
    color: var(--gavel); margin-right: 10px; vertical-align: 2px;
  }
  .message.bot::before {
    content: "VERDICT —";
    display: block;
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.3em;
    color: var(--gavel); margin-bottom: 8px;
  }
  .message strong, .message b { font-weight: 600; }
  .message em, .message i { font-family: var(--serif); font-size: 1.05em; }

  /* Hide Gradio avatar block (but not the .role flex-wrap that holds message content in v6+) */
  .avatar-container { display: none !important; }

  /* ---- Input row ---- */
  #input-row {
    border-top: 2px solid var(--ink);
    border-bottom: 1px solid var(--ink);
    padding: 12px 0 !important;
    margin-top: 18px !important;
    gap: 12px !important;
  }
  #msg-input textarea, #msg-input input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-family: var(--serif) !important;
    font-size: 22px !important;
    font-style: italic !important;
    color: var(--ink) !important;
    padding: 8px 4px !important;
  }
  #msg-input textarea::placeholder { color: var(--ink-soft) !important; opacity: 0.55; }
  #msg-input { background: transparent !important; border: none !important; }

  #send-btn, #clear-btn {
    background: var(--ink) !important;
    color: var(--paper) !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.25em !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
    border: none !important;
    padding: 14px 22px !important;
    box-shadow: none !important;
  }
  #send-btn:hover, #clear-btn:hover { background: var(--gavel) !important; }
  #clear-btn {
    background: transparent !important;
    color: var(--ink-soft) !important;
    border: 1px solid var(--rule-soft) !important;
    margin-top: 14px;
  }
  #clear-btn:hover { color: var(--paper) !important; background: var(--ink) !important; border-color: var(--ink) !important; }

  /* ---- The Floor (right column custom HTML) ---- */
  .floor { display: flex; flex-direction: column; gap: 14px; }
  .lane {
    border: 1px solid var(--rule);
    background: var(--paper);
    padding: 14px 16px 16px;
    position: relative;
    transition: background 240ms ease, border-color 240ms ease;
  }
  .lane[data-state="active"] {
    background: linear-gradient(180deg, var(--paper) 0%, var(--paper-2) 100%);
    border-color: var(--ink);
    box-shadow: 3px 3px 0 var(--ink);
  }
  .lane[data-state="done"] { opacity: 0.78; }
  .lane[data-state="idle"] .lane-stream { opacity: 0.35; }

  .lane-head { display: flex; align-items: baseline; gap: 12px; margin-bottom: 8px; }
  .lane-num {
    font-family: var(--serif); font-style: italic;
    font-size: 32px; line-height: 1;
    color: var(--lane-color, var(--ink));
    min-width: 36px;
  }
  .lane-name {
    font-family: var(--sans); font-size: 14px; font-weight: 500;
    color: var(--ink); letter-spacing: -0.01em;
    flex: 1;
  }
  .lane-name small {
    display: block;
    font-family: var(--mono); font-size: 9px;
    letter-spacing: 0.22em; text-transform: uppercase;
    color: var(--ink-soft); font-weight: 400; margin-top: 2px;
  }
  .lane-status {
    font-family: var(--mono); font-size: 9px; letter-spacing: 0.2em;
    text-transform: uppercase; color: var(--ink-soft);
    display: flex; align-items: center; gap: 6px;
  }
  .lane-status .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--rule-soft);
  }
  .lane[data-state="active"] .lane-status .dot {
    background: var(--lane-color, var(--ink));
    box-shadow: 0 0 0 0 var(--lane-color, var(--ink));
    animation: pulse 1.4s infinite;
  }
  .lane[data-state="done"] .lane-status .dot { background: var(--lane-color, var(--ink)); }
  .lane[data-state="active"] .lane-status { color: var(--lane-color, var(--ink)); }
  @keyframes pulse {
    0%   { box-shadow: 0 0 0 0 var(--lane-color, var(--ink)); }
    70%  { box-shadow: 0 0 0 8px transparent; }
    100% { box-shadow: 0 0 0 0 transparent; }
  }

  .lane-stream {
    font-family: var(--mono); font-size: 12px; line-height: 1.65;
    color: var(--ink); white-space: pre-wrap; word-break: break-word;
    max-height: 140px; overflow-y: auto;
    padding-top: 8px; border-top: 1px dashed var(--rule-soft);
    transition: opacity 240ms ease;
  }
  .lane-stream:empty::before {
    content: attr(data-placeholder);
    color: var(--ink-soft); opacity: 0.5; font-style: italic;
  }
  .lane-stream::-webkit-scrollbar { width: 4px; }
  .lane-stream::-webkit-scrollbar-thumb { background: var(--rule-soft); }

  /* The judge lane is special */
  .lane.judge { background: var(--ink); color: var(--paper); border-color: var(--ink); margin-top: 6px; }
  .lane.judge .lane-num { color: var(--gavel); font-size: 38px; }
  .lane.judge .lane-name { color: var(--paper); font-weight: 600; letter-spacing: 0.04em; }
  .lane.judge .lane-name small { color: rgba(244,239,226,0.55); }
  .lane.judge .lane-status { color: rgba(244,239,226,0.65); }
  .lane.judge .lane-stream { color: rgba(244,239,226,0.92); border-top-color: rgba(244,239,226,0.18); }
  .lane.judge[data-state="active"] { box-shadow: 4px 4px 0 var(--gavel); }
  .lane.judge .gavel-mark {
    position: absolute; top: 12px; right: 14px;
    font-family: var(--serif); font-style: italic; font-size: 18px;
    color: var(--gavel); opacity: 0.85;
  }

  /* Floor footer — connection state */
  .floor-foot {
    margin-top: 4px; padding-top: 10px;
    border-top: 1px solid var(--rule-soft);
    display: flex; justify-content: space-between;
    font-family: var(--mono); font-size: 9px; letter-spacing: 0.22em;
    text-transform: uppercase; color: var(--ink-soft);
  }
  .floor-foot .conn-dot {
    display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    background: var(--ink-soft); margin-right: 6px; vertical-align: 1px;
  }
  .floor-foot[data-conn="live"]    .conn-dot { background: oklch(0.65 0.16 145); }
  .floor-foot[data-conn="waiting"] .conn-dot { background: var(--gold); }

  /* Hide Gradio default markdown headers from the chatbot Markdown widget */
  .prose h1, .prose h2, .prose h3 { font-family: var(--serif) !important; font-weight: 400 !important; }
</style>

<script>
(function() {
  // Force light mode — this UI is light-only; Gradio picks up system dark mode otherwise
  function enforceLightMode() {
    if (document.body && document.body.classList.contains('dark')) {
      document.body.classList.remove('dark');
    }
  }
  enforceLightMode();
  var _modeObs = new MutationObserver(enforceLightMode);
  _modeObs.observe(document.documentElement, { childList: true, subtree: false });
  document.addEventListener('DOMContentLoaded', function() {
    enforceLightMode();
    _modeObs.observe(document.body, { attributes: true, attributeFilter: ['class'] });
  });

  var source = null;
  var _runComplete = false;

  // phase ids → lane element ids
  var LANE_IDS = {
    model_a: 'lane-a',
    model_b: 'lane-b',
    model_c: 'lane-c',
    round:   'lane-round',
    judge:   'lane-judge'
  };
  var ORDER = ['model_a','model_b','model_c','round','judge'];

  function setLane(phase, state) {
    var id = LANE_IDS[phase]; if (!id) return;
    var el = document.getElementById(id); if (!el) return;
    el.setAttribute('data-state', state);
    var status = el.querySelector('.lane-status .label');
    if (status) {
      status.textContent = state === 'active' ? 'On the floor' :
                           state === 'done'   ? 'Yielded'      :
                                                'Awaiting';
    }
  }

  var currentPhase = null;

  function setStatus(label, mode) {
    var foot = document.getElementById('floor-foot');
    if (!foot) return;
    foot.setAttribute('data-conn', mode || 'idle');
    var lbl = foot.querySelector('.label');
    if (lbl) lbl.textContent = label;
  }

  function clearAll() {
    ORDER.forEach(function(p) {
      setLane(p, 'idle');
      var id = LANE_IDS[p];
      var el = document.getElementById(id);
      if (el) {
        var s = el.querySelector('.lane-stream');
        if (s) s.textContent = '';
      }
    });
    currentPhase = null;
  }

  function appendToken(token) {
    if (!currentPhase) currentPhase = 'model_a';
    var id = LANE_IDS[currentPhase]; if (!id) return;
    var el = document.getElementById(id); if (!el) return;
    var s = el.querySelector('.lane-stream');
    if (!s) return;
    s.textContent += token;
    s.scrollTop = s.scrollHeight;
  }

  function startPhase(phase) {
    if (_runComplete) {
      clearAll();
      _runComplete = false;
    }
    if (currentPhase && currentPhase !== phase) {
      setLane(currentPhase, 'done');
    }
    currentPhase = phase;
    setLane(phase, 'active');
  }

  function connect() {
    if (source) return;
    source = new EventSource('http://localhost:7861/stream');
    setStatus('Connecting', 'waiting');

    source.onmessage = function(e) {
      try {
        var d = JSON.parse(e.data);
        if (d.token != null) {
          appendToken(d.token);
          setStatus('Live · streaming', 'live');
        }
      } catch(_) {}
    };
    source.addEventListener('phase', function(e) {
      try {
        var d = JSON.parse(e.data);
        if (d.phase) startPhase(d.phase);
      } catch(_) {}
    });
    source.addEventListener('done', function() {
      if (currentPhase) setLane(currentPhase, 'done');
      currentPhase = null;
      _runComplete = true;
      setStatus('Court adjourned', 'idle');
      source.close(); source = null;
      setTimeout(function() {
        // Re-arm for the next turn — keep deliberation content visible
        setStatus('Awaiting query', 'idle');
        connect();
      }, 1800);
    });
    source.onerror = function() {
      if (source) { source.close(); source = null; }
      setStatus('Reconnecting', 'waiting');
      setTimeout(connect, 1500);
    };
  }

  var obs = new MutationObserver(function() {
    if (document.getElementById('lane-a') && document.getElementById('floor-foot')) {
      obs.disconnect();
      setStatus('Awaiting query', 'idle');
      connect();
    }
  });
  obs.observe(document.body, { childList: true, subtree: true });
})();

;(function() {
  function watchClear() {
    var btn = document.getElementById('clear-btn');
    if (btn) {
      btn.addEventListener('click', function() {
        clearAll();
        _runComplete = false;
        setStatus('Awaiting query', 'idle');
      });
      return;
    }
    setTimeout(watchClear, 300);
  }
  watchClear();
})();
</script>
"""

MASTHEAD_HTML = f"""
<div class="masthead">
  <div class="meta">
    <div>VOL. I · NO. 001</div>
    <div>{ISSUE_DATE}</div>
  </div>
  <div>
    <h1>The <span class="amp">&</span> Debate</h1>
    <div class="strap">Three Models · One Verdict · In Open Court</div>
  </div>
  <div class="meta right">
    <div>POCKETFLOW EDITION</div>
    <div>STREAMING · LIVE</div>
  </div>
</div>
<div class="masthead-rule"></div>
"""

FLOOR_HTML = """
<div class="floor">
  <div class="sect-h"><span>The Floor — Live Deliberation</span><span class="num">II</span></div>

  <div class="lane" id="lane-a" data-state="idle" style="--lane-color: var(--gold);">
    <div class="lane-head">
      <div class="lane-num">I.</div>
      <div class="lane-name">Model A<small>Direct · Practical</small></div>
      <div class="lane-status"><span class="dot"></span><span class="label">Awaiting</span></div>
    </div>
    <div class="lane-stream" data-placeholder="—— quiet ——"></div>
  </div>

  <div class="lane" id="lane-b" data-state="idle" style="--lane-color: var(--teal);">
    <div class="lane-head">
      <div class="lane-num">II.</div>
      <div class="lane-name">Model B<small>Cautious · Analytical</small></div>
      <div class="lane-status"><span class="dot"></span><span class="label">Awaiting</span></div>
    </div>
    <div class="lane-stream" data-placeholder="—— quiet ——"></div>
  </div>

  <div class="lane" id="lane-c" data-state="idle" style="--lane-color: var(--olive);">
    <div class="lane-head">
      <div class="lane-num">III.</div>
      <div class="lane-name">Model C<small>Creative · Alternative</small></div>
      <div class="lane-status"><span class="dot"></span><span class="label">Awaiting</span></div>
    </div>
    <div class="lane-stream" data-placeholder="—— quiet ——"></div>
  </div>

  <div class="lane" id="lane-round" data-state="idle" style="--lane-color: var(--ink);">
    <div class="lane-head">
      <div class="lane-num">§</div>
      <div class="lane-name">Cross-Examination<small>Critique round</small></div>
      <div class="lane-status"><span class="dot"></span><span class="label">Awaiting</span></div>
    </div>
    <div class="lane-stream" data-placeholder="—— quiet ——"></div>
  </div>

  <div class="lane judge" id="lane-judge" data-state="idle" style="--lane-color: var(--gavel);">
    <div class="gavel-mark">⚖</div>
    <div class="lane-head">
      <div class="lane-num">¶</div>
      <div class="lane-name">The Judge<small>Synthesis · Ruling</small></div>
      <div class="lane-status"><span class="dot"></span><span class="label">Awaiting</span></div>
    </div>
    <div class="lane-stream" data-placeholder="—— chambers in session ——"></div>
  </div>

  <div class="floor-foot" id="floor-foot" data-conn="idle">
    <span><span class="conn-dot"></span><span class="label">Awaiting query</span></span>
    <span>SSE · :7861/stream</span>
  </div>
</div>
"""

CHAT_HEADER_HTML = """
<div class="sect-h"><span>Chamber — Question &amp; Verdict</span><span class="num">I</span></div>
"""

with gr.Blocks(
    fill_height=True,
    title="The Debate · PocketFlow",
) as demo:
    uuid_state = gr.State(uuid.uuid4())

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
        run_debate, [chatbot, uuid_state], [chatbot]
    )
    submit_btn.click(add_user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        run_debate, [chatbot, uuid_state], [chatbot]
    )
    clear_btn.click(
        lambda: ([{"role": "assistant", "content": "_The bench is empty. Pose a question and three models will take the floor._"}], uuid.uuid4()),
        None,
        [chatbot, uuid_state],
    )

demo.queue(default_concurrency_limit=5)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(), head=SSE_JS)
    try:
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        pass
