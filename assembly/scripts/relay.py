#!/usr/bin/env python3
"""asm-chat LLM relay (PLAN §2.10, Milestone 8).

Keeps TLS, auth, JSON, and provider differences OUT of the assembly server.
Plain-text contract consumed by engine_gateway.asm:

    POST /generate   Content-Type: text/plain   body = user message
    -> 200 OK        Content-Type: text/plain    body = model reply
       (4xx/5xx on any failure with a short text body)

Reads OPENROUTER_MODEL and OPENROUTER_API_KEY from the environment, falling
back to ../.env (the repo-root shared credentials file).
"""
import http.server
import socketserver
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

_DQUOTE = chr(34)   # "
_SQUOTE = chr(39)   # '


def load_env() -> None:
    # relay.py lives at <repo>/assembly/scripts; shared credentials live at
    # <repo>/.env as documented by every language implementation.
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in (_DQUOTE, _SQUOTE):
            v = v[1:-1]
        # The repository's selected model is intentionally controlled by
        # .env. Preserve process-level secret overrides for the API key.
        if k == "OPENROUTER_MODEL":
            os.environ[k] = v
        else:
            os.environ.setdefault(k, v)


load_env()
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-20b")
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
RELAY_PORT = int(os.environ.get("RELAY_PORT", "8081"))
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT = 60.0
MAX_IN = 8192
UPSTREAM_ATTEMPTS = 3
SYSTEM_PROMPT = """You are a precise, capable assistant. Answer correctly and
directly. Treat every explicit output constraint from the user as mandatory.
When the user asks for an exact form, only the answer, or no other text, emit
only that content: no preamble, explanation, label, or Markdown fence.
Otherwise give the minimum detail needed to answer well. Silently verify
arithmetic, logic, factual cause-and-effect claims, and requested text
transformations before responding. For transformations, compare the result to
the source and ensure no requested token was lost. If the available information
is insufficient, state that clearly instead of inventing an answer.

Use this internal procedure: solve the task, check the proposed answer against
the original input and every output constraint, correct any discrepancy, then
emit only the final response. Do not reveal this internal procedure.

Examples:
User: Reverse `red green blue`. Reply exactly with single spaces.
Assistant: blue green red
User: Compute 17 + 8 * 3. Reply exactly with the number.
Assistant: 41
User: All nims are lats. All lats are zogs. Must all nims be zogs? Reply exactly YES or NO.
Assistant: YES
User: Which is correct? A) Earth orbits the Sun B) The Sun orbits Earth. Reply exactly with the letter.
Assistant: A
User: Python: `print(7 // 2)`. Reply exactly with the output.
Assistant: 3"""


def extract_reply(data):
    """Return non-empty text from an OpenAI-compatible response."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("missing choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("missing message")
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        joined = "".join(parts).strip()
        if joined:
            return joined
    raise ValueError("empty content")


def request_completion(msg):
    """Call OpenRouter, retrying transient HTTP and zero-content responses."""
    payload = json.dumps(
        {
            "model": MODEL,
            "temperature": 0,
            "provider": {"require_parameters": True},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": msg},
            ],
        }
    ).encode("utf-8")
    last_error = None
    for attempt in range(UPSTREAM_ATTEMPTS):
        req = urllib.request.Request(OR_URL, data=payload, method="POST")
        req.add_header("Authorization", "Bearer " + API_KEY)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
                data = json.loads(response.read().decode("utf-8"))
            return extract_reply(data)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (408, 409, 429) and exc.code < 500:
                raise
        except (json.JSONDecodeError, TypeError, ValueError, KeyError, IndexError) as exc:
            last_error = exc
        if attempt + 1 < UPSTREAM_ATTEMPTS:
            time.sleep(0.25 * (attempt + 1))
    raise last_error


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # silence default access logging
        pass

    def _text(self, code, body):
        b = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):  # health check
        self._text(200, "relay ok model=%s port=%d" % (MODEL, RELAY_PORT))

    def do_POST(self):
        if self.path != "/generate":
            return self._text(404, "not found")
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            return self._text(400, "bad content-length")
        if n <= 0 or n > MAX_IN:
            return self._text(400, "empty or too large")
        msg = self.rfile.read(n).decode("utf-8", "replace")
        if not API_KEY:
            return self._text(502, "relay: OPENROUTER_API_KEY not set")
        try:
            self._text(200, request_completion(msg))
        except urllib.error.HTTPError as e:
            self._text(502, "relay: openrouter http %d" % e.code)
        except Exception as e:
            self._text(502, "relay: %s" % type(e).__name__)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    sys.stderr.write("[relay] model=%s port=%d\n" % (MODEL, RELAY_PORT))
    sys.stderr.flush()
    with Server(("127.0.0.1", RELAY_PORT), Handler) as s:
        try:
            s.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
