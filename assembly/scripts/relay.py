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
        os.environ.setdefault(k, v)


load_env()
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-20b")
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
RELAY_PORT = int(os.environ.get("RELAY_PORT", "8081"))
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT = 60.0
MAX_IN = 8192
SYSTEM_PROMPT = """You are a precise, capable assistant. Answer correctly and
directly. Treat every explicit output constraint from the user as mandatory.
When the user asks for an exact form, only the answer, or no other text, emit
only that content: no preamble, explanation, label, or Markdown fence.
Otherwise give the minimum detail needed to answer well. Silently verify
arithmetic, logic, and factual cause-and-effect claims before responding. If
the available information is insufficient, state that clearly instead of
inventing an answer."""


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
        payload = json.dumps(
            {
                "model": MODEL,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": msg},
                ],
            }
        ).encode("utf-8")
        req = urllib.request.Request(OR_URL, data=payload, method="POST")
        req.add_header("Authorization", "Bearer " + API_KEY)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                data = json.loads(r.read().decode("utf-8"))
            reply = data["choices"][0]["message"]["content"].strip()
            self._text(200, reply if reply else "(empty model reply)")
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
