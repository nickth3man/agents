# asm-chat

A native Windows x86-64 chat agent written in NASM syntax assembly, linked with
GoLink, using **no C runtime**. It serves a small browser UI over loopback
(`127.0.0.1:8080`) and answers `POST /chat` through a native Assembly
OpenRouter HTTPS client.

See [`PLAN.md`](./PLAN.md) for the full architecture and milestone roadmap.

## Status

Core milestones **complete and verified**:

| Milestone | What it delivers |
| --- | --- |
| M0 toolchain + repo baseline | NASM + GoLink (SHA-256 verified), preflight, skeleton |
| M1 ABI foundation | no-CRT exe, KERNEL32-only imports, call-frame discipline |
| M2 TCP echo server | blocking accept loop, `SO_EXCLUSIVEADDRUSE`, timeouts |
| M3 HTTP server | bounded reader, request-line + Content-Length parser, response builder |
| M4 router + embedded page | `GET /` `/version` `POST /chat`, 404/405/411, `chat.html` embedded via `incbin` |
| M5 dev loop | `dev.ps1` watcher: debounce, rebuild, restart, smoke, last-known-good, crash recovery |
| M7 hardening + dist | debug canaries/chunk caps, release build, `dist/chat-agent.exe`, 27 protocol tests, standalone run |
| M8 LLM gateway | `engine_gateway.asm` + WinHTTP → OpenRouter; `502` on upstream failure |

The `/chat` engine is implemented completely in Assembly. `engine_gateway.asm`
uses the Windows WinHTTP API for TLS, builds and escapes the OpenRouter JSON
request, authenticates it, and decodes the returned JSON content. There is no
runtime relay or non-LLM answer path.

## Prerequisites

| Tool | Purpose | Install |
| --- | --- | --- |
| NASM | Assembler | https://www.nasm.us/ · `winget install nasm` |
| GoLink | Primary linker | https://www.godevtool.com/ (manual download) |
| curl.exe | Smoke tests | Bundled with Windows 10+ |
| PowerShell 5.1+ | Build orchestration | Bundled with Windows |
| Python 3 | Intelligence evaluation scripts only; not used by the agent | https://python.org |

If NASM/GoLink are not on `PATH`, set env vars `NASM_PATH` and `GOLINK_PATH` to
their full executable paths, or pass `-NasmPath` / `-GoLinkPath` to `build.ps1`.

The run script reads `OPENROUTER_MODEL` and `OPENROUTER_API_KEY` from the shared
repo-root `.env`; the Assembly executable reads those inherited values directly.
When launching the executable without `run.ps1`, set both environment variables
in the parent process first.

## Build

```powershell
powershell.exe -NoProfile -File .\scripts\build.ps1 -Preflight   # check tools
powershell.exe -NoProfile -File .\scripts\build.ps1 -Config Debug
powershell.exe -NoProfile -File .\scripts\build.ps1 -Config Release
powershell.exe -NoProfile -File .\scripts\build.ps1 -Linker msvc # optional MSVC linker
```

Artifact: `out\current\chat-agent.exe`. Release copy: `dist\chat-agent.exe`.

## Run

```powershell
# server only (GET /, /version, routing work standalone):
.\out\current\chat-agent.exe

# LLM-powered /chat (one native Assembly process):
powershell.exe -NoProfile -File .\scripts\run.ps1
```

The detached launcher starts the agent and returns the terminal:

```powershell
powershell -NoProfile -File .\scripts\run.ps1    # start
powershell -NoProfile -File .\scripts\stop.ps1   # stop
```

Then open http://127.0.0.1:8080/ in a browser.

## Test

```powershell
powershell.exe -NoProfile -File .\scripts\test.ps1 -All
```

Cases: Smoke, Routes, OversizedBody (413), ContentLength (including overflow and
unsupported transfer coding), FragmentedRequest, Malformed, Repeat (100×), and
an opt-in `-Gateway` live-model check. Always uses `curl.exe`, never the alias.

## Development loop

```powershell
powershell.exe -NoProfile -File .\scripts\dev.ps1
```

Watches `src/`, `include/`, `web/`; on save: new build ID → rebuild (old server
keeps serving) → restart → `/version` readiness poll → smoke test → mark
last-known-good. Recovers from crashes (restarts LKG) and from build failures
(old server retained). The browser polls `/version` and reloads on change.

## Architecture (one-liner per module)

`start` (entry+accept loop) · `net_init`/`net_io` (winsock lifecycle, `send_all`,
timeouts) · `http_read` (bounded reader state machine) · `http_parse` (request
line, Content-Length, Transfer-Encoding) · `http_write` (response builder) ·
`router` (exact dispatch) · `engine_gateway` (native WinHTTP/OpenRouter client) · `assets`
(embedded `chat.html` + build ID) · `decimal`/`text` (checked decimal + byte
utils) · `log` (stderr structured logging) · `state` (.bss buffers). See
[PLAN.md §3](./PLAN.md#3-repository-and-module-layout).

## Constraints (core release)

Loopback only · one connection at a time · outbound TLS through WinHTTP in Assembly
· no threads · no heap · HTTP/1.1 subset (exact methods/paths, single
`Content-Length`, `Connection: close`, graceful `shutdown(SD_SEND)`).
