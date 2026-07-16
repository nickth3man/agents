# asm-chat

A native Windows x86-64 chat agent written in NASM syntax assembly, linked with
GoLink, using no C runtime. It serves a small browser UI over loopback
(`127.0.0.1:8080`) and replies with a deterministic ELIZA-style rules engine.

See [`PLAN.md`](./PLAN.md) for the full architecture and milestone roadmap.

## Status

Milestone 0 in progress — toolchain + repository baseline. See the todo list.

## Prerequisites

| Tool     | Purpose                  | Install                                                                 |
| -------- | ------------------------ | ----------------------------------------------------------------------- |
| NASM     | Assembler                | https://www.nasm.us/  ·  `winget install nasm`                          |
| GoLink   | Primary linker           | https://www.godevtool.com/  (manual download; not in winget/choco)      |
| curl.exe | Smoke tests              | Bundled with Windows 10+                                                |
| PowerShell 5.1+ | Build orchestration  | Bundled with Windows                                                    |

Optional: Visual Studio Build Tools (for the `link.exe` alternative path),
x64dbg, Sysinternals Process Monitor, WinDbg.

If NASM/GoLink are not on `PATH`, set env vars `NASM_PATH` and `GOLINK_PATH` to
their full executable paths, or pass `-NasmPath` / `-GoLinkPath` to `build.ps1`.

## Build

```powershell
powershell.exe -NoProfile -File .\scripts\build.ps1 -Preflight
powershell.exe -NoProfile -File .\scripts\build.ps1
```

## Run

```powershell
.\out\current\chat-agent.exe
```

Then open http://127.0.0.1:8080/ in a browser.

## Test

```powershell
powershell.exe -NoProfile -File .\scripts\test.ps1 -All
```

## Development loop

```powershell
powershell.exe -NoProfile -File .\scripts\dev.ps1
```

## Layout

See [PLAN.md §3](./PLAN.md#3-repository-and-module-layout) for the module map.

## Constraints (core release)

Loopback only · one connection at a time · no TLS · no threads · no heap ·
HTTP/1.1 subset (exact methods/paths, single `Content-Length`, `Connection: close`).
