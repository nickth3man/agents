# Repository Guidelines

## Project Structure & Module Organization

This repository implements a no-CRT Windows x86-64 chat server in NASM assembly. Production modules live in `src/`: `start.asm` owns the polling loop, `http_*` handles protocol framing, `router.asm` dispatches endpoints, and `engine_gateway.asm` talks to OpenRouter through WinHTTP. Shared ABI declarations, constants, capacities, and structures belong in `include/`. The browser client is the self-contained `web/chat.html`. PowerShell automation lives in `scripts/`; vendored GoLink tooling is under `tools/`. Treat `generated/`, `out/`, and `dist/` as build outputs, not hand-edited source.

## Build, Test, and Development Commands

Run commands from the `assembly\` project directory:

```powershell
powershell.exe -NoProfile -File .\scripts\build.ps1 -Preflight
powershell.exe -NoProfile -File .\scripts\build.ps1 -Config Debug
powershell.exe -NoProfile -File .\scripts\build.ps1 -Config Release
powershell.exe -NoProfile -File .\scripts\test.ps1 -All
powershell.exe -NoProfile -File .\scripts\dev.ps1
```

Preflight verifies NASM, curl, and the linker. Builds publish `out\current\chat-agent.exe`; Release also updates `dist\`. Tests require a running server on `127.0.0.1:8080`. Use `scripts\run.ps1` and `scripts\stop.ps1` to start or stop it. The development loop watches `src/`, `include/`, and `web/`, then rebuilds, restarts, and smoke-tests changes.

## Coding Style & Naming Conventions

Use four-space indentation and align assembly operands and comments consistently with nearby code. Name exported routines and state in `snake_case`, local labels as `.descriptive_label`, and constants in `UPPER_SNAKE_CASE`. Every module uses `default rel`; never write `[label + register]` because GoLink cannot relocate that form safely. Follow the Win64 ABI rules in `include/win64.inc`: reserve shadow space, align `RSP` before calls, preserve nonvolatile registers, and document callable routines with inputs, outputs, clobbers, and bounds. No automated formatter or linter is configured.

## Testing Guidelines

`scripts/test.ps1` is the protocol test harness, using `curl.exe` and `TcpClient`. Add focused `Check` cases for route, framing, timeout, concurrency, and error-path changes. Run `-All` before committing; use `-Gateway` separately when live OpenRouter credentials are available. No coverage threshold is defined.

## Git & Commit Workflow

This is a solo project. Make changes directly on `main` in the parent repository at `C:\Users\nicolas\Documents\GitHub\agents`; do not create feature branches. Commit early and often at logical checkpoints, such as after a focused refactor, test addition, or build fix. Keep each commit limited to one coherent change and avoid bundling unrelated edits. Follow the existing Conventional Commit style: `feat:`, `fix:`, `refactor:`, or scoped forms such as `test(assembly):`. Before committing, review the diff and run the relevant build and test commands.

## Security & Configuration

Never commit credentials. `scripts/run.ps1` reads `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` from the shared parent `.env`. Preserve loopback-only binding, bounded static buffers, and generated-file exclusions.
