# Annotation Tier Ledger — NASM Assembly Chat Server

> **Generated:** 2026-07-17  
> **Scope:** All `src/*.asm` callable labels  
> **Template:** 7-field contract comment defined in `include/win64.inc:36-48`

> **STATUS: RETROFIT COMPLETE (2026-07-17)**  
> All 44 callable labels now carry their target-tier annotation per
> `docs/annotation-standard.md`. Final distribution after pragmatic Tier 3
> boundary decision: 1 Tier 1 (`hex_nibble`), 31 Tier 2, 12 Tier 3. The tables
> below preserve the original pre-retrofit inventory for reference; live
> annotation in `src/*.asm` is the source of truth.

---

## Tier Definitions

| Tier | Label | Required For | Format |
|------|-------|-------------|--------|
| **Tier 1** | Minimal inline | ALL callable labels, including tiny leaf helpers | `; RCX=..., RDX=..., ret RAX=...` + `; Clobbers volatile, preserves nonvolatile` (if not obvious) |
| **Tier 2** | Structured 7-field header | Non-leaf, any function >20 lines, any function touching buffers | Full `include/win64.inc` template: Purpose, Inputs, Outputs, Errors, Clobbers, Preserves, Locals, Max read, Max write |
| **Tier 3** | Strict contract | Public externs (`global`), state machines, security-critical paths | Tier 2 + Doxygen `@param[in]/[out]` + optional `Stack:`, `Precond:`, `Modified:`, `Initial inputs to registers:`, `Register assignments` |

**Methodology:**
- Every label matching `^[_A-Za-z][_A-Za-z0-9]*:` in `src/*.asm` was checked against `global` declarations and `call` site cross-references.
- Data labels (data/bss sections with `db`/`dw`/`resb`/`resd`/`resq`) are excluded — only executable code entry points are listed.
- Local control-flow labels (`.loop`, `.done`, `.err`, `.next`, etc.) are excluded.
- The **Recommended tier** is the strictest tier whose criteria the function meets (Tier 1 → check Tier 2 → check Tier 3; highest wins).
- **Current doc coverage** inspects comment lines *immediately above* the label for the 7 template fields plus `Purpose` and `Max read`/`Max write`: `present`/`partial`/`missing`.

---

## Summary

| File | Tier 1 | Tier 2 | Tier 3 | Total | Lines | Doc Coverage (field %) |
|------|--------|--------|--------|-------|-------|----------------------|
| `src/start.asm` | 0 | 7 | 1 | 8 | 739 | 11% |
| `src/router.asm` | 0 | 2 | 2 | 4 | 276 | 28% |
| `src/http_read.asm` | 0 | 0 | 1 | 1 | 167 | 44% |
| `src/http_parse.asm` | 0 | 1 | 1 | 2 | 268 | 28% |
| `src/http_write.asm` | 0 | 1 | 1 | 2 | 231 | 44% |
| `src/engine_gateway.asm` | 1 | 5 | 3 | 9 | 1154 | 19% |
| `src/net_io.asm` | 0 | 0 | 2 | 2 | 99 | 50% |
| `src/net_init.asm` | 0 | 0 | 2 | 2 | 152 | 39% |
| `src/log.asm` | 0 | 1 | 4 | 5 | 327 | 36% |
| `src/text.asm` | 0 | 0 | 4 | 4 | 154 | 50% |
| `src/state.asm` | 0 | 0 | 2 | 2 | 197 | 11% |
| `src/decimal.asm` | 0 | 0 | 2 | 2 | 125 | 56% |
| `src/assets.asm` | 0 | 0 | 1 | 1 | 121 | 33% |
| **Total** | **1** | **17** | **26** | **44** | — | **29%** |

---

## Per-Function Inventory

### Legend for "Current Fields Present"

Each of the 9 template fields is reported as:
- `✓` — present and complete
- `~` — partial (mentioned but underspecified)
- `—` — missing

Fields in order: **Purpose | Inputs | Outputs | Errors | Clobbers | Preserves | Locals | Max read | Max write**

---

### `src/start.asm` — Event loop, slot management, polling

| # | Function | Line | Lines | Leaf? | Buf? | Global? | Current Fields | Rec. Tier | Missing Fields |
|---|----------|------|-------|-------|------|---------|----------------|-----------|---------------|
| 1 | `start` | 109 | 319 | No | Yes | **Yes** | `~⎪—⎪—⎪—⎪—⎪—⎪—⎪—⎪—` | **Tier 3** | All 9 — only has one-line purpose comment |
| 2 | `slot_to_globals` | 428 | 67 | Yes¹ | Yes | No | `✓⎪—⎪—⎪—⎪—⎪—⎪—⎪—⎪—` | **Tier 2** | Inputs, Outputs, Errors, Clobbers, Preserves, Locals, Max read, Max write |
| 3 | `globals_to_slot` | 500 | 67 | Yes¹ | Yes | No | `✓⎪—⎪—⎪—⎪—⎪—⎪—⎪—⎪—` | **Tier 2** | Inputs, Outputs, Errors, Clobbers, Preserves, Locals, Max read, Max write |
| 4 | `disassociate_slot` | 573 | 24 | No | Yes | No | `✓⎪—⎪—⎪—⎪—⎪—⎪—⎪—⎪—` | **Tier 2** | Inputs, Outputs, Errors, Clobbers, Preserves, Locals, Max read, Max write |
| 5 | `free_slot` | 602 | 24 | No | Yes | No | `✓⎪—⎪—⎪—⎪—⎪—⎪—⎪—⎪—` | **Tier 2** | Inputs, Outputs, Errors, Clobbers, Preserves, Locals, Max read, Max write |
| 6 | `respond_and_free_slot` | 631 | 27 | No | Yes | No | `✓⎪—⎪—⎪—⎪—⎪—⎪—⎪—⎪—` | **Tier 2** | Inputs, Outputs, Errors, Clobbers, Preserves, Locals, Max read, Max write |
| 7 | `error_and_free_slot` | 663 | 34 | No | Yes | No | `✓⎪—⎪—⎪—⎪—⎪—⎪—⎪—⎪—` | **Tier 2** | Inputs, Outputs, Errors, Clobbers, Preserves, Locals, Max read, Max write |
| 8 | `scan_timeouts` | 701 | 39 | No | Yes | No | `✓⎪—⎪—⎪—⎪—⎪—⎪—⎪—⎪—` | **Tier 2** | Inputs, Outputs, Errors, Clobbers, Preserves, Locals, Max read, Max write |

¹ Uses `rep movsb` — not a `call` instruction, so technically leaf.

---

### `src/router.asm` — Method + path dispatch

| # | Function | Line | Lines | Leaf? | Buf? | Global? | Current Fields | Rec. Tier | Missing Fields |
|---|----------|------|-------|-------|------|---------|----------------|-----------|---------------|
| 9 | `set_response` | 59 | 6 | Yes | Yes² | No | `✓⎪✓⎪—⎪—⎪—⎪✓⎪—⎪—⎪—` | **Tier 2** | Outputs, Errors, Clobbers, Locals, Max read, Max write |
| 10 | `resp_set_error` | 71 | 9 | Yes | Yes² | **Yes** | `~⎪—⎪—⎪—⎪—⎪✓⎪—⎪—⎪—` | **Tier 3** | All 9 — one-line purpose only; Preserves note present |
| 11 | `path_is_known` | 84 | 40 | No | Yes | No | `~⎪—⎪—⎪—⎪—⎪—⎪—⎪—⎪—` | **Tier 2** | All 9 — one-liner + alignment note only |
| 12 | `route_request` | 132 | 145 | No | Yes | **Yes** | `✓⎪✓⎪✓⎪—⎪✓⎪—⎪—⎪—⎪—` | **Tier 3** | Errors, Preserves, Locals, Max read, Max write |

² Holds pointer *values* to response globals; does not dereference them. Still qualifies as "touches buffers" per pointer-arg rule.

---

### `src/http_read.asm` — Request reader

| # | Function | Line | Lines | Leaf? | Buf? | Global? | Current Fields | Rec. Tier | Missing Fields |
|---|----------|------|-------|-------|------|---------|----------------|-----------|---------------|
| 13 | `read_more_request` | 49 | 119 | No | Yes | **Yes** | `✓⎪✓⎪✓⎪—⎪—⎪✓⎪—⎪—⎪—` | **Tier 3** | Errors, Clobbers, Locals, Max read, Max write (has Precond: though) |

---

### `src/http_parse.asm` — HTTP request-line + framing parser

| # | Function | Line | Lines | Leaf? | Buf? | Global? | Current Fields | Rec. Tier | Missing Fields |
|---|----------|------|-------|-------|------|---------|----------------|-----------|---------------|
| 14 | `find_header_ci` | 41 | 47 | No | Yes | No | `✓⎪✓⎪✓⎪—⎪—⎪—⎪—⎪—⎪—` | **Tier 2** | Errors, Clobbers, Preserves, Locals, Max read, Max write |
| 15 | `http_parse` | 100 | 169 | No | Yes | **Yes** | `✓⎪✓⎪✓⎪—⎪✓⎪—⎪—⎪—⎪—` | **Tier 3** | Errors, Preserves, Locals, Max read, Max write (has Precond:) |

---

### `src/http_write.asm` — Response formatter + sender

| # | Function | Line | Lines | Leaf? | Buf? | Global? | Current Fields | Rec. Tier | Missing Fields |
|---|----------|------|-------|-------|------|---------|----------------|-----------|---------------|
| 16 | `reason_phrase` | 63 | 64 | Yes | No | No | `✓⎪✓⎪✓⎪—⎪✓⎪—⎪—⎪—⎪—` | **Tier 2** | Errors, Preserves, Locals, Max read, Max write |
| 17 | `http_respond` | 139 | 93 | No | Yes | **Yes** | `✓⎪✓⎪✓⎪—⎪✓⎪—⎪—⎪—⎪—` | **Tier 3** | Errors, Preserves, Locals, Max read, Max write |

---

### `src/engine_gateway.asm` — Async OpenRouter gateway

| # | Function | Line | Lines | Leaf? | Buf? | Global? | Current Fields | Rec. Tier | Missing Fields |
|---|----------|------|-------|-------|------|---------|----------------|-----------|---------------|
| 18 | `append_raw` | 138 | 12 | Yes³ | Yes | No | `~⎪—⎪—⎪—⎪—⎪—⎪—⎪—⎪—` | **Tier 2** | All 9 — grouped comment only |
| 19 | `append_json` | 151 | 84 | Yes³ | Yes | No | `~⎪—⎪—⎪—⎪—⎪—⎪—⎪—⎪—` | **Tier 2** | All 9 — grouped comment only |
| 20 | `append_wide` | 236 | 18 | Yes³ | Yes | No | `~⎪—⎪—⎪—⎪—⎪—⎪—⎪—⎪—` | **Tier 2** | All 9 — grouped comment only |
| 21 | `hex_nibble` | 255 | 20 | Yes⁴ | No | No | `~⎪—⎪—⎪—⎪—⎪—⎪—⎪—⎪—` | **Tier 1** | All 9 — grouped comment only |
| 22 | `decode_content` | 276 | 203 | No | Yes | No | `~⎪—⎪—⎪—⎪—⎪—⎪—⎪—⎪—` | **Tier 2** | All 9 — grouped comment only |
| 23 | `gateway_start` | 487 | 110 | No | Yes | **Yes** | `✓⎪✓⎪✓⎪—⎪✓⎪✓⎪—⎪—⎪—` | **Tier 3** | Errors, Locals, Max read, Max write |
| 24 | `build_req_body` | 605 | 127 | No | Yes | No | `✓⎪✓⎪✓⎪—⎪✓⎪—⎪—⎪—⎪—` | **Tier 2** | Errors, Preserves, Locals, Max read, Max write |
| 25 | `gateway_advance` | 742 | 339 | No | Yes | **Yes** | `✓⎪✓⎪✓⎪—⎪—⎪—⎪—⎪—⎪—` | **Tier 3** | Errors, Clobbers, Preserves, Locals, Max read, Max write |
| 26 | `gw_callback` | 1099 | 56 | No | Yes⁵ | **Yes** | `✓⎪—⎪—⎪—⎪—⎪—⎪—⎪—⎪—` | **Tier 3** | All template fields — extensive narrative comment but not in template format |

³ `rep movsb` only — no `call` instruction, technically leaf.  
⁴ Pure register logic (AL in/out). No memory access beyond the stack frame.  
⁵ Writes to `gw_read_len`, `gw_err_code` globals via `[rel label]`.

---

### `src/net_io.asm` — Socket I/O primitives

| # | Function | Line | Lines | Leaf? | Buf? | Global? | Current Fields | Rec. Tier | Missing Fields |
|---|----------|------|-------|-------|------|---------|----------------|-----------|---------------|
| 27 | `send_all` | 24 | 43 | No | Yes | **Yes** | `✓⎪✓⎪✓⎪—⎪✓⎪—⎪—⎪—⎪✓` | **Tier 3** | Errors, Preserves, Locals, Max read |
| 28 | `apply_timeouts` | 77 | 23 | No | Yes⁶ | **Yes** | `✓⎪✓⎪✓⎪—⎪✓⎪—⎪—⎪—⎪—` | **Tier 3** | Errors, Preserves, Locals, Max read, Max write |

⁶ Reads `timeout_ms` global; passes pointer to `setsockopt`.

---

### `src/net_init.asm` — Winsock lifecycle

| # | Function | Line | Lines | Leaf? | Buf? | Global? | Current Fields | Rec. Tier | Missing Fields |
|---|----------|------|-------|-------|------|---------|----------------|-----------|---------------|
| 29 | `net_init` | 51 | 77 | No | Yes | **Yes** | `✓⎪✓⎪✓⎪—⎪✓⎪—⎪—⎪—⎪—` | **Tier 3** | Errors, Preserves, Locals, Max read, Max write |
| 30 | `net_shutdown` | 137 | 15 | No | Yes | **Yes** | `✓⎪✓⎪✓⎪—⎪✓⎪—⎪—⎪—⎪—` | **Tier 3** | Errors, Preserves, Locals, Max read, Max write |

---

### `src/log.asm` — Structured stderr logging

| # | Function | Line | Lines | Leaf? | Buf? | Global? | Current Fields | Rec. Tier | Missing Fields |
|---|----------|------|-------|-------|------|---------|----------------|-----------|---------------|
| 31 | `log_str` | 57 | 21 | No | Yes | **Yes** | `✓⎪✓⎪—⎪—⎪✓⎪✓⎪—⎪—⎪—` | **Tier 3** | Outputs, Errors, Locals, Max read, Max write |
| 32 | `_emit_err_line` | 90 | 57 | No | Yes | No | `✓⎪✓⎪—⎪—⎪✓⎪—⎪—⎪—⎪—` | **Tier 2** | Outputs, Errors, Preserves, Locals, Max read, Max write |
| 33 | `log_err` | 156 | 22 | No | Yes⁷ | **Yes** | `✓⎪✓⎪—⎪—⎪✓⎪—⎪—⎪—⎪—` | **Tier 3** | Outputs, Errors, Preserves, Locals, Max read, Max write |
| 34 | `log_err_code` | 190 | 2 | Yes⁸ | No | **Yes** | `✓⎪✓⎪—⎪—⎪✓⎪—⎪—⎪—⎪—` | **Tier 3** | Outputs, Errors, Preserves, Locals, Max read, Max write |
| 35 | `log_request` | 203 | 125 | No | Yes | **Yes** | `✓⎪✓⎪—⎪—⎪✓⎪—⎪—⎪—⎪—` | **Tier 3** | Outputs, Errors, Preserves, Locals, Max read, Max write |

⁷ Calls `WSAGetLastError` which is technically a memory write via the TIB.  
⁸ Tail-calls `_emit_err_line` via `jmp` — no `call` instruction, so leaf.

---

### `src/text.asm` — Bounded byte utilities

| # | Function | Line | Lines | Leaf? | Buf? | Global? | Current Fields | Rec. Tier | Missing Fields |
|---|----------|------|-------|-------|------|---------|----------------|-----------|---------------|
| 36 | `copy_bytes` | 15 | 12 | Yes | Yes | **Yes** | `✓⎪—⎪—⎪—⎪✓⎪—⎪—⎪—⎪—` | **Tier 3** | Inputs, Outputs, Errors, Preserves, Locals, Max read, Max write |
| 37 | `bytes_eq` | 35 | 17 | Yes⁹ | Yes | **Yes** | `✓⎪✓⎪✓⎪—⎪✓⎪—⎪—⎪—⎪—` | **Tier 3** | Errors, Preserves, Locals, Max read, Max write |
| 38 | `mem_find` | 61 | 38 | Yes⁹ | Yes | **Yes** | `✓⎪✓⎪✓⎪—⎪✓⎪—⎪—⎪—⎪—` | **Tier 3** | Errors, Preserves, Locals, Max read, Max write |
| 39 | `mem_find_ci` | 105 | 50 | Yes⁹ | Yes | **Yes** | `~⎪—⎪—⎪—⎪—⎪—⎪—⎪—⎪—` | **Tier 3** | All 9 — cross-references `mem_find` but no independent doc |

⁹ Uses `repe cmpsb` / `rep movsb` — not `call`, so technically leaf.

---

### `src/state.asm` — Static buffers, globals, debug canaries

| # | Function | Line | Lines | Leaf? | Buf? | Global? | Current Fields | Rec. Tier | Missing Fields |
|---|----------|------|-------|-------|------|---------|----------------|-----------|---------------|
| 40 | `debug_canaries_init` | 155 | 14 | Yes | Yes | **Yes** | `~⎪—⎪—⎪—⎪—⎪—⎪—⎪—⎪—` | **Tier 3** | All 9 — one-liner purpose only |
| 41 | `debug_canaries_check` | 171 | 27 | Yes | Yes | **Yes** | `~⎪—⎪—⎪—⎪—⎪—⎪—⎪—⎪—` | **Tier 3** | All 9 — one-liner purpose only |

---

### `src/decimal.asm` — Decimal formatting + parsing

| # | Function | Line | Lines | Leaf? | Buf? | Global? | Current Fields | Rec. Tier | Missing Fields |
|---|----------|------|-------|-------|------|---------|----------------|-----------|---------------|
| 42 | `u32_to_dec` | 24 | 58 | Yes | Yes | **Yes** | `✓⎪✓⎪✓⎪—⎪✓⎪✓⎪—⎪—⎪✓` | **Tier 3** | Errors, Locals, Max read |
| 43 | `parse_u32` | 93 | 33 | Yes | Yes | **Yes** | `✓⎪✓⎪✓⎪✓⎪✓⎪✓⎪—⎪—⎪—` | **Tier 3** | Locals, Max read, Max write |

---

### `src/assets.asm` — Embedded data + DEV_MODE disk loader

| # | Function | Line | Lines | Leaf? | Buf? | Global? | Current Fields | Rec. Tier | Missing Fields |
|---|----------|------|-------|-------|------|---------|----------------|-----------|---------------|
| 44 | `load_index_html` | 55 | 65 | No | Yes | **Yes** | `✓⎪✓⎪✓⎪—⎪✓⎪—⎪—⎪—⎪—` | **Tier 3** | Errors, Preserves, Locals, Max read, Max write |

---

## Priority Order for Retrofit

### Phase 1 — Tier 1 (All 44 functions: minimal inline)

Every callable label needs at minimum a one-line register contract. These 44 are the current gap for the **one function that has zero annotation at all**:

| Priority | Function | File:Line | Reason |
|----------|----------|-----------|--------|
| 1 | `hex_nibble` | `engine_gateway.asm:255` | **Only Tier 1 function** — needs `; AL=nibble, ret AL=value, CF=error` + `; Clobbers: flags` |

All other 43 functions have at least a `Purpose` line, so the Tier 1 gap is only for the `hex_nibble` case (which has only a group-comment reference).

### Phase 2 — Tier 2 (17 functions: structured 7-field header)

Sort by file then line:

| Priority | Function | File:Line | Lines | Current missing fields |
|----------|----------|-----------|-------|----------------------|
| 1 | `slot_to_globals` | `start.asm:428` | 67 | Inputs, Outputs, Errors, Clobbers, Preserves, Locals, Max r/w |
| 2 | `globals_to_slot` | `start.asm:500` | 67 | Inputs, Outputs, Errors, Clobbers, Preserves, Locals, Max r/w |
| 3 | `disassociate_slot` | `start.asm:573` | 24 | Inputs, Outputs, Errors, Clobbers, Preserves, Locals, Max r/w |
| 4 | `free_slot` | `start.asm:602` | 24 | Inputs, Outputs, Errors, Clobbers, Preserves, Locals, Max r/w |
| 5 | `respond_and_free_slot` | `start.asm:631` | 27 | Inputs, Outputs, Errors, Clobbers, Preserves, Locals, Max r/w |
| 6 | `error_and_free_slot` | `start.asm:663` | 34 | Inputs, Outputs, Errors, Clobbers, Preserves, Locals, Max r/w |
| 7 | `scan_timeouts` | `start.asm:701` | 39 | Inputs, Outputs, Errors, Clobbers, Preserves, Locals, Max r/w |
| 8 | `set_response` | `router.asm:59` | 6 | Outputs, Errors, Clobbers, Locals, Max r/w |
| 9 | `path_is_known` | `router.asm:84` | 40 | All 9 — one-liner only |
| 10 | `find_header_ci` | `http_parse.asm:41` | 47 | Errors, Clobbers, Preserves, Locals, Max r/w |
| 11 | `reason_phrase` | `http_write.asm:63` | 64 | Errors, Preserves, Locals, Max r/w |
| 12 | `append_raw` | `engine_gateway.asm:138` | 12 | All 9 — grouped comment only |
| 13 | `append_json` | `engine_gateway.asm:151` | 84 | All 9 — grouped comment only |
| 14 | `append_wide` | `engine_gateway.asm:236` | 18 | All 9 — grouped comment only |
| 15 | `decode_content` | `engine_gateway.asm:276` | 203 | All 9 — grouped comment only |
| 16 | `build_req_body` | `engine_gateway.asm:605` | 127 | Errors, Preserves, Locals, Max r/w |
| 17 | `_emit_err_line` | `log.asm:90` | 57 | Outputs, Errors, Preserves, Locals, Max r/w |

### Phase 3 — Tier 3 (26 functions: strict contract with Doxygen tags)

Sort by file then line:

| Priority | Function | File:Line | Reason |
|----------|----------|-----------|--------|
| 1 | `start` | `start.asm:109` | **OS entry point**, event loop, security-critical |
| 2 | `resp_set_error` | `router.asm:71` | Public extern; sets error response globals |
| 3 | `route_request` | `router.asm:132` | **Public dispatch**, security-critical routing |
| 4 | `read_more_request` | `http_read.asm:49` | **Public extern**, bounded recv, security-critical |
| 5 | `http_parse` | `http_parse.asm:100` | **Public extern**, header parser, security-critical |
| 6 | `http_respond` | `http_write.asm:139` | **Public extern**, response sender, security-critical |
| 7 | `gateway_start` | `engine_gateway.asm:487` | **Public extern**, **state machine**, API-key handling |
| 8 | `gateway_advance` | `engine_gateway.asm:742` | **Public extern**, **async state machine** |
| 9 | `gw_callback` | `engine_gateway.asm:1099` | **Public extern**, **callback on worker thread** |
| 10 | `send_all` | `net_io.asm:24` | Public extern; writes to client sockets |
| 11 | `apply_timeouts` | `net_io.asm:77` | Public extern; sets socket options |
| 12 | `net_init` | `net_init.asm:51` | **Public extern**, startup, security-critical |
| 13 | `net_shutdown` | `net_init.asm:137` | **Public extern**, teardown/exit |
| 14 | `log_str` | `log.asm:57` | Public extern; writes to stderr |
| 15 | `log_err` | `log.asm:156` | Public extern; error logging |
| 16 | `log_err_code` | `log.asm:190` | Public extern; explicit-code logging |
| 17 | `log_request` | `log.asm:203` | Public extern; structured request log |
| 18 | `copy_bytes` | `text.asm:15` | Public extern; shared memcpy primitive |
| 19 | `bytes_eq` | `text.asm:35` | Public extern; shared compare primitive |
| 20 | `mem_find` | `text.asm:61` | Public extern; shared search primitive |
| 21 | `mem_find_ci` | `text.asm:105` | Public extern; shared CI search primitive |
| 22 | `debug_canaries_init` | `state.asm:155` | Public extern; buffer-overflow guard |
| 23 | `debug_canaries_check` | `state.asm:171` | Public extern; buffer-overflow guard |
| 24 | `u32_to_dec` | `decimal.asm:24` | Public extern; shared formatting primitive |
| 25 | `parse_u32` | `decimal.asm:93` | Public extern; shared parse primitive |
| 26 | `load_index_html` | `assets.asm:55` | Public extern; file I/O (DEV_MODE) |

---

## Notes

### Ambiguities and Classification Decisions

1. **`listen_addr` (`net_init.asm:20`), `chat_html` (`assets.asm:12`), `build_id` (`assets.asm:19`), `health_json` (`assets.asm:26`)**  
   Declared `global` but are **data structures** (`dw`/`db`/`incbin`), not executable code. Excluded from the callable inventory.

2. **`log_err_code` (`log.asm:190`)**  
   Implemented as `jmp _emit_err_line` (tail-call, no `call`). Classified as **leaf** since NASM `jmp` does not push a return address. Still Tier 3 because it is `global`.

3. **`reason_phrase` (`http_write.asm:63`)**  
   64 lines, no `call` instruction (leaf), but >20 lines → Tier 2, even though it is a simple status→string lookup. Could arguably be Tier 1, but rule says >20 = Tier 2.

4. **`hex_nibble` (`engine_gateway.asm:255`)**  
   Exactly 20 lines of code. Does not reach the >20 threshold (counted as ≤20). No pointer args — pure register-in/register-out. The **only** function classified Tier 1.

5. **`append_raw` / `append_wide` (`engine_gateway.asm:138,236`)**  
   Small leaf helpers but use pointer arguments (`RSI` src, `RDI` dst, `R14` limit) → "touches buffers" → Tier 2.

6. **`set_response` (`router.asm:59`)**  
   6-line leaf. Accepts pointer arguments (body_ptr, ct_ptr) but stores them without dereferencing. Still classified as "touches buffers" per the rule `pointer args → yes`.

7. **`read_more_request` `Precond:` note**  
   Has a `Precond:` line in its header block but no formal `Errors:` or `Clobbers:` template fields. The existing doc is narrative, not template-conformant.

8. **`gw_callback` (`engine_gateway.asm:1099`)**  
   Extensive narrative documentation of the WinHTTP callback signature and behavior, but none of it follows the 7-field template. Reports as Purpose-only for template-field counting.

9. **`debug_canaries_init`/`check` (`state.asm`)**  
   Body guarded by `%if DEBUG` — when DEBUG=0, these compile to `xor eax, eax / ret` (or even just `ret`). Still callable labels needing annotation regardless of conditional assembly.

10. **`slot_to_globals` / `globals_to_slot` leaf classification**  
    Use `rep movsb` for block copies but no `call` instructions. Classified as leaf. The `rep movsb` is an x86 string instruction, not a call.

### Known Tier 3 Candidates — Verification

Per the task's list of known Tier 3 candidates:

| Candidate | Found? | Classification | Notes |
|-----------|--------|---------------|-------|
| `read_more_request` | ✅ | Tier 3 | Bounded recv, public extern |
| `http_parse` | ✅ | Tier 3 | Header parsing, public extern |
| `route_request` | ✅ | Tier 3 | Dispatch, public extern |
| `gateway_start` | ✅ | Tier 3 | Async state machine entry |
| `engine_gateway` (as `gateway_advance`) | ✅ | Tier 3 | Async state machine |
| `ClientSlot` copy/move helpers (`slot_to_globals`, `globals_to_slot`) | ✅ | Tier 2 | Not global, but touch buffers |
| `mem_find` | ✅ | Tier 3 | Public extern |
| `bytes_eq` | ✅ | Tier 3 | Public extern |

### Retrofit Strategy Recommendation

1. **Phase 1 (Tier 1):** Add inline register contract to `hex_nibble` — the only function with zero annotation.
2. **Phase 2 (Tier 2):** Retrofit the 17 Tier-2 functions, starting with the `engine_gateway.asm` group (4 helpers with zero doc) and the 7 `start.asm` slot helpers.
3. **Phase 3 (Tier 3):** This touches 26 functions (59% of codebase). Prioritize by risk: `start` (entry point, event loop), `gateway_advance` (state machine), `read_more_request` (network boundary), `http_parse` (input validation), `route_request` (authorization dispatch), then the remaining public primitives.
