# Annotation Standard

This document is the authoritative reference for annotating routines in the
`assembly/` project. Read it once; apply it forever. The research justification
lives in `docs/assembly-annotation-research.md` — cite that for *why*, read this
for *how*.

The companion file `include/win64.inc` defines the project's call-frame macros
and the 7-field contract-comment template this standard extends. The two files
are meant to be read together: win64.inc is the short reference card pasted at
the top of every routine, this file is the rulebook that picks the tier,
defines each field, and lists the forbidden variants.

---

## 1. Purpose & scope

NASM has no native type-hint or docstring feature; the assembler enforces
layout (`struc`, `equ`, `%define`, `default rel`) but never semantics. This
standard codifies three things that together give the safety and toolability
that type hints and docstrings give in higher-level languages:

1. A **tiered comment convention** (Tier 1/2/3) calibrated to routine size and
   risk.
2. **NASM-enforced syntax rules** (the project's effective "type system"):
   `default rel`, the `[label + reg]` ban, `struc`/`%define CAP_*`, and the
   `PROLOGUE`/`EPILOGUE`/`SHADOW_ONLY`/`SHADOW_FREE` macros from
   `include/win64.inc`.
3. **Macro discipline** that mechanically enforces Win64 ABI shadow space,
   16-byte alignment, and callee-save register handling.

Scope: every `.asm` and `.inc` file under `src/` and `include/`. Every label
that is the target of a `call` instruction, or declared `global`, falls under
this standard. Local control-flow labels (`.loop`, `.done`, etc.) are exempt.

This is the standard distilled from research §5 (three example styles),
§6 (recommended standard), §7 (rollout plan), and §8.6 (glossary). The research
doc is the evidence base; this doc is the contract.

---

## 2. The three tiers

Pick the tier using the decision flowchart in §7. Then copy the matching
template verbatim. Templates are copy-pasteable — do not re-invent field names,
order, or capitalization. Existing field order matches `win64.inc:39-47`.

### 2.1 Tier 1 — Minimal inline (all callable labels including leaves)

**When to use:** every `label:` reachable by `call` or declared `global`. This
is the floor. A leaf helper under ~20 lines that does not dereference a pointer
past its own arguments stays at Tier 1.

**Rule:** one purpose line, one inputs line that names every register argument
and the return register. The Clobbers/Preserves line may be elided only when
the routine is provably a pure leaf that touches nothing but volatile
registers; otherwise state it.

**Template:**

```asm
; <one-line purpose>
; RCX=..., RDX=..., R8=..., R9=..., ret RAX=...
; Clobbers: volatile   Preserves: nonvolatile     ; (elide only if obvious leaf)
label:
```

**Worked example — `set_response` (router.asm:55-64), shown as-is then to-Tier1:**

Current state (router.asm:55-58), already close to Tier 1:

```asm
; ---------------------------------------------------------------------------
; set_response - fill the four response globals from register args. Leaf.
; Inputs:  RCX=body_ptr, RDX=body_len, R8=ct_ptr, R9=ct_len. Preserves RAX.
; ---------------------------------------------------------------------------
set_response:
    mov     [resp_body_ptr], rcx
    mov     [resp_body_len], rdx
    mov     [resp_ct_ptr], r8
    mov     [resp_ct_len], r9
    ret
```

Canonical Tier 1 (one purpose line, one inputs line, clobbers/preserves
explicit):

```asm
; set_response - store (body_ptr,len) and (ct_ptr,len) into resp_* globals
; RCX=body_ptr, RDX=body_len, R8=ct_ptr, R9=ct_len, ret RAX=unmodified
; Clobbers: none   Preserves: all (including RAX)
set_response:
    mov     [resp_body_ptr], rcx
    mov     [resp_body_len], rdx
    mov     [resp_ct_ptr], r8
    mov     [resp_ct_len], r9
    ret
```

Tier 1 is grep-searchable: `;.*ret RAX=` finds every callable leaf in the tree.

---

### 2.2 Tier 2 — Structured header (non-leaf, >20 lines, or touches buffers)

**When to use:** any of the following — the routine contains a `call`
instruction (non-leaf), the body exceeds ~20 lines, or the routine reads or
writes memory through a pointer argument or a global buffer.

**Rule:** paste the 7-field template from `win64.inc:39-47` verbatim, in that
exact field order. Add the optional `; Precond:` line when the caller must
guarantee something (alignment, non-null, buffer validity, state-machine
state). The dash form `; fn_name -` on the first line carries the inline
purpose and is mandatory — it is what `grep '; .* -'` keys on.

**Template (verbatim from `win64.inc:39-47` plus the optional Precond line):**

```asm
; ---------------------------------------------------------------------------
; fn_name - <one-line purpose>
; Purpose:        <longer paragraph if the one-liner is insufficient>
; Inputs:         RCX=..., RDX=..., R8=..., R9=...
; Outputs:        RAX=...
; Errors:         RAX=<sentinel> on failure
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11   (volatile)
; Preserves:      RBX,RBP,RDI,RSI,R12-R15      (nonvolatile)
; Locals:         N bytes
; Max read:       M bytes   Max write: W bytes
; Precond:        <optional; omit when none>
; ---------------------------------------------------------------------------
```

**Worked example — `read_more_request` (http_read.asm:28-49), shown as the full
Tier 2 header it already largely obeys:**

```asm
; ---------------------------------------------------------------------------
; read_more_request - one non-blocking recv into recv_buf, check for completion.
; Purpose:        Incremental HTTP request reader. Does ONE recv, accumulates
;                 into recv_buf, detects CRLFCRLF header terminator, enforces
;                 CAP_REQUEST and CAP_HEADERS. Never retries with Sleep.
; Inputs:         RCX = client socket (SOCKET)
; Precond:        req_used, req_header_end reflect current accumulation
; Outputs:        RAX = 0 complete; 1 more needed; 2 WSAEWOULDBLOCK; >=400 error
; Errors:         RAX >= 400; resp_* set by set_response on the error path
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11
; Preserves:      RBX,RBP,RDI,RSI,R12-R15
; Locals:         32 (shadow only; rbx,rsi saved via push, frame via PROLOGUE 0)
; Max read:       (CAP_REQUEST - req_used) bytes via recv into [recv_buf+req_used]
; Max write:      (CAP_REQUEST - req_used) bytes into recv_buf
; ---------------------------------------------------------------------------
global read_more_request
read_more_request:
    push    rbp
    mov     rbp, rsp
    push    rbx                      ; socket
    push    rsi                      ; n (bytes read)
    sub     rsp, 32                  ; shadow; 3 pushes keep RSP 16-aligned at CALL
    mov     rbx, rcx                 ; save socket
    ; ...
```

Every field is present and in canonical order. `Max read` and `Max write`
cite the `CAP_REQUEST` bound — that is the whole point of having the field.

---

### 2.3 Tier 3 — Strict contract (public externs, state machines, security-critical)

**When to use:** the routine is exported (`global`), is reachable from another
module, drives a state machine, parses untrusted input, or sits on a
security-critical path. The current Tier 3 set: `read_more_request`,
`http_parse`, `route_request`, `gateway_start`, `gateway_advance`,
`mem_find`, `bytes_eq`, and every `copy_slot_*` / `copy_globals_*` helper in
`start.asm`. Anything that touches a `ClientSlot` (see `config.inc:72-93`)
counts.

**Rule:** Tier 2 in full, *plus* Doxygen `@param[in]`/`@param[out]` tags for
each register argument and return value, *plus* the optional fields that the
situation warrants:

- `Stack:` — shadow + stack-arg + locals layout, for routines with stack args
  beyond arg 5 or unusual frame math.
- `Modified:` — edk2 alias for `Clobbers`. Include it when a future Doxygen
  pass would otherwise miss the clobber list; harmless when redundant.
- `Initial inputs to registers:` — SymCrypt-style mapping from logical names
  to entry registers. Mandatory when the routine immediately shuffles args
  into nonvolatile registers (e.g., `RCX → RBX`, `RDX → R12`).
- `Register assignments` — phase table for state machines and multi-pass
  algorithms. One line per phase naming every live register's role.

**Template (Tier 3, complete):**

```asm
; ===========================================================================
; fn_name - <one-line purpose>
; Purpose:        <paragraph>
; @param[in]      RCX - <type> <name>: <meaning>
; @param[in]      RDX - <type> <name>: <meaning>
; @param[in]      R8  - <type> <name>: <meaning>
; @param[in]      R9  - <type> <name>: <meaning>
; @param[out]     RAX - <type> <meaning>
; Inputs:         RCX=..., RDX=..., R8=..., R9=...
; Outputs:        RAX=...
; Errors:         RAX=<sentinel> on failure
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11
; Preserves:      RBX,RBP,RDI,RSI,R12-R15
; Locals:         N bytes
; Max read:       M bytes
; Max write:      W bytes
; Precond:        <caller obligations>
; Stack:          SHADOW 32 + <stack args> + <locals>, RSP 16-aligned at CALL
; Modified:       RAX,RCX,RDX,R8,R9,R10,R11
; Initial inputs to registers: <logical>->RCX, <logical>->RDX, ...
; Register assignments:
;   phase1: RCX=cursor, RDX=src, R8=remaining
;   phase2: RAX=result, R9=tmp
; ===========================================================================
```

The `===` separator (79 chars) signals Tier 3 and distinguishes it from the
Tier 2 `---` separator.

**Worked example — `route_request` (router.asm), shown at Tier 3 as live-retrofitted:**

```asm
; ===========================================================================
; route_request - dispatch parsed request to matching response handler
; Purpose:        Reads req_* globals (populated by http_read/http_parse),
;                 matches method + path, sets resp_* globals, and either
;                 returns synchronously (static routes) or kicks the async
;                 gateway for POST /chat. Single entry point for all HTTP
;                 response selection.
; @param[in]      none (operates on req_* globals)
; @param[out]     RAX - u16 HTTP status (200, 404, 405, 411, 503)
; Inputs:         none; operates on req_method_ptr, req_method_len,
;                 req_path_ptr, req_path_len, req_has_cl,
;                 req_content_length, req_header_end, recv_buf globals
; Outputs:        RAX = HTTP status; resp_body_ptr/len, resp_ct_ptr/len set
; Errors:         RAX in {404,405,411,503}; err_body set via set_response
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11
; Preserves:      RBX,RBP,RDI,RSI,R12-R15
; Locals:         32 (shadow only; push rbp + sub 32)
; Max read:       req_path_len bytes from [req_path_ptr], req_method_len from
;                 [req_method_ptr] (via bytes_eq); CAP_CHAT_BODY from
;                 [recv_buf + req_header_end] (forwarded to gateway_start)
; Max write:      0 (writes go to resp_* globals, not caller buffers)
; Precond:        http_parse complete; req_path_len, req_method_len valid;
;                 req_header_end set; recv_buf populated
; Stack:          SHADOW 32, single CALL to gateway_start at aligned RSP
; Modified:       RAX,RCX,RDX,R8,R9,R10,R11
; Initial inputs to registers: none (operates on globals)
; Register assignments:
;   match_phase:  RCX=method_ptr, RDX=method_len, R8=candidate, R9=cand_len
;                 (for bytes_eq); then RCX=path_ptr, RDX=path_len
;   route_phase:  RCX=body_ptr, RDX=body_len, R8=ct_ptr, R9=ct_len
;                 (for set_response); or RCX=body_ptr, RDX=body_len
;                 for gateway_start; RAX=HTTP status for error paths
; ===========================================================================
```

Tier 3 is verbose on purpose. The cost is paid once for ~10-15 public APIs and
bought back every time a contributor has to reason about a state machine or a
buffer boundary without re-reading the implementation.

---

## 3. Field definitions

Nine base fields are required at Tier 2+. Six Tier 3 extensions are optional
and added only when they carry information the base fields cannot.

| Field                  | Required tier | Definition                                                          | Example                                                            |
|------------------------|---------------|---------------------------------------------------------------------|--------------------------------------------------------------------|
| Purpose                | 2, 3          | One-line or short paragraph; what the routine does, not how.        | `Copies R8 bytes non-overlapping.`                                 |
| Inputs                 | 2, 3          | Every register arg (RCX,RDX,R8,R9) plus stack args; include types.  | `RCX=u8* dst, RDX=u8* src, R8=usize len 0..CAP_REQUEST`            |
| Outputs                | 2, 3          | Return register (usually RAX) and any globals the routine writes.   | `RAX=HTTP status; resp_* globals set`                              |
| Errors                 | 2, 3          | Sentinel value, CF state, or HTTP code; what failure looks like.    | `RAX=0 on error, CF may be set` or `RAX>=400`                      |
| Clobbers               | 2, 3          | Set of volatile registers destroyed; full list, never "etc.".       | `RAX,RCX,RDX,R8,R9,R10,R11`                                        |
| Preserves              | 2, 3          | Nonvolatile registers guaranteed intact on return.                   | `RBX,RBP,RDI,RSI,R12-R15` or `non-volatile`                        |
| Locals                 | 2, 3          | Stack bytes after PROLOGUE rounding; cite SHADOW explicitly.        | `24 (32 shadow + 24 locals, PROLOGUE rounds to 64)`                |
| Max read               | 2, 3          | Upper bound on bytes read through any pointer; cite the CAP_*.      | `R8 bytes from [RDX]` or `(CAP_REQUEST - req_used) bytes`          |
| Max write              | 2, 3          | Upper bound on bytes written through any pointer; cite the CAP_*.   | `R8 bytes to [RCX]`                                                |
| Precond (optional)     | 2, 3          | Caller obligations: validity, alignment, non-null, state, no overlap.| `RCX,RDX valid for R8 bytes; R8<=CAP_REQUEST; RSP 16-aligned`      |
| Stack (optional)       | 3             | Shadow + stack-arg + locals layout; alignment state.                | `SHADOW 32, [rsp+20h]=arg5, [rbp-8..]=locals`                      |
| @param[in/out]         | 3             | Doxygen version of Inputs/Outputs for HTML doc generation.          | `@param[in] RCX - u8* dst: destination buffer`                     |
| Modified (optional)    | 3             | edk2 alias for Clobbers; include for forward toolability.           | `Modified: RAX,RCX,RDX,R8,R9,R10,R11`                              |
| Initial inputs         | 3             | Logical-to-register map; mandatory when args are shuffled on entry. | `dst->RCX, src->RDX, len->R8`                                      |
| Register assignments   | 3             | Phase table; one line per phase naming every live register's role.  | `phase1: RCX=cursor, RDX=src, R8=remaining`                        |

When a field has nothing to say, write `none` (not blank, not `n/a`). Blank
fields look like the author forgot; `none` is a positive claim. For Clobbers
specifically, `none` means the routine touches nothing outside its own
arguments — this is the FDOS `CHG: -` pattern adapted to our field name (see
research §8.6).

---

## 4. NASM-enforced syntax (the "type system" layer)

The assembler cannot check register contracts, but it can check structural
rules. These rules are the project's effective type system. Violate one and
the build breaks or the binary crashes at runtime under ASLR.

- **`default rel` mandatory at the top of every `.asm` file.** See
  `win64.inc:26-34`. RIP-relative addressing is ASLR-safe and needs no
  relocation entry.
- **Never write `[label + reg]`.** Under `default rel`, that form silently
  becomes absolute and GoLink does not emit the imm64 fixup under
  `/dynamicbase`, so the pointer is wrong at runtime. Always two-step:
  ```asm
  lea rax, [rel label]   ; RIP-relative address of label
  add rax, <reg>         ; then add any register offset
  ```
  See `win64.inc:28-34`. Plain `[label]` with no index is fine under
  `default rel`.
- **Use `struc`/`endstruc`/`resq`/`resb`/`resd` for every structure.** The
  exemplar is `ClientSlot` in `config.inc:72-93`. Pair every struc with a
  size constant: `CLIENT_SLOT_SIZE equ ClientSlot_size` (`config.inc:95`).
- **Use `equ` for byte offsets and sizes derived from struc, never magic
  numbers.** Field offsets come from `.field` labels; size comes from
  `StrucName_size`.
- **Use `%define CAP_*` for every buffer bound.** All capacities live in
  `config.inc:15-26` (`CAP_REQUEST`, `CAP_HEADERS`, `CAP_CHAT_BODY`, ...).
  Refer to these by name in `Max read`/`Max write` lines, never the literal
  `8192`.
- **Use `PROLOGUE`/`EPILOGUE` for non-leaf routines.** Defined at
  `win64.inc:64-76`. They emit the `push rbp` / `mov rbp,rsp` frame, reserve
  SHADOW + padded(locals) bytes rounded up to 16, and restore on exit. Never
  hand-roll `sub rsp, N` in a non-leaf without using the macro — the rounding
  at `win64.inc:68` is what keeps RSP 16-aligned at every CALL.
- **Use `SHADOW_ONLY`/`SHADOW_FREE` at leaf call sites that need shadow space
  without a full frame.** Defined at `win64.inc:80-85`. The macros pair 1:1;
  every `SHADOW_ONLY` must have a matching `SHADOW_FREE` on every return path.
- **Reserve 32 bytes of shadow space at every CALL site, even for zero-arg
  callees.** Win64 rule; documented at `win64.inc:8-9`. The macros above
  enforce this when used.
- **RSP must be 16-byte aligned at the point of every CALL instruction.** See
  `win64.inc:12-14`. At callee entry RSP ≡ 8 mod 16 because CALL pushed the
  return address; the PROLOGUE `sub rsp, %%total` (rounded to 16) restores
  alignment.
- **Volatile set is `RAX,RCX,RDX,R8,R9,R10,R11,XMM0-5`; nonvolatile set is
  `RBX,RBP,RDI,RSI,R12-R15,XMM6-15`.** Any nonvolatile you touch must be
  saved and restored. See `win64.inc:15-18`.

These rules are non-negotiable. The comment standard documents intent; these
rules document reality. When the two disagree, reality wins and the comment
gets fixed in the same commit.

---

## 5. Accepted keyword glossary

Project-standard comment keywords vs. forbidden variants. Distilled from
research §8.6 to the conventions this project actually uses.

**Use (project standard):**

| Keyword                          | Meaning                                                |
|----------------------------------|--------------------------------------------------------|
| `Inputs:`                        | Register + stack arguments, with types.                |
| `Outputs:`                       | Return register and globals written.                   |
| `Errors:`                        | Sentinels, CF state, HTTP codes.                       |
| `Clobbers:`                      | Volatile registers destroyed.                          |
| `Preserves:`                     | Nonvolatile registers guaranteed intact.               |
| `Locals:`                        | Stack bytes after PROLOGUE rounding.                   |
| `Max read:`                      | Upper bound on bytes read through pointers.            |
| `Max write:`                     | Upper bound on bytes written through pointers.         |
| `Precond:`                       | Caller obligations (optional, Tier 2+).                |
| `@param[in]` / `@param[out]`     | Doxygen toolable arg/return tags (Tier 3).             |
| `Modified:`                      | edk2 alias for Clobbers (Tier 3, forward toolability). |
| `Initial inputs to registers:`   | Logical-to-register map (Tier 3, entry shuffle).       |
| `Register assignments`           | Phase table (Tier 3, state machines).                  |
| `Stack:`                         | Shadow + args + locals layout (Tier 3).                |

**Forbidden in this project:**

- **HeavyThing lowercase `prolog`/`epilog`** — name-collides with the
  uppercase `PROLOGUE`/`EPILOGUE` macros at `win64.inc:64-76`. Use the
  uppercase macros only.
- **AT&T `%reg` syntax** (`%rax`, `%rcx`) — GAS-specific, not NASM. Never
  appears in this project.
- **C block comments `/* ... */`** — NASM uses `;`. The Linux kernel style
  (research §3.3) is interesting evidence but not portable here.
- **`sal.h` SAL macros** (`_In_reads_bytes_`, `_Out_writes_bytes_`) — no NASM
  toolchain parses them. Keep SAL as *inspiration* for the meaning of
  `Max read`/`Max write`; do not paste the macros into source.
- **Double-semicolon `;; ` convention** (dotnet/runtime style, research §3.3)
  — we use single `;`. Doubling adds no information and breaks grep parity
  with the existing tree.
- **`IN:`/`OUT:`/`CHG:`/`TRASHES:` uppercase variants from other projects** —
  research §8.6 lists them as evidence; we standardize on `Inputs:`/`Outputs:`
  /`Clobbers:` (already in `win64.inc:39-47`). Do not mix vocabularies.

---

## 6. Doxygen integration (optional Tier 3 toolability)

The `@param[in]`/`@param[out]` tags in Tier 3 are forward-compatible with
Doxygen, the only tool proven to parse `.asm`/`.inc` files in a production
codebase (tianocore/edk2, research §3.3). The configuration that works:

```
FILE_PATTERNS     = *.asm *.inc
EXTENSION_MAPPING = inc=C asm=C
EXTRACT_ALL       = YES
```

With that mapping, Doxygen treats `;` lines as C comments, picks up
`@param`/`@return`/`Modified:` tags, and emits HTML for free — without
breaking `grep` over the source. The tags are additive: they duplicate
information already in `Inputs:`/`Outputs:`/`Clobbers:`. That redundancy is
the price of toolability.

**No `Doxyfile` is added in this phase.** Tooling is deferred (research §7.2).
The tags go into Tier 3 headers now so that a future `scripts/check-contracts.ps1`
or a vendored `nasm-lint` (research §7.2, §7.6) can consume them without a
second touch of every public routine. Treat `@param` lines as documentation
that happens to be machine-readable, not as a separate system.

---

## 7. Decision flowchart

Use this tree when annotating a new function. It always terminates at a
concrete tier.

```
Is the label callable (target of `call` or declared `global`)?
├─ No  → no annotation required (it is a local control-flow label like .loop)
└─ Yes → Tier 1 minimum required.
   Is it non-leaf (contains `call`) OR >20 lines OR touches buffers via pointers?
   ├─ No  → Tier 1 is enough.
   └─ Yes → Tier 2 required.
      Is it `global`, a state machine, or security-critical
      (recv/parse/route/gateway/ClientSlot)?
      ├─ No  → Tier 2 is enough.
      └─ Yes → Tier 3 required.
```

Borderline cases, resolved:

- A leaf helper under 20 lines that dereferences a pointer argument
  (`mem_find`, `bytes_eq`) → **Tier 2** (the "touches buffers" rule wins over
  the "leaf" rule; bounds matter).
- A 5-line `global` routine that only sets globals (`resp_set_error`) →
  **Tier 1** is acceptable today, **Tier 3** is required the moment another
  module starts depending on its exact register contract.
- A non-leaf routine under 20 lines → **Tier 2** (the "non-leaf" rule wins
  over the length rule).

---

## 8. Worked examples (same function at all three tiers)

The same bounded copy rendered at each tier. The function: copy `R8` bytes
from `[RDX]` to `[RCX]`, return original `RCX` in `RAX`, no overlap, caller
guarantees `R8 <= CAP_REQUEST`. Adapted from research §5.1, §5.2, §5.3 and
made Win64-accurate with project macros.

### 8.1 Tier 1 — minimal inline

```asm
default rel
section .text
; mem_copy - copy R8 bytes from [RDX] to [RCX], return original RCX in RAX
; RCX=dst u8*, RDX=src u8*, R8=len usize, ret RAX=dst
; Clobbers: RAX,RCX,RDX,R8,R9,R10,R11   Preserves: nonvolatile
mem_copy:
    mov     rax, rcx                 ; save dst for return
    mov     r9,  rdi                 ; (illustrative; real impl uses rep movsb)
    mov     rcx, r8
    rep     movsb
    ret
```

### 8.2 Tier 2 — structured header

```asm
default rel
section .text
; ---------------------------------------------------------------------------
; mem_copy - bounded non-overlapping copy
; Purpose:        Copies R8 bytes from [RDX] to [RCX]. Caller must guarantee
;                 R8 <= CAP_REQUEST and no overlap.
; Inputs:         RCX=u8* dst, RDX=u8* src, R8=usize len (0..CAP_REQUEST)
; Outputs:        RAX=u8* original dst (RCX on entry)
; Errors:         none
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11
; Preserves:      RBX,RBP,RDI,RSI,R12-R15
; Locals:         0 (leaf, SHADOW not required: no CALL)
; Max read:       R8 bytes from [RDX]
; Max write:      R8 bytes to [RCX]
; Precond:        R8 <= CAP_REQUEST; [RCX,R8) and [RDX,R8) do not overlap
; ---------------------------------------------------------------------------
mem_copy:
    mov     rax, rcx                 ; save dst for return
    mov     rsi, rdx                 ; src
    mov     rdi, rcx                 ; dst
    mov     rcx, r8                  ; count
    rep     movsb
    ret
```

### 8.3 Tier 3 — strict contract

```asm
default rel
section .text
; ===========================================================================
; mem_copy - bounded non-overlapping copy (public, security-critical buffer op)
; Purpose:        Copies R8 bytes from [RDX] to [RCX] using rep movsb. Used
;                 for request-buffer moves inside ClientSlot copy helpers.
; @param[in]      RCX - u8* dst: destination buffer, valid for R8 bytes
; @param[in]      RDX - u8* src: source buffer, valid for R8 bytes
; @param[in]      R8  - usize count: 0..CAP_REQUEST, validated by caller
; @param[out]     RAX - u8* original dst (equals RCX on entry)
; Inputs:         RCX=u8* dst, RDX=u8* src, R8=usize len (0..CAP_REQUEST)
; Outputs:        RAX=u8* dst
; Errors:         none; R8 > CAP_REQUEST is a Precond violation (caller bails)
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11
; Preserves:      RBX,RBP,RDI,RSI,R12-R15
; Locals:         0 (leaf)
; Max read:       R8 bytes from [RDX]
; Max write:      R8 bytes to [RCX]
; Precond:        R8 <= CAP_REQUEST; buffers do not overlap; R8 == 0 is a no-op
; Stack:          no frame (leaf); caller's RSP unchanged
; Modified:       RAX,RCX,RDX,R8,R9,R10,R11
; Initial inputs to registers: dst->RCX, src->RDX, len->R8
; Register assignments:
;   save_phase:   RAX=dst (saved for return)
;   copy_phase:   RSI=src cursor, RDI=dst cursor, RCX=remaining count
; ===========================================================================
global mem_copy
mem_copy:
    mov     rax, rcx                 ; save dst for return (RAX = Win64 return)
    mov     rsi, rdx                 ; src cursor
    mov     rdi, rcx                 ; dst cursor
    mov     rcx, r8                  ; byte count for rep movsb
    rep     movsb
    ret
```

Every code block above obeys `default rel` and the `[label + reg]` ban. The
three headers are progressively more expensive to write and progressively
cheaper to reason about across module boundaries.

---

## 9. Maintenance discipline

Treat contracts as code. A stale `Clobbers:` line is worse than no comment at
all because it lies with authority. Rules (research §7.6):

- **Update the contract in the same commit as the register change.** If you
  add a `push rbx` to save a new nonvolatile, `Preserves:` and `Clobbers:`
  must reflect the new regime in the same diff.
- **Re-derive `Max read`/`Max write` whenever a `CAP_*` constant changes.**
  If `CAP_REQUEST` moves from 8192 to 16384, every `Max read:` citing it is
  automatically still correct (cite the symbol, not the number) — but every
  bound check inside the routine must be re-audited.
- **Tier is sticky upward.** A routine that grows past 20 lines or gains a
  `call` gets promoted to the next tier in the same commit. A routine never
  silently drops a tier.
- **Reviewer checklist (every PR touching `.asm`/`.inc`):**
  1. Is `default rel` at the top of every `.asm` file? (§4)
  2. Does every `global` and `call` target have a tier-appropriate header? (§2, §7)
  3. Does `Clobbers:` exactly match the set of volatile registers the body
     touches, and does `Preserves:` exactly match the nonvolatile registers
     saved via `push`? (§3)
  4. Do `Max read:` and `Max write:` cite a `CAP_*` symbol, not a literal? (§4)
  5. Is every `[label + reg]` replaced by the two-step `lea`/`add` idiom? (§4)
- **Future linter integration** is anticipated but not blocking. If
  `jedi-knights/nasm-lint` (research §7.2) ships rule NL043 register liveness,
  the existing `Clobbers:`/`Preserves:` lines become machine-checkable with no
  rewrite. Write them as if a linter is already reading them.

---

## 10. What NOT to do

The vocabulary anti-patterns (HeavyThing `prolog`/`epilog`, AT&T `%reg`,
`/* */` C comments, `sal.h` macros, `;; ` double-semicolon) are listed in
§5. Do not repeat them here. The items below are the *behavioral*
anti-patterns — mistakes in how fields are filled, not which keywords are
allowed.

- **Do not** invent new field names or reorder the 7-field template. The order
  in `win64.inc:39-47` is canonical; copy it verbatim.
- **Do not** make Doxygen `@param` tags mandatory at Tier 1 or Tier 2. They
  are a Tier 3 tool; pasting them on every leaf adds noise without tooling
  payoff (research §7.7).
- **Do not** write `Clobbers: etc.` or `Preserves: usual`. Enumerate the full
  register set, or write `none`, or write `volatile`/`non-volatile` as
  shorthand for the canonical Win64 sets at `win64.inc:15-18`.
- **Do not** cite a literal buffer size (`8192`) in `Max read`/`Max write`.
  Cite the `CAP_*` symbol so the bound tracks the constant.
- **Do not** skip `Max read`/`Max write` on a routine that touches a buffer.
  These two fields are the project's defense against the class of bugs (recv
  overflow, header parse overrun) that the rest of the convention exists to
  prevent.
- **Do not** treat this standard as aspirational. Every field you write is a
  claim about reality; if reality changes and the comment does not, the
  comment is a bug.

---

**Companion files:**
- `include/win64.inc` — call-frame macros and the 7-field template (lines 1-87).
- `include/config.inc` — `CAP_*` bounds, `struc ClientSlot`, `GW_*` and `CS_*`
  state enums (lines 1-97).
- `docs/assembly-annotation-research.md` — the 950-line evidence base for this
  standard; cite it, do not duplicate it.
