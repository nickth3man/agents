; ===========================================================================
; win64.inc - Win64 ABI conventions + project call-frame discipline
; ===========================================================================
; Reference: Microsoft x64 calling convention (PLAN.md §2.2, §4.2).
;
; Rules every internal/external call must obey:
;   * Integer/pointer args 1-4 in RCX, RDX, R8, R9 (in that order).
;   * Caller ALWAYS reserves 32 bytes of shadow ("home") space on the stack,
;     even if the callee takes fewer than 4 args.
;   * Args 5+ are passed on the stack at [rsp+0x20], [rsp+0x28], [rsp+0x30], ...
;     (i.e. immediately after the 32-byte shadow area).
;   * RSP MUST be 16-byte aligned at the point of every CALL instruction.
;     (Equivalently: at callee entry, RSP ≡ 8 mod 16, because CALL pushed the
;      8-byte return address.)
;   * Volatile / caller-saved: RAX, RCX, RDX, R8, R9, R10, R11, XMM0-5.
;     Do NOT keep important values only in these across a call.
;   * Nonvolatile / callee-saved: RBX, RBP, RDI, RSI, R12-R15, XMM6-15.
;     A function that clobbers any of these MUST save/restore it.
;   * Return value (int/ptr) in RAX.
;
; Project rules (PLAN §2.2):
;   1. Every callable routine begins with a contract comment (see template below).
;   2. One consistent function-frame pattern across the codebase.
;   3. Large buffers in .bss, never on the stack.
;   4. Counted strings as (pointer, length) pairs; NUL only when an API needs it.
;   5. ADDRESSING (ASLR-safe, MANDATORY): every module begins with `default rel`.
;      RIP-relative `[rel label]` is ASLR-safe and needs no relocation. BUT
;      RIP-relative CANNOT take an index register, so `[label + reg]` silently
;      becomes ABSOLUTE and GoLink does NOT relocate the imm64 under /dynamicbase
;      -> wrong pointer at runtime -> crash. Therefore NEVER write
;      `[label + reg]`; compute in two steps:
;          lea rax, [rel label]   ; RIP-relative address of label
;          add rax, <reg>         ; then add any register offset
;      Plain `[label]` (no index) is fine under `default rel`.
;
; ---------------------------------------------------------------------------
; Contract-comment template (Tier 2 baseline — paste at top of every
; non-leaf / >20-line / buffer-touching callable routine):
; ---------------------------------------------------------------------------
;   ; fn_name -
;   ; Purpose:
;   ; Inputs:        RCX=..., RDX=..., R8=..., R9=...
;   ; Outputs:       RAX=...
;   ; Errors:        RAX=<sentinel> on failure
;   ; Clobbers:      RAX,RCX,RDX,R8,R9,R10,R11  (volatile)
;   ; Preserves:     RBX,RBP,RDI,RSI,R12-R15     (nonvolatile)
;   ; Locals:        N bytes
;   ; Max read:      M bytes   Max write: W bytes
;   ; Precond:       <optional; caller obligations>
; ---------------------------------------------------------------------------
; Annotation tiers (condensed — full rules in docs/annotation-standard.md):
; ---------------------------------------------------------------------------
; Tier 1 - Minimal inline (ALL callable labels, including tiny leaves):
;   ; <one-line purpose>
;   ; RCX=..., RDX=..., R8=..., R9=..., ret RAX=...
;   ; Clobbers: volatile   Preserves: nonvolatile   ; (elide only if obvious leaf)
;
; Tier 2 - Structured header (non-leaf, OR >20 lines, OR touches buffers):
;   Paste the Tier 2 template above verbatim. Add `Precond:` when callers owe
;   obligations (alignment, non-null, state, no overlap). `---` separator.
;
; Tier 3 - Strict contract (public externs, state machines, security-critical):
;   Tier 2 + Doxygen `@param[in]`/`@param[out]` tags + optional fields:
;     Stack:, Modified:, Initial inputs to registers:, Register assignments.
;   `===` separator (79 chars) distinguishes Tier 3 from Tier 2.
;
; Decision flowchart: docs/annotation-standard.md §7.
; Worked examples (same fn at all 3 tiers): docs/annotation-standard.md §8.
; Per-function tier classification: docs/annotation-ledger.md.
; ---------------------------------------------------------------------------

%ifndef WIN64_INC
%define WIN64_INC

; Minimum caller-provided shadow space (bytes), always reserved per call site.
%define SHADOW          32

; Standard non-leaf function prologue/epilogue.
; Usage:
;   PROLOGUE 24        ; 24 bytes of locals (excluding shadow); padded to 16
;   ...
;   EPILOGUE
; After PROLOGUE: RBP = saved frame, RSP aligned to 16 and holding
; SHADOW + padded(locals) bytes. [rbp+0]  = old RBP, [rbp+8] = return addr,
; [rbp+16] = first stack arg (arg 5), [rbp-8..] = locals.
%macro PROLOGUE 1-2 0      ; %1=locals, %2=extra stack-arg bytes (beyond arg 5)
  push rbp
  mov  rbp, rsp
  %assign %%total  SHADOW + %1 + %2
  %assign %%total  ((%%total + 15) / 16) * 16     ; round up to multiple of 16
  sub  rsp, %%total
%endmacro

%macro EPILOGUE 0
  lea  rsp, [rbp]
  pop  rbp
  ret
%endmacro

; Convenience: reserve shadow space at a leaf-ish call site that does not need
; a full frame. Allocates 32 bytes (aligned only if RSP was 16-aligned before).
%macro SHADOW_ONLY 0
  sub  rsp, SHADOW
%endmacro
%macro SHADOW_FREE 0
  add  rsp, SHADOW
%endmacro

%endif ; WIN64_INC
