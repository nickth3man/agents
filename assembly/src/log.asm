; ===========================================================================
; src/log.asm - stderr structured logging (PLAN §4.4, §7.5)
; ===========================================================================
%include "win64.inc"
%include "winapi.inc"
%include "config.inc"

extern WSAGetLastError
extern u32_to_dec
extern copy_bytes
extern log_scratch
extern last_wsa
extern req_id
extern req_method_ptr
extern req_method_len
extern req_path_ptr
extern req_path_len
extern req_content_length

default rel

section .data
err_pfx:    db  "[error] stage="
ERR_PFX_LEN equ $-err_pfx
err_mid:    db  " wsa="
ERR_MID_LEN equ $-err_mid
err_nl:     db  10
ERR_NL_LEN  equ $-err_nl
; request-line fragments (PLAN §4.4)
rq_pfx:     db  "[request] id="
RQ_PFX_LEN  equ $-rq_pfx
rq_m:       db  " method="
RQ_M_LEN    equ $-rq_m
rq_p:       db  " path="
RQ_P_LEN    equ $-rq_p
rq_s:       db  " status="
RQ_S_LEN    equ $-rq_s
rq_in:      db  " in="
RQ_IN_LEN   equ $-rq_in
rq_out:     db  " out="
RQ_OUT_LEN  equ $-rq_out
rq_nl:      db  10
RQ_NL_LEN   equ $-rq_nl

section .bss
log_written: resd 1                  ; WriteFile bytes-written slot

section .text

; ---------------------------------------------------------------------------
; log_str - write a counted string to stderr (no formatting).
; Purpose:        Writes RDX bytes from [RCX] to stderr via WriteFile. Used by
;                 all log emission paths.
; Inputs:         RCX=u8* buf ptr, RDX=usize len (bytes to write)
; Outputs:        RAX=WriteFile return (BOOL; non-zero=success, 0=failure);
;                 side effect: writes to stderr, updates [log_written]
; Errors:         none (caller ignores log failures by convention)
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11 (volatile via Win32 calls)
; Preserves:      RBX,RBP,RDI,RSI,R12-R15 (all nonvolatile; rbx,rsi,rbp saved)
; Locals:         48 (32 shadow + 16 for 5th arg lpOverlapped; hand-rolled)
; Max read:       RDX bytes from [RCX]
; Max write:      4 bytes to [log_written] (DWORD from WriteFile)
; Precond:        RCX valid for RDX bytes; RDX may be 0 (no-op)
; ---------------------------------------------------------------------------
global log_str
log_str:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    rsi
    sub     rsp, 48              ; shadow(32)+16; [rsp+32]=5th arg
    mov     rbx, rcx             ; buf
    mov     rsi, rdx             ; len
    mov     rcx, STD_ERROR_HANDLE
    call    GetStdHandle         ; rax = handle
    mov     rcx, rax             ; hFile
    mov     rdx, rbx             ; lpBuffer
    mov     r8,  rsi             ; len
    lea     r9,  [log_written]   ; &written
    mov     qword [rsp+0x20], 0  ; lpOverlapped=NULL
    call    WriteFile
    add     rsp, 48
    pop     rsi
    pop     rbx
    pop     rbp
    ret

; ---------------------------------------------------------------------------
; copy_bytes is provided by text.asm (shared).
; ---------------------------------------------------------------------------

; ---------------------------------------------------------------------------
; _emit_err_line - build "[error] stage=<stage> wsa=<code>\n" + emit.
; Purpose:        Formats an error line into log_scratch, then emits via
;                 log_str. Used by log_err and log_err_code.
; Inputs:         RCX=u8* stage ptr, RDX=usize stage len, R8=u32 code
; Outputs:        RAX=log_str return (WriteFile BOOL); side effect: error line
;                 written to stderr via log_str
; Errors:         none
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11 (volatile via copy_bytes,
;                 u32_to_dec, log_str)
; Preserves:      RBX,RBP,RDI,RSI,R12-R15 (all nonvolatile; rbx,rsi,r12,r13,
;                 r14,rbp saved via push/pop)
; Locals:         32 (shadow only; 6 pushes via hand-rolled frame)
; Max read:       RDX bytes from [RCX] (stage string); fixed constants from
;                 err_pfx, err_mid, err_nl
; Max write:      CAP_LOG_LINE bytes to [log_scratch] (no runtime clamp;
;                 caller must ensure total fits within buffer)
; Precond:        RCX valid for RDX bytes; R8 contains valid u32; total
;                 formatted line must fit in CAP_LOG_LINE bytes
; ---------------------------------------------------------------------------
_emit_err_line:
    push    rbp
    mov     rbp, rsp
    push    rbx                  ; stage ptr
    push    rsi                  ; stage len
    push    r12                  ; code
    push    r13                  ; write cursor
    push    r14                  ; (alignment / scratch)
    sub     rsp, 32              ; shadow; aligned
    mov     rbx, rcx
    mov     rsi, rdx
    mov     r12d, r8d
    lea     r13, [log_scratch]
    ; "[error] stage="
    mov     rcx, r13
    lea     r8,  [err_pfx]
    mov     rdx, ERR_PFX_LEN
    call    copy_bytes
    add     r13, ERR_PFX_LEN
    ; <stage>
    mov     rcx, r13
    mov     r8,  rbx
    mov     rdx, rsi
    call    copy_bytes
    add     r13, rsi
    ; " wsa="
    mov     rcx, r13
    lea     r8,  [err_mid]
    mov     rdx, ERR_MID_LEN
    call    copy_bytes
    add     r13, ERR_MID_LEN
    ; <code>
    mov     ecx, r12d            ; value (32-bit -> rcx zero-extended)
    mov     rdx, r13             ; buf
    mov     r8,  16              ; cap
    call    u32_to_dec           ; rax = digit count
    add     r13, rax
    ; "\n"
    mov     rcx, r13
    lea     r8,  [err_nl]
    mov     rdx, ERR_NL_LEN
    call    copy_bytes
    add     r13, ERR_NL_LEN
    ; emit: rcx=buf, rdx=len
    lea     r11, [rel log_scratch]   ; start address (volatile scratch reg)
    mov     rdx, r13                 ; cursor
    sub     rdx, r11                 ; length = cursor - start
    mov     rcx, r11                 ; buf = start
    call    log_str
    add     rsp, 32
    pop     r14
    pop     r13
    pop     r12
    pop     rsi
    pop     rbx
    pop     rbp
    ret

; ---------------------------------------------------------------------------
; log_err - capture WSAGetLastError THEN emit. Use for all socket-call errors.
; Purpose:        Calls WSAGetLastError, saves result to [last_wsa], then
;                 forwards to _emit_err_line with the error code. Must be
;                 called immediately after the failing socket call (PLAN §7.5).
; Inputs:         RCX=u8* stage ptr, RDX=usize stage len
; Outputs:        RAX=_emit_err_line return (WriteFile BOOL); side effect:
;                 [last_wsa] updated, error line emitted to stderr
; Errors:         none
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11 (volatile via WSAGetLastError and
;                 _emit_err_line)
; Preserves:      RBX,RBP,RDI,RSI,R12-R15 (all nonvolatile; rbx,rsi,r12,rbp
;                 saved via push/pop)
; Locals:         32 (shadow only; 4 pushes via hand-rolled frame)
; Max read:       RDX bytes from [RCX] (stage string, forwarded)
; Max write:      4 bytes to [last_wsa] (WSA error code); plus CAP_LOG_LINE
;                 bytes to [log_scratch] via _emit_err_line
; Precond:        Call immediately after failing WSA call; RCX valid for RDX bytes
; ---------------------------------------------------------------------------
global log_err
log_err:
    push    rbp
    mov     rbp, rsp
    push    rbx                  ; stage ptr
    push    rsi                  ; stage len
    push    r12                  ; code
    sub     rsp, 32              ; shadow; aligned
    mov     rbx, rcx
    mov     rsi, rdx
    call    WSAGetLastError      ; capture FIRST
    mov     [last_wsa], eax
    mov     r12d, eax
    mov     rcx, rbx
    mov     rdx, rsi
    mov     r8,  r12
    call    _emit_err_line
    add     rsp, 32
    pop     r12
    pop     rsi
    pop     rbx
    pop     rbp
    ret

; ---------------------------------------------------------------------------
; log_err_code - emit with an EXPLICIT code (WSAStartup special case, PLAN §4.4).
; Purpose:        Thin tail-jump wrapper to _emit_err_line. Forwards all three
;                 register args unchanged. 2-line leaf — no frame, no shadow.
;                 Unlike log_err, does NOT call WSAGetLastError; the caller
;                 provides the code explicitly in R8.
; Inputs:         RCX=u8* stage ptr, RDX=usize stage len, R8=u32 code
; Outputs:        RAX=_emit_err_line return (WriteFile BOOL); side effect:
;                 error line emitted to stderr
; Errors:         none
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11 (volatile; same as _emit_err_line
;                 which performs all work)
; Preserves:      RBX,RBP,RDI,RSI,R12-R15 (all nonvolatile; no frame, no
;                 push/pop in this wrapper; _emit_err_line preserves them)
; Locals:         0 (leaf tail-jump; no CALL, no sub rsp, no frame)
; Max read:       RDX bytes from [RCX] (stage string, forwarded)
; Max write:      CAP_LOG_LINE bytes to [log_scratch] via _emit_err_line
; Precond:        R8 contains explicit error code (not WSAGetLastError); same
;                 preconditions as _emit_err_line apply
; ---------------------------------------------------------------------------
global log_err_code
log_err_code:
    jmp     _emit_err_line

; ---------------------------------------------------------------------------
; log_request - emit "[request] id=N method=M path=P status=S in=I out=O\n"
; Purpose:        Builds a structured request log line in log_scratch from
;                 globals (req_id, req_method_*, req_path_*, req_content_length)
;                 and register args (status, out bytes). Method and path are
;                 clamped to remaining scratch capacity. Emits via log_str.
; Inputs:         RCX=u32 status (HTTP status code), RDX=u64 out_bytes
;                 (response bytes sent). Reads req_* globals.
; Outputs:        RAX=log_str return (WriteFile BOOL); side effect: request
;                 log line emitted to stderr
; Errors:         none
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11 (volatile via copy_bytes,
;                 u32_to_dec, log_str)
; Preserves:      RBX,RBP,RDI,RSI,R12-R15 (all nonvolatile; rbx,rsi,r12,r13,
;                 r14,r15,rbp saved via push/pop)
; Locals:         32 (shadow only; 7 pushes via hand-rolled frame)
; Max read:       req_method_len bytes from [req_method_ptr]; req_path_len
;                 bytes from [req_path_ptr]; req_id (8 B), req_content_length
;                 (8 B) from globals
; Max write:      CAP_LOG_LINE bytes to [log_scratch] (method/path clamped to
;                 remaining capacity)
; Precond:        req_* globals populated by http_read/http_parse; RCX, RDX
;                 carry caller's status/byte count
; ---------------------------------------------------------------------------
global log_request
log_request:
    push    rbp
    mov     rbp, rsp
    push    rbx                     ; status
    push    rsi                     ; out bytes
    push    r12                     ; write cursor
    push    r13                     ; end limit (scratch + CAP_LOG_LINE)
    push    r14                     ; remaining
    push    r15                     ; clamped length
    sub     rsp, 32                 ; shadow; aligned
    mov     rbx, rcx                ; status
    mov     rsi, rdx                ; out
    lea     r12, [rel log_scratch]
    lea     r13, [rel log_scratch]
    add     r13, CAP_LOG_LINE
    ; "[request] id="
    mov     rcx, r12
    lea     r8,  [rq_pfx]
    mov     rdx, RQ_PFX_LEN
    call    copy_bytes
    add     r12, RQ_PFX_LEN
    ; id decimal
    mov     rcx, [rel req_id]
    mov     rdx, r12
    mov     r8,  12
    call    u32_to_dec
    add     r12, rax
    ; " method="
    mov     rcx, r12
    lea     r8,  [rq_m]
    mov     rdx, RQ_M_LEN
    call    copy_bytes
    add     r12, RQ_M_LEN
    ; method span (clamped to remaining)
    mov     r14, r13
    sub     r14, r12
    test    r14, r14
    jle     .emit
    mov     r15, [rel req_method_len]
    cmp     r15, r14
    jle     .m_ok
    mov     r15, r14
.m_ok:
    mov     rcx, r12
    mov     r8,  [rel req_method_ptr]
    mov     rdx, r15
    call    copy_bytes
    add     r12, r15
    ; " path="
    mov     rcx, r12
    lea     r8,  [rq_p]
    mov     rdx, RQ_P_LEN
    call    copy_bytes
    add     r12, RQ_P_LEN
    ; path span (clamped)
    mov     r14, r13
    sub     r14, r12
    test    r14, r14
    jle     .emit
    mov     r15, [rel req_path_len]
    cmp     r15, r14
    jle     .p_ok
    mov     r15, r14
.p_ok:
    mov     rcx, r12
    mov     r8,  [rel req_path_ptr]
    mov     rdx, r15
    call    copy_bytes
    add     r12, r15
    ; " status="
    mov     rcx, r12
    lea     r8,  [rq_s]
    mov     rdx, RQ_S_LEN
    call    copy_bytes
    add     r12, RQ_S_LEN
    ; status decimal
    mov     rcx, rbx
    mov     rdx, r12
    mov     r8,  6
    call    u32_to_dec
    add     r12, rax
    ; " in="
    mov     rcx, r12
    lea     r8,  [rq_in]
    mov     rdx, RQ_IN_LEN
    call    copy_bytes
    add     r12, RQ_IN_LEN
    ; in decimal = content_length
    mov     rcx, [rel req_content_length]
    mov     rdx, r12
    mov     r8,  12
    call    u32_to_dec
    add     r12, rax
    ; " out="
    mov     rcx, r12
    lea     r8,  [rq_out]
    mov     rdx, RQ_OUT_LEN
    call    copy_bytes
    add     r12, RQ_OUT_LEN
    ; out decimal
    mov     rcx, rsi
    mov     rdx, r12
    mov     r8,  12
    call    u32_to_dec
    add     r12, rax
    ; "\n"
    mov     rcx, r12
    lea     r8,  [rq_nl]
    mov     rdx, RQ_NL_LEN
    call    copy_bytes
    add     r12, RQ_NL_LEN
.emit:
    lea     rcx, [rel log_scratch]
    mov     rdx, r12
    sub     rdx, rcx                ; total length
    call    log_str
    add     rsp, 32
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rsi
    pop     rbx
    pop     rbp
    ret
