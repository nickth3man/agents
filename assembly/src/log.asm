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
; Inputs:  RCX = buf ptr, RDX = len.
; Clobbers: volatile.  Preserves: nonvolatile.
; Alignment: push rbp/rbx/rsi (entry≡8 -> ≡0); sub 48 -> ≡0. OK.
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
; Inputs:  RCX = stage ptr, RDX = stage len, R8 = numeric code.
; Clobbers: volatile + saved nonvolatile (rbx,rsi,r12,r13,r14).
; Alignment: 6 pushes (rbp,rbx,rsi,r12,r13,r14) from entry≡8 -> rsp≡0;
;           sub 32 -> ≡0. OK.
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
; Inputs:  RCX = stage ptr, RDX = stage len.
; IMPORTANT (PLAN §7.5): call IMMEDIATELY after the failing socket call.
; Clobbers: volatile + saved nonvolatile.
; Alignment: 4 pushes (rbp,rbx,rsi,r12) from entry≡8 -> ≡0; sub 32 -> ≡0. OK.
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
; Inputs:  RCX = stage ptr, RDX = stage len, R8 = code (the API's return value).
; Clobbers: volatile.
; Alignment: tail-style: just set up rcx/rdx/r8 and call _emit_err_line. The
;           CALL to _emit_err_line needs rsp aligned; we have not modified rsp
;           since entry, and entry≡8 (caller's responsibility was to align at
;           the call to US). We pass through unchanged. OK as long as caller
;           aligned at its call to log_err_code (which it did).
; ---------------------------------------------------------------------------
global log_err_code
log_err_code:
    jmp     _emit_err_line

; ---------------------------------------------------------------------------
; log_request - emit "[request] id=N method=M path=P status=S in=I out=O\n"
; ---------------------------------------------------------------------------
; Inputs:  RCX = status code, RDX = response bytes sent (out).
; Reads globals: req_id, req_method_ptr/len, req_path_ptr/len, req_content_length.
; The line is built in log_scratch and truncated (path/method) at CAP_LOG_LINE.
; Clobbers: volatile + saved rbx,rsi,r12,r13,r14,r15.
; Alignment: 7 pushes (rbp,rbx,rsi,r12,r13,r14,r15) entry≡8 -> ≡0; sub 32 -> ≡0.
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
