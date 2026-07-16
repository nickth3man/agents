; ===========================================================================
; src/log.asm - stderr structured logging (PLAN §4.4, §7.5)
; ===========================================================================
%include "win64.inc"
%include "winapi.inc"
%include "config.inc"

extern WSAGetLastError
extern u32_to_dec
extern log_scratch
extern last_wsa

section .data
err_pfx:    db  "[error] stage="
ERR_PFX_LEN equ $-err_pfx
err_mid:    db  " wsa="
ERR_MID_LEN equ $-err_mid
err_nl:     db  10
ERR_NL_LEN  equ $-err_nl

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
; copy_bytes - tiny memcpy. rcx=dst, r8=src, rdx=len. Leaf (no calls).
; Clobbers: rax,rcx,rdx,r8. Returns rcx/r8 advanced past end.
; ---------------------------------------------------------------------------
copy_bytes:
    test    rdx, rdx
    jz      .done
.loop:
    mov     al, [r8]
    mov     [rcx], al
    inc     rcx
    inc     r8
    dec     rdx
    jnz     .loop
.done:
    ret

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
    mov     rcx, r12d            ; value
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
    ; emit
    mov     rcx, log_scratch
    mov     rdx, r13
    sub     rdx, log_scratch
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
