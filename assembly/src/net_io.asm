; ===========================================================================
; src/net_io.asm - socket I/O primitives (PLAN §2.5)
; ===========================================================================
%include "win64.inc"
%include "winsock.inc"
%include "config.inc"

extern timeout_ms

default rel

section .text

; ---------------------------------------------------------------------------
; send_all - send exactly len bytes, advancing through partial sends.
; ---------------------------------------------------------------------------
; Inputs:  RCX = socket, RDX = buffer ptr, R8 = byte count.
; Outputs: RAX = 0 on success, nonzero on error.
; Clobbers: volatile + saved (rbx,rsi,r12).
; Alignment: 4 pushes (rbp,rbx,rsi,r12) entry≡8 -> ≡8; sub 40 -> ≡0. OK.
; Max write: exactly R8 bytes across one or more send() calls.
; ---------------------------------------------------------------------------
global send_all
send_all:
    push    rbp
    mov     rbp, rsp
    push    rbx                  ; socket
    push    rsi                  ; buffer cursor
    push    r12                  ; remaining count
    sub     rsp, 40              ; shadow(32)+8; aligned
    mov     rbx, rcx
    mov     rsi, rdx
    mov     r12, r8
.loop:
    test    r12, r12
    jz      .ok
    mov     rcx, rbx             ; sock
    mov     rdx, rsi             ; buf
    mov     r8,  r12             ; len
%if DEBUG
    cmp     r8, DBG_SEND_CHUNK
    jbe     .send_chunk_ready
    mov     r8, DBG_SEND_CHUNK
.send_chunk_ready:
%endif
    xor     r9d, r9d             ; flags = 0
    call    send                 ; eax = bytes sent, or SOCKET_ERROR (-1)
    mov     r10d, eax            ; capture as zero-extended 32-bit
    test    r10d, r10d
    jle     .fail                ; error or no forward progress
    ; advance by bytes actually sent
    add     rsi, r10
    sub     r12, r10
    jmp     .loop
.ok:
    xor     eax, eax
    jmp     .out
.fail:
    mov     eax, 1
.out:
    add     rsp, 40
    pop     r12
    pop     rsi
    pop     rbx
    pop     rbp
    ret

; ---------------------------------------------------------------------------
; apply_timeouts - set SO_RCVTIMEO and SO_SNDTIMEO on a socket.
; ---------------------------------------------------------------------------
; Inputs:  RCX = socket.
; Outputs: none (best-effort; errors ignored for M2).
; Clobbers: volatile + saved (rbx).
; Alignment: 2 pushes (rbp,rbx) entry≡8 -> ≡8; sub 40 -> ≡0. OK.
; ---------------------------------------------------------------------------
global apply_timeouts
apply_timeouts:
    push    rbp
    mov     rbp, rsp
    push    rbx
    sub     rsp, 40              ; shadow+8; [rsp+32]=optlen; aligned
    mov     rbx, rcx
    mov     dword [rsp+0x20], 4  ; optlen = sizeof(DWORD)
    ; SO_RCVTIMEO
    mov     rcx, rbx
    mov     edx, SOL_SOCKET
    mov     r8d, SO_RCVTIMEO
    lea     r9,  [timeout_ms]
    call    setsockopt
    ; SO_SNDTIMEO
    mov     rcx, rbx
    mov     edx, SOL_SOCKET
    mov     r8d, SO_SNDTIMEO
    lea     r9,  [timeout_ms]
    call    setsockopt
    add     rsp, 40
    pop     rbx
    pop     rbp
    ret
