; ===========================================================================
; src/net_io.asm - socket I/O primitives (PLAN §2.5)
; ===========================================================================
%include "win64.inc"
%include "winsock.inc"
%include "config.inc"

extern timeout_ms

default rel

section .text

; ===========================================================================
; send_all - send exactly len bytes, advancing through partial sends
; Purpose:        Calls send() in a loop until all R8 bytes from [RDX] have
;                 been written to socket RCX. Advances cursor on partial sends.
;                 Bounded only by R8; caller guarantees buffer validity.
; @param[in]      RCX - SOCKET client socket: connected TCP socket
; @param[in]      RDX - u8* buf: buffer to send, valid for R8 bytes
; @param[in]      R8  - usize len: exact number of bytes to send
; @param[out]     RAX - int 0 on success, 1 on failure
; Inputs:         RCX=SOCKET, RDX=u8* buf, R8=usize len
; Outputs:        RAX=0 success, 1 failure
; Errors:         RAX=1 on send failure (SOCKET_ERROR); WSAGetLastError for code
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11
; Preserves:      RBX,RBP,RDI,RSI,R12-R15
; Locals:         40 (32 shadow + 8 alignment padding)
; Max read:       R8 bytes from [RDX]
; Max write:      R8 bytes to socket via send() (Winsock reads from buffer)
; Precond:        RCX valid SOCKET; [RDX,R8) readable; R8 > 0
; Stack:          4 pushes (RBP,RBX,RSI,R12) + sub 40; RSP 16-aligned at CALL
; Modified:       RAX,RCX,RDX,R8,R9,R10,R11
; Initial inputs to registers: socket->RCX, buf->RDX, len->R8
; Register assignments:
;   init_phase:   RCX=socket->RBX, RDX=buf->RSI, R8=len->R12
;   loop_phase:   RBX=socket, RSI=cursor, R12=remaining,
;                 RCX,RDX,R8,R9=send() args, RAX=bytes sent (->R10D for test)
;   ok_phase:     RAX=0
;   error_phase:  RAX=1
; ===========================================================================
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

; ===========================================================================
; apply_timeouts - set SO_RCVTIMEO and SO_SNDTIMEO on a socket
; Purpose:        Calls setsockopt twice to configure receive and send
;                 timeouts from the [timeout_ms] global DWORD. Errors are
;                 silently ignored (best-effort per Milestone 2).
; @param[in]      RCX - SOCKET client socket: socket to configure
; @param[out]     RAX - int last setsockopt return (callers ignore)
; Inputs:         RCX=SOCKET; reads [timeout_ms] global DWORD
; Outputs:        RAX=last setsockopt return (unused)
; Errors:         none (errors suppressed; socket retains defaults)
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11
; Preserves:      RBX,RBP,RDI,RSI,R12-R15
; Locals:         40 (32 shadow + 8 for [rsp+0x20]=optlen)
; Max read:       4 bytes from [timeout_ms] (read by Winsock via setsockopt)
; Max write:      0 (no caller buffers written)
; Precond:        RCX valid SOCKET; [timeout_ms] initialised to valid DWORD
; Stack:          2 pushes (RBP,RBX) + sub 40; RSP 16-aligned at CALL
; Modified:       RAX,RCX,RDX,R8,R9,R10,R11
; Initial inputs to registers: socket->RCX
; Register assignments:
;   prep_phase:   RCX=socket->RBX; [rsp+0x20]=4 (optlen=sizeof DWORD)
;   rcv_phase:    RCX=RBX, RDX=SOL_SOCKET, R8D=SO_RCVTIMEO,
;                 R9=&timeout_ms, RAX=setsockopt return
;   snd_phase:    RCX=RBX, RDX=SOL_SOCKET, R8D=SO_SNDTIMEO,
;                 R9=&timeout_ms, RAX=setsockopt return
; ===========================================================================
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
