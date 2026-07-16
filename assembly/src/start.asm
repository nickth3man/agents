; ===========================================================================
; src/start.asm - program entry point + main accept/echo loop (PLAN M2)
; ===========================================================================
%include "win64.inc"
%include "winapi.inc"
%include "winsock.inc"
%include "config.inc"
%include "generated/version.inc"

default rel

extern net_init
extern net_shutdown
extern apply_timeouts
extern send_all
extern log_err
extern log_str
extern listen_sock
extern client_sock
extern recv_buf

section .data
banner:
    db  "[startup] build=", BUILD_ID, " listen=127.0.0.1:8080", 10
BANNER_LEN  equ $-banner
s_accept:   db "accept"
S_ACCEPT_LEN equ $-s_accept
s_recv:     db "recv"
S_RECV_LEN  equ $-s_recv

section .text

; ---------------------------------------------------------------------------
; start - true OS entry point (no caller, never returns).
; Brings up the listener, then loops: accept -> recv/echo until peer close ->
; closesocket. Accept failure is fatal; recv errors close the client and
; continue to the next accept.
; Alignment: `and rsp,-16; sub rsp,48` keeps rsp≡0 at every call site.
; ---------------------------------------------------------------------------
global start
start:
    and     rsp, -16                 ; force 16-byte alignment
    sub     rsp, 48                  ; shadow(32)+16; aligned
    ; --- startup banner ---
    lea     rcx, [banner]
    mov     rdx, BANNER_LEN
    call    log_str
    ; --- bring up winsock + listener ---
    call    net_init                 ; rax = listen socket (also stored globally)
    ; ===================== accept loop =====================
.accept_loop:
    mov     rcx, [listen_sock]
    xor     edx, edx                 ; addr = NULL
    xor     r8d, r8d                 ; addrlen = NULL
    call    accept
    cmp     rax, INVALID_SOCKET
    je      .accept_failed
    mov     [client_sock], rax
    mov     rcx, rax
    call    apply_timeouts
    ; ---------------- recv/echo until peer closes ----------------
.recv_loop:
    mov     rcx, [client_sock]
    lea     rdx, [recv_buf]
    mov     r8,  CAP_REQUEST         ; bounded read window
    xor     r9d, r9d                 ; flags = 0
    call    recv
    mov     r11d, eax                ; capture bytes (zero-extended 32-bit)
    test    r11d, r11d
    js      .recv_failed             ; SOCKET_ERROR
    je      .client_done             ; 0 = orderly peer shutdown
    ; echo exactly what we received
    mov     rcx, [client_sock]
    lea     rdx, [recv_buf]
    mov     r8,  r11
    call    send_all
    test    eax, eax
    jnz     .client_done             ; send error -> drop client
    jmp     .recv_loop
.client_done:
    mov     rcx, [client_sock]
    call    closesocket
    jmp     .accept_loop
.recv_failed:
    lea     rcx, [s_recv]
    mov     rdx, S_RECV_LEN
    call    log_err
    mov     rcx, [client_sock]
    call    closesocket
    jmp     .accept_loop
.accept_failed:
    lea     rcx, [s_accept]
    mov     rdx, S_ACCEPT_LEN
    call    log_err
    mov     rcx, 1
    call    net_shutdown             ; exits
    ud2
