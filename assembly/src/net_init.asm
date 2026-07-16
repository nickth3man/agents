; ===========================================================================
; src/net_init.asm - Winsock lifecycle: startup, listener, shutdown (PLAN §2.3)
; ===========================================================================
%include "win64.inc"
%include "winapi.inc"
%include "winsock.inc"
%include "config.inc"

extern wsadata
extern listen_sock
extern excl_enable
extern log_err
extern log_err_code

default rel

section .data
; sockaddr_in for bind: AF_INET, port 8080 (net order), 127.0.0.1 (net order).
global listen_addr
listen_addr:
    dw  AF_INET               ; sin_family = 2
    dw  PORT_8080_NET         ; sin_port   = htons(8080) = 0x901F
    dd  IP_127_0_0_1_NET      ; sin_addr   = 127.0.0.1 net order = 0x0100007F
    dq  0                     ; sin_zero   (8 bytes)

; error stage strings
s_startup:  db "wsastartup"   ; S_STARTUP_LEN
S_STARTUP_LEN equ $-s_startup
s_socket:   db "socket"
S_SOCKET_LEN  equ $-s_socket
s_setopt:   db "setsockopt_excl"
S_SETOPT_LEN  equ $-s_setopt
s_bind:     db "bind"
S_BIND_LEN    equ $-s_bind
s_listen:   db "listen"
S_LISTEN_LEN  equ $-s_listen
s_clean:    db "wsacleanup"
S_CLEAN_LEN   equ $-s_clean

section .text

; ---------------------------------------------------------------------------
; net_init - bring up Winsock 2.2 and the loopback listener.
; Inputs:  none.
; Outputs: RAX = listening SOCKET on success.
;          On fatal error: logs and exits the process (never returns).
; Clobbers: volatile + rbx (saved).
; Alignment: push rbp/rbx (entry≡8 -> ≡8); sub 40 -> ≡0. OK.
; ---------------------------------------------------------------------------
global net_init
net_init:
    push    rbp
    mov     rbp, rsp
    push    rbx                  ; hold listening socket across calls
    sub     rsp, 40              ; shadow(32)+8; [rsp+32]=5th arg for setsockopt
    ; --- WSAStartup(0x0202, &wsadata) : RCX=version, RDX=&wsadata ---
    mov     rcx, WINSOCK_VER
    lea     rdx, [wsadata]
    call    WSAStartup
    test    eax, eax
    jz      .socket_create
    lea     rcx, [rel s_startup]
    mov     rdx, S_STARTUP_LEN
    mov     r8,  rax             ; WSAStartup returns code directly
    call    log_err_code
    mov     rcx, 1
    call    net_shutdown         ; exits
.socket_create:
    ; --- socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) ---
    mov     rcx, AF_INET
    mov     rdx, SOCK_STREAM
    mov     r8,  IPPROTO_TCP
    call    socket
    cmp     rax, INVALID_SOCKET
    jne     .have_sock
    lea     rcx, [rel s_socket]
    mov     rdx, S_SOCKET_LEN
    call    log_err
    mov     rcx, 1
    call    net_shutdown
.have_sock:
    mov     rbx, rax             ; rbx = listen socket
    mov     [listen_sock], rax
    ; --- setsockopt(SO_EXCLUSIVEADDRUSE) before bind ---
    mov     rcx, rbx
    mov     edx, SOL_SOCKET
    mov     r8d, SO_EXCLUSIVEADDRUSE
    lea     r9,  [excl_enable]   ; &DWORD(1)
    mov     dword [rsp+0x20], 4  ; optlen = 4
    call    setsockopt
    test    eax, eax
    jz      .do_bind
    lea     rcx, [rel s_setopt]
    mov     rdx, S_SETOPT_LEN
    call    log_err
    ; non-fatal: continue to bind (the option is a hardening measure)
.do_bind:
    ; --- bind(listen_sock, &listen_addr, 16) ---
    mov     rcx, rbx
    lea     rdx, [listen_addr]
    mov     r8,  SOCKADDR_IN_SIZE
    call    bind
    test    eax, eax
    jz      .do_listen
    lea     rcx, [rel s_bind]
    mov     rdx, S_BIND_LEN
    call    log_err
    mov     rcx, 1
    call    net_shutdown
.do_listen:
    ; --- listen(listen_sock, backlog=16) ---
    mov     rcx, rbx
    mov     rdx, 16
    call    listen
    test    eax, eax
    jz      .ok
    lea     rcx, [rel s_listen]
    mov     rdx, S_LISTEN_LEN
    call    log_err
    mov     rcx, 1
    call    net_shutdown
.ok:
    mov     rax, rbx             ; return listening socket
    add     rsp, 40
    pop     rbx
    pop     rbp
    ret

; ---------------------------------------------------------------------------
; net_shutdown - orderly shutdown: close listener, WSACleanup, exit.
; Inputs:  RCX = process exit code.
; Outputs: never returns (calls ExitProcess).
; Clobbers: volatile + rbx (saved, though we never return).
; Alignment: push rbp/rbx (entry≡8 -> ≡8); sub 40 -> ≡0. OK.
; ---------------------------------------------------------------------------
global net_shutdown
net_shutdown:
    push    rbp
    mov     rbp, rsp
    push    rbx                  ; exit code
    sub     rsp, 40              ; shadow+8; aligned
    mov     rbx, rcx             ; save exit code
    ; closesocket(listen_sock) if it is a real socket
    mov     rcx, [listen_sock]
    cmp     rcx, INVALID_SOCKET
    je      .skip_close
    call    closesocket
.skip_close:
    call    WSACleanup
    mov     rcx, rbx             ; exit code
    call    ExitProcess          ; no return
    ud2
