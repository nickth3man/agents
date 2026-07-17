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

; ===========================================================================
; net_init - bring up Winsock 2.2 and the loopback listener
; Purpose:        Initialises Winsock via WSAStartup, creates a TCP socket,
;                 binds to 127.0.0.1:8080 with SO_EXCLUSIVEADDRUSE, begins
;                 listening (backlog=64), and stores the listen socket in
;                 [listen_sock] and RBX. On any fatal error, logs the stage
;                 name and calls net_shutdown(1) to exit the process.
; @param[out]     RAX - SOCKET listening socket on success; error never returns
; Inputs:         none (operates on globals: wsadata, listen_addr, excl_enable)
; Outputs:        RAX=SOCKET (listen_sock); [listen_sock] set globally
; Errors:         On WSAStartup/socket/bind/listen failure: logs via
;                 log_err/log_err_code then calls net_shutdown(1) (exits)
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11
; Preserves:      RBX,RBP,RDI,RSI,R12-R15
; Locals:         40 (32 shadow + 8 for [rsp+0x20]=optlen)
; Max read:       0 (no caller buffers read)
; Max write:      0 (no caller buffers written; globals listen_addr, listen_sock)
; Precond:        Winsock not yet initialised; 127.0.0.1:8080 available;
;                 wsadata, listen_sock, excl_enable globals exist
; Stack:          2 pushes (RBP,RBX) + sub 40; RSP 16-aligned at CALL
; Modified:       RAX,RCX,RDX,R8,R9,R10,R11
; Initial inputs to registers: none
; Register assignments:
;   wsastartup_phase: RCX=WINSOCK_VER, RDX=&wsadata, RAX=WSAStartup result
;   socket_phase:     RCX=AF_INET, RDX=SOCK_STREAM, R8=IPPROTO_TCP,
;                     RAX=SOCKET -> RBX and [listen_sock]
;   setopt_phase:     RCX=RBX, RDX=SOL_SOCKET, R8D=SO_EXCLUSIVEADDRUSE,
;                     R9=&excl_enable, [rsp+0x20]=4, RAX=setsockopt result
;   bind_phase:       RCX=RBX, RDX=&listen_addr, R8=SOCKADDR_IN_SIZE,
;                     RAX=bind result
;   listen_phase:     RCX=RBX, RDX=64 (backlog), RAX=listen result
;   ok_phase:         RAX=RBX (return listen socket)
; ===========================================================================
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
    ; --- listen(listen_sock, backlog=64) ---
    mov     rcx, rbx
    mov     rdx, 64
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

; ===========================================================================
; net_shutdown - orderly shutdown: close listener, WSACleanup, exit
; Purpose:        Closes [listen_sock] if valid, calls WSACleanup, then
;                 terminates the process via ExitProcess with the given exit
;                 code. Designed to be called from net_init error paths and
;                 from start.asm on graceful shutdown.
; @param[in]      RCX - int exit_code: process exit code for ExitProcess
; Inputs:         RCX=int exit_code; reads [listen_sock] global
; Outputs:        never returns (calls ExitProcess)
; Errors:         never returns (ExitProcess terminates the process)
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11
; Preserves:      RBX,RBP,RDI,RSI,R12-R15
; Locals:         40 (32 shadow + 8 alignment padding)
; Max read:       8 bytes from [listen_sock]
; Max write:      0 (no caller buffers written)
; Precond:        net_init may or may not have completed; listen_sock is
;                 either INVALID_SOCKET or a valid SOCKET from net_init
; Stack:          2 pushes (RBP,RBX) + sub 40; RSP 16-aligned at CALL
; Modified:       RAX,RCX,RDX,R8,R9,R10,R11
; Initial inputs to registers: exit_code->RCX
; Register assignments:
;   closesocket_phase: RCX=[listen_sock]; skip close if INVALID_SOCKET
;   cleanup_phase:     RCX=exit_code saved in RBX; calls WSACleanup
;   exit_phase:        RCX=RBX (exit code); calls ExitProcess (no return)
; ===========================================================================
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
