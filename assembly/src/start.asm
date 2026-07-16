; ===========================================================================
; src/start.asm - program entry + accept/respond loop (PLAN Milestone 4)
; ===========================================================================
%include "win64.inc"
%include "winapi.inc"
%include "winsock.inc"
%include "http.inc"
%include "config.inc"
%include "generated/version.inc"

default rel

extern net_init
extern net_shutdown
extern apply_timeouts
extern http_read_request
extern route_request
extern resp_set_error
extern http_respond
extern log_err
extern log_str
extern log_request
extern listen_sock
extern client_sock
extern req_id
extern resp_body_len
extern debug_canaries_init
extern debug_canaries_check

section .data
banner:
    db  "[startup] build=", BUILD_ID, " listen=127.0.0.1:8080", 10
BANNER_LEN  equ $-banner
s_accept:    db "accept"
S_ACCEPT_LEN equ $-s_accept

section .text

; ---------------------------------------------------------------------------
; start - true OS entry point (no caller, never returns).
;   accept -> read+parse -> route -> respond -> close.
; Alignment: and rsp,-16; sub rsp,64 keeps rsp≡0 at every call site.
; ---------------------------------------------------------------------------
global start
start:
    and     rsp, -16
    sub     rsp, 64
    lea     rcx, [banner]
    mov     rdx, BANNER_LEN
    call    log_str
    call    debug_canaries_init
    call    net_init
.accept_loop:
    mov     rcx, [listen_sock]
    xor     edx, edx
    xor     r8d, r8d
    call    accept
    cmp     rax, INVALID_SOCKET
    je      .accept_failed
    mov     [client_sock], rax
    inc     qword [req_id]           ; per-request id for logging (PLAN §4.4)
    mov     rcx, rax
    call    apply_timeouts
    mov     rcx, [client_sock]
    call    http_read_request        ; rax = 0 ok, or HTTP status
    mov     [rsp+0x30], rax
    call    debug_canaries_check
    test    eax, eax
    jnz     .canary_err
    mov     rax, [rsp+0x30]
    test    eax, eax
    jnz     .read_err
    call    route_request            ; rax = status; resp globals set
    mov     [rsp+0x30], rax
    call    debug_canaries_check
    test    eax, eax
    jnz     .canary_err
    mov     rax, [rsp+0x30]
    jmp     .respond
.canary_err:
    mov     eax, HTTP_500
    call    resp_set_error
    jmp     .respond
.read_err:
    call    resp_set_error           ; sets err body; preserves eax (the code)
.respond:
    mov     [rsp+0x30], eax          ; save status across http_respond
    mov     rcx, [client_sock]
    mov     edx, eax
    call    http_respond
    ; structured request log (PLAN §4.4)
    mov     rcx, [rsp+0x30]          ; status
    mov     rdx, [resp_body_len]     ; out bytes
    call    log_request
    ; graceful close: signal FIN so the response is delivered before close
    ; (avoids RST discarding the reply when the request had an unread body)
    mov     rcx, [client_sock]
    mov     edx, SD_SEND
    call    shutdown
    mov     rcx, [client_sock]
    call    closesocket
    jmp     .accept_loop
.accept_failed:
    lea     rcx, [s_accept]
    mov     rdx, S_ACCEPT_LEN
    call    log_err
    mov     rcx, 1
    call    net_shutdown
    ud2

; dev-loop test: harmless comment 1784166174

; harmless 1784166273389101600

; harmless 1784166399107126600

; harmless 1784166506784326500
