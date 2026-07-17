; ===========================================================================
; src/start.asm - program entry + WFMO event loop (async rewrite)
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
extern read_more_request
extern route_request
extern resp_set_error
extern http_respond
extern gateway_start
extern gateway_advance
extern log_err
extern log_str
extern log_request
extern listen_sock
extern client_sock
extern client_event
extern req_id
extern req_used
extern req_header_end
extern req_has_cl
extern req_has_te
extern req_content_length
extern req_method_ptr
extern req_method_len
extern req_path_ptr
extern req_path_len
extern resp_body_len
extern req_cl_count
extern debug_canaries_init
extern debug_canaries_check
extern gw_state
extern pending_chat_sock
extern pending_req_id
extern pending_req_clen
extern listen_event
extern gw_event
extern wsaevents
extern WSAGetLastError
extern ResetEvent

section .data
banner:
    db  "[startup] build=", BUILD_ID, " listen=127.0.0.1:8080", 10
BANNER_LEN  equ $-banner
s_accept:   db "accept"
S_ACCEPT_LEN equ $-s_accept
s_event:    db "WSAEnumNetworkEvents"
S_EVENT_LEN equ $-s_event
s_client_timeout: db "client timeout"
S_CLIENT_TIMEOUT_LEN equ $-s_client_timeout
s_client_close: db "client close"
S_CLIENT_CLOSE_LEN equ $-s_client_close
req_post:   db "POST"
REQ_POST_LEN equ $-req_post
req_path_chat: db "/chat"
REQ_PATH_CHAT_LEN equ $-req_path_chat

; WFMO timeout applied when a client request is in progress.
; If no progress is made for this many milliseconds, the client read
; is aborted (HTTP 400) to prevent a slow client from starving gw_event.
CLIENT_READ_MS    equ 8000

; Single DWORD register for ioctlsocket(FIONBIO) argument.
section .bss
fionbio_arg: resd 1

section .text

; ---------------------------------------------------------------------------
; start - true OS entry point (never returns).
;   Event loop: WaitForMultipleObjects on {listen_event, gw_event}.
;   listen_event: accept & read & route (async /chat defers response).
;   gw_event:     advance gateway state machine; respond when complete.
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

    ; --- Create Winsock event for listen socket (FD_ACCEPT) ---
    call    WSACreateEvent              ; RAX = event handle
    mov     [listen_event], rax
    mov     rcx, [listen_sock]
    mov     rdx, rax
    mov     r8d, FD_ACCEPT
    call    WSAEventSelect

    ; --- Create Winsock event for incremental client reads ---
    ; Created once; associated with the active client socket via
    ; WSAEventSelect(client_sock, client_event, FD_READ|FD_CLOSE)
    ; after accept; disassociated by WSAEventSelect with 0 events
    ; when the request is complete.  Manual-reset (from WSACreateEvent);
    ; WSAEnumNetworkEvents resets it.
    call    WSACreateEvent
    mov     [client_event], rax

    ; --- Create auto-reset event for gateway completion ---
    xor     ecx, ecx                    ; lpEventAttributes = NULL
    xor     edx, edx                    ; bManualReset = FALSE (auto-reset)
    xor     r8d, r8d                    ; bInitialState = FALSE
    xor     r9d, r9d                    ; lpName = NULL
    call    CreateEventW
    mov     [gw_event], rax

    ; --- Initial state: gateway idle ---
    mov     dword [gw_state], GW_IDLE

; ========================== EVENT LOOP ===================================
; Three-handle WaitForMultipleObjects:
;   [0] = listen_event   — FD_ACCEPT on listen_sock
;   [1] = client_event   — FD_READ | FD_CLOSE on active client socket
;   [2] = gw_event       — WinHTTP async callback completion
;
; A constant 8-second timeout prevents a slow client from starving the
; event loop indefinitely (WSAEWOULDBLOCK with no further data).
; -------------------------------------------------------------------------
.event_loop:
    ; Build handle array above the 32-byte Win64 callee shadow space.
    mov     rax, [listen_event]
    mov     [rsp+32], rax
    mov     rax, [client_event]
    mov     [rsp+40], rax
    mov     rax, [gw_event]
    mov     [rsp+48], rax

    ; WaitForMultipleObjects(3, &handles, FALSE, CLIENT_READ_MS)
    mov     ecx, 3
    lea     rdx, [rsp+32]
    xor     r8d, r8d                    ; bWaitAll = FALSE
    mov     r9d, CLIENT_READ_MS         ; timeout ms (0=return, -1=INFINITE)
    call    WaitForMultipleObjects
    cmp     eax, WAIT_OBJECT_0
    je      .handle_accept
    cmp     eax, WAIT_OBJECT_0 + 1
    je      .handle_client_read
    cmp     eax, WAIT_OBJECT_0 + 2
    je      .handle_gateway
    cmp     eax, WAIT_TIMEOUT
    je      .handle_client_timeout
    ; WAIT_FAILED: retry
    jmp     .event_loop

; ====================== ACCEPT EVENT =====================================
.handle_accept:
    ; WSAEnumNetworkEvents resets BOTH the event record AND the event handle.
    ; Must loop accept() until WSAEWOULDBLOCK.
    mov     rcx, [listen_sock]
    mov     rdx, [listen_event]
    lea     r8,  [wsaevents]
    call    WSAEnumNetworkEvents
    test    eax, eax
    jnz     .event_fail

.accept_loop:
    xor     edx, edx
    xor     r8d, r8d
    mov     rcx, [listen_sock]
    call    accept
    cmp     rax, INVALID_SOCKET
    jne     .have_client
    ; accept failed — check why
    call    WSAGetLastError
    cmp     eax, WSAEWOULDBLOCK
    je      .event_loop                  ; no more connections
    lea     rcx, [s_accept]
    mov     rdx, S_ACCEPT_LEN
    call    log_err
    jmp     .event_loop

.have_client:
    ; Save client socket
    mov     [client_sock], rax
    inc     qword [req_id]

    ; Clear WSAEventSelect inheritance from listen socket.
    ; WSAEventSelect(sock, NULL, 0) clears the event record so FD_ACCEPT
    ; is no longer bound to this child socket.  It does NOT restore
    ; blocking mode (the socket stays non-blocking; ioctlsocket will be
    ; called after the request is read and before the response is sent).
    ; SO_RCVTIMEO set by apply_timeouts is harmless on non-blocking recv.
    mov     rcx, [client_sock]
    xor     edx, edx
    xor     r8d, r8d
    call    WSAEventSelect

    ; Apply timeouts
    mov     rcx, [client_sock]
    call    apply_timeouts

    ; Reset per-request reader state before associating the client event.
    mov     qword [req_used], 0
    mov     qword [req_header_end], 0
    mov     dword [req_has_cl], 0
    mov     dword [req_has_te], 0
    mov     qword [req_content_length], 0
    mov     dword [req_cl_count], 0
    mov     qword [req_method_ptr], 0
    mov     qword [req_method_len], 0
    mov     qword [req_path_ptr], 0
    mov     qword [req_path_len], 0

    ; Enable event-loop reads: associate client socket with client_event.
    ; This also puts the socket in non-blocking mode (WSAEventSelect property).
    mov     rcx, [client_sock]
    mov     rdx, [client_event]
    mov     r8d, FD_READ | FD_CLOSE
    call    WSAEventSelect

    ; Return to event loop — the first FD_READ fires immediately when
    ; data arrives (which it already has, from the SYN+data bundled by TCP).
    jmp     .event_loop

; ====================== CLIENT READ EVENT ================================
; The level-triggered client_event fires whenever data is available on the
; active client socket.  read_more_request does ONE non-blocking recv and
; returns the outcome code.  We loop within this handler when data is
; available, but never call Sleep or busy-wait — WSAEWOULDBLOCK returns 2
; and we go back to WFMO.
; -------------------------------------------------------------------------
.handle_client_read:
    mov     rcx, [client_sock]
    call    read_more_request

    cmp     eax, 0
    je      .client_complete            ; request fully read → route
    cmp     eax, 1
    je      .handle_client_read         ; recv succeeded, more expected
    cmp     eax, 2
    je      .client_wouldblock          ; no more data right now

    ; >= HTTP_400 — read error
    ; Save error code before WSAEventSelect/ioctlsocket clobber eax.
    mov     [rsp+0x30], eax
    mov     rcx, [client_sock]
    xor     edx, edx
    xor     r8d, r8d
    call    WSAEventSelect
    mov     rcx, [client_sock]
    mov     edx, FIONBIO
    lea     r8,  [fionbio_arg]
    mov     dword [r8], 0
    call    ioctlsocket
    mov     eax, [rsp+0x30]
    call    resp_set_error              ; preserves eax
    jmp     .respond_now                ; http_respond + log_request + close

.client_complete:
    ; Request fully accumulated.  Disassociate client_event from the socket
    ; so the client_event handle stops firing.  Then restore blocking mode
    ; for the response path (http_respond / send call blocking send).
    ; WSAEventSelect(sock, NULL, 0) clears the event record but DOES NOT
    ; restore blocking mode (per MSDN).  Call ioctlsocket(FIONBIO, 0)
    ; afterwards to return the socket to blocking mode.
    mov     rcx, [client_sock]
    xor     edx, edx
    xor     r8d, r8d
    call    WSAEventSelect

    mov     rcx, [client_sock]
    mov     edx, FIONBIO
    lea     r8,  [fionbio_arg]
    mov     dword [r8], 0               ; 0 = blocking mode
    call    ioctlsocket

    ; Sanity-check canary integrity before routing.
    call    debug_canaries_check
    test    eax, eax
    jnz     .canary_err

    ; Save gw_state BEFORE route_request to detect if THIS request
    ; starts an async gateway round.
    mov     eax, [gw_state]
    mov     [rsp+0x28], eax

    ; Route the request
    call    route_request
    mov     [rsp+0x30], eax
    call    debug_canaries_check
    test    eax, eax
    jnz     .canary_err
    mov     rax, [rsp+0x30]

    ; Did THIS request start an async gateway round?
    mov     edx, [gw_state]
    cmp     edx, [rsp+0x28]
    je      .respond_now

    ; Yes — async /chat deferral.
.gateway_started:
    mov     rcx, [client_sock]
    mov     [pending_chat_sock], rcx
    mov     qword [client_sock], 0          ; zero so timeout handler doesn't kill this client
    mov     rcx, [client_event]
    test    rcx, rcx
    jz      .gs_done
    call    ResetEvent                      ; clear stale signal from client socket
.gs_done:
    mov     rax, [req_id]
    mov     [pending_req_id], rax
    mov     rax, [req_content_length]
    mov     [pending_req_clen], rax
    call    gateway_advance
    test    eax, eax
    jnz     .respond_gateway
    ; First submission in flight; event loop will pick up gw_event.
    ; client_sock stays open for the eventual async response.
    jmp     .event_loop

.client_wouldblock:
    ; No more data right now.  Call WSAEnumNetworkEvents to reset the
    ; level-triggered client_event.  This also reports FD_CLOSE if the
    ; client disconnected.
    mov     rcx, [client_sock]
    mov     rdx, [client_event]
    lea     r8,  [wsaevents]
    call    WSAEnumNetworkEvents
    test    eax, eax
    jnz     .event_fail

    ; Check for FD_READ | FD_CLOSE in lNetworkEvents.
    ; If FD_READ is set, more data arrived between the recv and
    ; WSAEnumNetworkEvents — drain it before treating FD_CLOSE as
    ; an incomplete close.
    mov     eax, [wsaevents]            ; lNetworkEvents
    test    eax, FD_READ
    jnz     .handle_client_read         ; data available: try to read it
    test    eax, FD_CLOSE
    jnz     .client_closed              ; FD_CLOSE alone → clean disconnect
    jmp     .event_loop

.client_closed:
    ; Client closed the connection before completing the request.
    lea     rcx, [s_client_close]
    mov     rdx, S_CLIENT_CLOSE_LEN
    call    log_err

    ; Restore blocking mode then respond with error.
    mov     rcx, [client_sock]
    xor     edx, edx
    xor     r8d, r8d
    call    WSAEventSelect
    mov     rcx, [client_sock]
    mov     edx, FIONBIO
    lea     r8,  [fionbio_arg]
    mov     dword [r8], 0
    call    ioctlsocket
    mov     eax, HTTP_400
    call    resp_set_error
    jmp     .respond_now

.handle_client_timeout:
    ; If there's no client socket (server idle or async gateway in flight),
    ; silently return to the event loop without logging.
    cmp     qword [client_sock], 0
    je      .event_loop

    ; Slow client — abort the partial request.
    lea     rcx, [s_client_timeout]
    mov     rdx, S_CLIENT_TIMEOUT_LEN
    call    log_err

    ; Restore blocking mode then respond with error.
    mov     rcx, [client_sock]
    xor     edx, edx
    xor     r8d, r8d
    call    WSAEventSelect
    mov     rcx, [client_sock]
    mov     edx, FIONBIO
    lea     r8,  [fionbio_arg]
    mov     dword [r8], 0
    call    ioctlsocket
    mov     eax, HTTP_400
    call    resp_set_error
    jmp     .respond_now

; ====================== SYNC RESPONSE / CLOSE =============================
.respond_now:
    mov     [rsp+0x30], eax
    mov     rcx, [client_sock]
    mov     edx, eax
    call    http_respond
    mov     rcx, [rsp+0x30]
    mov     rdx, [resp_body_len]
    call    log_request

.respond_and_close:
    ; Graceful close
    mov     rcx, [client_sock]
    mov     edx, SD_SEND
    call    shutdown
    mov     rcx, [client_sock]
    call    closesocket
    mov     qword [client_sock], 0
    ; Reset the manual-reset WSA event so WFMO doesn't spin on the stale
    ; signal from the previous socket's FD_READ/FD_CLOSE.
    mov     rcx, [client_event]
    test    rcx, rcx
    jz      .rac_done
    call    ResetEvent
.rac_done:

    ; Check for more pending connections in the backlog
    jmp     .accept_loop

; ====================== GATEWAY EVENT ====================================
.handle_gateway:
    call    gateway_advance
    test    eax, eax
    jz      .event_loop                  ; still in progress

    ; Gateway completed — respond on pending_chat_sock.
.respond_gateway:
    mov     [rsp+0x30], eax
    mov     rcx, [pending_chat_sock]
    mov     edx, eax
    call    http_respond

    ; --- Restore originating chat metadata for log_request ---
    mov     rax, [req_id]
    mov     [rsp+0x38], rax
    mov     rax, [pending_req_id]
    mov     [req_id], rax
    mov     rax, [pending_req_clen]
    mov     [req_content_length], rax
    lea     rax, [req_post]
    mov     [req_method_ptr], rax
    mov     qword [req_method_len], REQ_POST_LEN
    lea     rax, [req_path_chat]
    mov     [req_path_ptr], rax
    mov     qword [req_path_len], REQ_PATH_CHAT_LEN

    mov     rcx, [rsp+0x30]
    mov     rdx, [resp_body_len]
    call    log_request

    mov     rax, [rsp+0x38]
    mov     [req_id], rax

    ; Graceful close
    mov     rcx, [pending_chat_sock]
    mov     edx, SD_SEND
    call    shutdown
    mov     rcx, [pending_chat_sock]
    call    closesocket
    mov     qword [pending_chat_sock], 0
    mov     qword [client_sock], 0          ; prevent timeout handler false match

    jmp     .event_loop

; ====================== ERROR HELPERS ====================================
.canary_err:
    mov     eax, HTTP_500
    jmp     .respond_now

.event_fail:
    lea     rcx, [s_event]
    mov     rdx, S_EVENT_LEN
    call    log_err
    mov     rcx, 1
    call    net_shutdown
    ud2
