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
extern http_read_request
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
extern req_id
extern req_content_length
extern req_method_ptr
extern req_method_len
extern req_path_ptr
extern req_path_len
extern resp_body_len
extern debug_canaries_init
extern debug_canaries_check
extern gw_state
extern pending_chat_sock
extern pending_req_id
extern pending_req_clen
extern listen_event
extern gw_event
extern wsaevents
extern log_err
extern WSAGetLastError

section .data
banner:
    db  "[startup] build=", BUILD_ID, " listen=127.0.0.1:8080", 10
BANNER_LEN  equ $-banner
s_accept:   db "accept"
S_ACCEPT_LEN equ $-s_accept
s_event:    db "WSAEnumNetworkEvents"
S_EVENT_LEN equ $-s_event
req_post:   db "POST"
REQ_POST_LEN equ $-req_post
req_path_chat: db "/chat"
REQ_PATH_CHAT_LEN equ $-req_path_chat

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

    ; --- Create auto-reset event for gateway completion ---
    ; Auto-reset: WFMO automatically resets the event when it returns,
    ; preventing busy-loops from stale signals.
    xor     ecx, ecx                    ; lpEventAttributes = NULL
    xor     edx, edx                    ; bManualReset = FALSE (auto-reset)
    xor     r8d, r8d                    ; bInitialState = FALSE
    xor     r9d, r9d                    ; lpName = NULL
    call    CreateEventW
    mov     [gw_event], rax

    ; --- Initial state: gateway idle ---
    mov     dword [gw_state], GW_IDLE

; ========================== EVENT LOOP ===================================
.event_loop:
    ; Keep the handle array above the 32-byte Win64 callee shadow space.
    mov     rax, [listen_event]
    mov     [rsp+32], rax
    mov     rax, [gw_event]
    mov     [rsp+40], rax

    ; WaitForMultipleObjects(2, &handles, FALSE, INFINITE)
    mov     ecx, 2
    lea     rdx, [rsp+32]
    xor     r8d, r8d                    ; bWaitAll = FALSE
    xor     r9d, r9d                    ; dwMilliseconds (0 means RETURN, use -1 for INFINITE)
    dec     r9                          ; 0xFFFFFFFF = INFINITE
    call    WaitForMultipleObjects
    cmp     eax, WAIT_OBJECT_0
    je      .handle_accept
    cmp     eax, WAIT_OBJECT_0 + 1
    je      .handle_gateway
    ; WAIT_FAILED: log and retry
    ; (could log here but for now just retry)
    jmp     .event_loop

; ====================== ACCEPT EVENT =====================================
.handle_accept:
    ; WSAEnumNetworkEvents resets BOTH the Winsock event record AND the
    ; event handle, which is essential for correct edge detection.
    ; Must loop accept() until WSAEWOULDBLOCK.
    mov     rcx, [listen_sock]
    mov     rdx, [listen_event]
    lea     r8,  [wsaevents]
    call    WSAEnumNetworkEvents
    test    eax, eax
    jnz     .event_fail                  ; WSAEnumNetworkEvents failed

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
    ; Save client socket in the standard global (for existing code)
    mov     [client_sock], rax
    inc     qword [req_id]
    ; accepted sockets inherit this. Per MSDN, ioctlsocket(FIONBIO) is
    ; ineffective on sockets that inherited the mode from WSAEventSelect.
    ; The only reliable restore is to call WSAEventSelect with 0 events:
    ;   WSAEventSelect(sock, NULL, 0)
    ; This clears the socket's event record and returns it to blocking mode.
    mov     rcx, [client_sock]
    xor     edx, edx                    ; hEventObject = NULL
    xor     r8d, r8d                    ; lNetworkEvents = 0
    call    WSAEventSelect

    ; Apply timeouts (reload socket — RAX was clobbered by ioctlsocket)
    mov     rcx, [client_sock]
    call    apply_timeouts

    ; Read and parse the request
    mov     rcx, [client_sock]
    call    http_read_request
    mov     [rsp+0x30], rax              ; save read result
    call    debug_canaries_check
    test    eax, eax
    jnz     .canary_err
    mov     rax, [rsp+0x30]
    test    eax, eax
    jnz     .read_err

    ; Save gw_state BEFORE route_request to detect if THIS request
    ; started a gateway round (vs a previous request still in flight).
    mov     eax, [gw_state]
    mov     [rsp+0x28], eax

    ; Route the request
    call    route_request                ; eax = status; sets resp_* for normal routes
    mov     [rsp+0x30], rax
    call    debug_canaries_check
    test    eax, eax
    jnz     .canary_err
    mov     rax, [rsp+0x30]

    ; Did THIS request start an async gateway round?
    ; (Compare current gw_state to the saved pre-route value.
    ;  A previous gateway still in flight won't match because gateway_start
    ;  is the only thing that changes gw_state during route_request.)
    mov     edx, [gw_state]
    cmp     edx, [rsp+0x28]
    je      .respond_now                 ; no change → normal sync response

.gateway_started:
    ; Save the client socket for the eventual async response
    mov     rcx, [client_sock]
    mov     [pending_chat_sock], rcx
    ; Save originating chat metadata so that .respond_gateway can
    ; restore it before log_request, avoiding corruption from later
    ; requests (e.g. /version polls) that overwrite req_* globals.
    mov     rax, [req_id]
    mov     [pending_req_id], rax
    mov     rax, [req_content_length]
    mov     [pending_req_clen], rax
    ; Kick off the first gateway advancement (starts WinHTTP chain)
    call    gateway_advance

    ; If gateway_advance returned immediately with success/error (not async),
    ; respond now. Otherwise wait for gw_event.
    test    eax, eax
    jnz     .respond_gateway

    ; Still in progress — wait for gw_event to fire.
    ; The client socket stays open; event loop will respond later.
    ; Check for more pending connections in the backlog first.
    jmp     .accept_loop

.respond_now:
    mov     [rsp+0x30], eax              ; save status across http_respond
    mov     rcx, [client_sock]
    mov     edx, eax
    call    http_respond
    ; Structured request log
    mov     rcx, [rsp+0x30]
    mov     rdx, [resp_body_len]
    call    log_request
    ; Graceful close
    mov     rcx, [client_sock]
    mov     edx, SD_SEND
    call    shutdown
    mov     rcx, [client_sock]
    call    closesocket
    ; Check for more pending connections in the backlog
    jmp     .accept_loop

; ====================== GATEWAY EVENT ====================================
.handle_gateway:
    ; Advance the gateway state machine (one async step completed)
    call    gateway_advance

    ; RAX = 0 (in progress), 200 (done OK), 502 (error)
    test    eax, eax
    jz      .event_loop                  ; still in progress

    ; Gateway completed (success or failure). resp_* globals are set.
    ; Send response to the pending chat client.
.respond_gateway:
    mov     [rsp+0x30], eax             ; save status across http_respond
    mov     rcx, [pending_chat_sock]
    mov     edx, eax                     ; status (200 or 502)
    call    http_respond

    ; --- Restore originating chat metadata for log_request ---
    ; Later requests (e.g. /version polls) may have overwritten
    ; req_id, req_content_length, req_method_*, and req_path_*
    ; since the original /chat was accepted.  Restore them here so
    ; log_request emits the correct id, method=POST, path=/chat,
    ; and input length.
    mov     rax, [req_id]
    mov     [rsp+0x38], rax             ; preserve latest monotonic request id
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

    ; Structured request log
    mov     rcx, [rsp+0x30]             ; status (reload — http_respond clobbered eax)
    mov     rdx, [resp_body_len]
    call    log_request

    ; Restore the latest id observed before the temporary chat-log override.
    ; This may be several requests beyond pending_req_id when /version was
    ; polled repeatedly while the chat was in flight.
    mov     rax, [rsp+0x38]
    mov     [req_id], rax

    ; Graceful close
    mov     rcx, [pending_chat_sock]
    mov     edx, SD_SEND
    call    shutdown
    mov     rcx, [pending_chat_sock]
    call    closesocket
    mov     qword [pending_chat_sock], 0

    jmp     .event_loop

; ====================== ERROR HELPERS ====================================
.canary_err:
    mov     eax, HTTP_500
    jmp     .respond_error
.read_err:
    ; resp_set_error uses the code in eax and sets response globals
    call    resp_set_error
    ; eax is preserved as the status code
    jmp     .respond_now

.event_fail:
    lea     rcx, [s_event]
    mov     rdx, S_EVENT_LEN
    call    log_err
    mov     rcx, 1
    call    net_shutdown
    ud2

; Legacy label: respond with error body (preserves eax)
.respond_error:
    call    resp_set_error
    jmp     .respond_now
