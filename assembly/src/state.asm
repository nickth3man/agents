; ===========================================================================
; src/state.asm - static runtime buffers and global state (PLAN §2.7)
; ===========================================================================
%include "win64.inc"
%include "winapi.inc"
%include "winsock.inc"
%include "config.inc"

; --- New includes for async support -----------------------------------------
%include "engine.inc"

default rel

section .bss
align 16
global wsadata
wsadata:        resb WSADATA_SIZE        ; WSAStartup output (402 bytes)

global listen_sock
listen_sock:    resq 1                   ; listening SOCKET (server)

global req_id
req_id:         resq 1                   ; monotonic request counter (for logs)

global recv_buf
recv_guard_lo:  resq 1
recv_buf:       resb CAP_REQUEST         ; 8 KiB request buffer (headers+body)
recv_guard_hi:  resq 1

global last_wsa
last_wsa:       resd 1               ; most recent WSAGetLastError value

global log_scratch
log_guard_lo:   resq 1
log_scratch:    resb CAP_LOG_LINE        ; 512-byte log-line workspace
log_guard_hi:   resq 1

global dec_scratch
dec_scratch:    resb 16                  ; decimal formatting scratch (u32)

; --- request reader / parser state (PLAN §2.4) -----------------------------
global req_used
req_used:           resq 1               ; bytes accumulated in recv_buf
global req_header_end
req_header_end:     resq 1               ; offset of first body byte (past CRLFCRLF)
global req_content_length
req_content_length: resq 1               ; declared Content-Length value
global req_has_cl
req_has_cl:         resd 1               ; 1 if a Content-Length header is present
global req_has_te
req_has_te:         resd 1               ; 1 if a Transfer-Encoding header is present
global req_cl_count
req_cl_count:       resd 1               ; number of Content-Length headers seen
global req_method_ptr
req_method_ptr:     resq 1
global req_method_len
req_method_len:     resq 1
global req_path_ptr
req_path_ptr:       resq 1
global req_path_len
req_path_len:       resq 1

; --- response builder state (PLAN §2.6) ------------------------------------
global resp_hdr_buf
resp_guard_lo:  resq 1
resp_hdr_buf:   resb CAP_RESP_HDR        ; 512-byte response-header workspace
resp_guard_hi:  resq 1
global resp_body_ptr
resp_body_ptr:   resq 1
global resp_body_len
resp_body_len:   resq 1
global resp_ct_ptr
resp_ct_ptr:     resq 1
global resp_ct_len
resp_ct_len:     resq 1

; --- native OpenRouter gateway state --------------------------------------
global gw_used
gw_used:         resq 1               ; bytes accumulated from WinHTTP
global gateway_req
gateway_guard_lo: resq 1
gateway_req:     resb CAP_GATEWAY_REQ ; request JSON, then decoded model reply
global gateway_resp
gateway_resp:    resb CAP_GATEWAY_RESP ; raw OpenRouter response JSON
gateway_guard_hi: resq 1
global gateway_draft
gateway_draft:   resb CAP_GATEWAY_DRAFT
global gateway_draft2
gateway_draft2:  resb CAP_GATEWAY_DRAFT
global decode_target_ptr
decode_target_ptr: resq 1
global decode_target_cap
decode_target_cap: resq 1
global gateway_api_key
gateway_api_key: resb CAP_API_KEY
global gateway_model
gateway_model:   resb CAP_MODEL
global gateway_headers_w
gateway_headers_w: resw CAP_AUTH_WCHARS

; --- Async gateway state machine (added for event-loop rewrite) ------------
global gw_state
gw_state:        resd 1               ; GW_* constant from config.inc
global gw_stage
gw_stage:        resd 1               ; 0=first solver, 1=checker, 2=final
global gw_event
gw_event:        resq 1               ; auto-reset event (signaled by WinHTTP callback)
global gw_hSession
gw_hSession:     resq 1               ; WinHTTP session handle (reused across 3 calls)
global gw_hConnect
gw_hConnect:     resq 1               ; WinHTTP connection handle (reused)
global gw_hRequest
gw_hRequest:     resq 1               ; WinHTTP request handle (new per stage)
global gw_read_len
gw_read_len:     resd 1               ; bytes from last READ_COMPLETE callback
global gw_err_code
gw_err_code:     resd 1               ; last WinHTTP async error code
global gw_draft1_len
gw_draft1_len:   resq 1               ; length of first solver draft in gateway_draft
global gw_draft2_len
gw_draft2_len:   resq 1               ; length of checker draft in gateway_draft2
global gw_user_msg
gw_user_msg:     resb CAP_CHAT_BODY   ; saved user message (recv_buf may be reused)
global gw_user_msg_len
gw_user_msg_len: resq 1
global gw_model_len
gw_model_len:    resd 1               ; length of string in gateway_model
global gw_key_len
gw_key_len:      resd 1               ; length of string in gateway_api_key
global gw_headers_len
gw_headers_len:  resd 1               ; auth-header WCHAR count (for SendRequest)
global listen_event
listen_event:    resq 1               ; WSA event for listen_sock (FD_ACCEPT)
global wsaevents
wsaevents:       resb WSANETWORKEVENTS_SIZE  ; buffer for WSAEnumNetworkEvents

; --- Per-slot client state arrays (aligned 16) -----------------------------
global client_slots
client_slots:    resb CLIENT_SLOT_SIZE * MAX_CLIENTS  ; aligned 16
global gw_pending_slot
gw_pending_slot: resd 1                               ; slot index awaiting /chat response (-1=none)

section .data
global excl_enable
excl_enable:    dd 1                     ; SO_EXCLUSIVEADDRUSE optval = TRUE

global timeout_ms
timeout_ms:     dd 5000                  ; per-client recv/send timeout (ms)

section .text

; Initialize and verify fixed guards around the critical protocol buffers.
; Both routines are leaf functions and touch no memory outside the guard words.
global debug_canaries_init
debug_canaries_init:
%if DEBUG
    mov     rax, CANARY_VALUE
    mov     [recv_guard_lo], rax
    mov     [recv_guard_hi], rax
    mov     [log_guard_lo], rax
    mov     [log_guard_hi], rax
    mov     [resp_guard_lo], rax
    mov     [resp_guard_hi], rax
    mov     [gateway_guard_lo], rax
    mov     [gateway_guard_hi], rax
%endif
    xor     eax, eax
    ret

global debug_canaries_check
debug_canaries_check:
%if DEBUG
    mov     rax, CANARY_VALUE
    cmp     [recv_guard_lo], rax
    jne     .bad
    cmp     [recv_guard_hi], rax
    jne     .bad
    cmp     [log_guard_lo], rax
    jne     .bad
    cmp     [log_guard_hi], rax
    jne     .bad
    cmp     [resp_guard_lo], rax
    jne     .bad
    cmp     [resp_guard_hi], rax
    jne     .bad
    cmp     [gateway_guard_lo], rax
    jne     .bad
    cmp     [gateway_guard_hi], rax
    jne     .bad
%endif
    xor     eax, eax
    ret
%if DEBUG
.bad:
    mov     eax, 1
    ret
%endif
