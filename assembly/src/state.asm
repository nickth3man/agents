; ===========================================================================
; src/state.asm - static runtime buffers and global state (PLAN §2.7)
; ===========================================================================
%include "win64.inc"
%include "winapi.inc"
%include "winsock.inc"
%include "config.inc"

default rel

section .bss
align 16
global wsadata
wsadata:        resb WSADATA_SIZE        ; WSAStartup output (402 bytes)

global listen_sock
listen_sock:    resq 1                   ; listening SOCKET (server)

global client_sock
client_sock:    resq 1                   ; current accepted SOCKET

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

; --- gateway engine state (PLAN §2.10, Milestone 8) ------------------------
global gw_sock
gw_sock:         resq 1               ; outbound socket to the relay
global gw_used
gw_used:         resq 1               ; bytes accumulated in gateway_resp
global gw_header_end
gw_header_end:   resq 1               ; offset of relay reply body
global gateway_req
gateway_guard_lo: resq 1
gateway_req:     resb CAP_RESP_HDR    ; 512-byte gateway request workspace
global gateway_resp
gateway_resp:    resb CAP_GATEWAY_RESP ; 16 KiB bounded LLM reply
gateway_guard_hi: resq 1

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
