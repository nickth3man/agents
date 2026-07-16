; ===========================================================================
; src/state.asm - static runtime buffers and global state (PLAN §2.7)
; ===========================================================================
%include "win64.inc"
%include "winapi.inc"
%include "winsock.inc"
%include "config.inc"

section .bss
align 16
global wsadata
wsadata:        resb WSADATA_SIZE        ; WSAStartup output (402 bytes)

global listen_sock
listen_sock:    resq 1                   ; listening SOCKET (server)

global client_sock
client_sock:    resq 1                   ; current accepted SOCKET

global recv_buf
recv_buf:       resb CAP_REQUEST         ; 8 KiB request buffer (headers+body)

global last_wsa
last_wsa:       resd 1                   ; most recent WSAGetLastError value

global log_scratch
log_scratch:    resb CAP_LOG_LINE        ; 512-byte log-line workspace

global dec_scratch
dec_scratch:    resb 16                  ; decimal formatting scratch (u32)

section .data
global excl_enable
excl_enable:    dd 1                     ; SO_EXCLUSIVEADDRUSE optval = TRUE

global timeout_ms
timeout_ms:     dd 5000                  ; per-client recv/send timeout (ms)
