; ===========================================================================
; src/http_write.asm - HTTP response formatter + sender (PLAN §2.6)
; ===========================================================================
%include "win64.inc"
%include "winsock.inc"
%include "http.inc"
%include "config.inc"

extern send_all
extern copy_bytes
extern u32_to_dec
extern resp_hdr_buf
extern resp_body_ptr
extern resp_body_len
extern resp_ct_ptr
extern resp_ct_len

default rel

section .data
s_http_ver:  db "HTTP/1.1 "
S_HTTP_VER_LEN  equ $-s_http_ver       ; 9
s_sp:         db " "
s_cl_hdr:     db 13,10,"Content-Length: "
S_CL_HDR_LEN  equ $-s_cl_hdr           ; 18
s_ct_hdr:     db 13,10,"Content-Type: "
S_CT_HDR_LEN  equ $-s_ct_hdr           ; 17
s_tail:       db 13,10,"Connection: close",13,10,"Cache-Control: no-store",13,10,13,10
S_TAIL_LEN    equ $-s_tail

; reason phrases
r200: db "OK"
R200_LEN equ $-r200
r400: db "Bad Request"
R400_LEN equ $-r400
r404: db "Not Found"
R404_LEN equ $-r404
r405: db "Method Not Allowed"
R405_LEN equ $-r405
r411: db "Length Required"
R411_LEN equ $-r411
r413: db "Payload Too Large"
R413_LEN equ $-r413
r431: db "Request Header Fields Too Large"
R431_LEN equ $-r431
r500: db "Internal Server Error"
R500_LEN equ $-r500
r501: db "Not Implemented"
R501_LEN equ $-r501
r502: db "Bad Gateway"
R502_LEN equ $-r502
r503: db "Service Unavailable"
R503_LEN equ $-r503

section .text

; ---------------------------------------------------------------------------
; reason_phrase - map HTTP status code to reason-phrase string
; Purpose:        Returns a static, NUL-less reason-phrase string and its
;                 length for the given HTTP status code. Falls back to
;                 "Internal Server Error" (500) for unrecognized codes.
; Inputs:         EAX = HTTP status code (u32)
; Outputs:        RAX = ptr to static phrase (u8*), RDX = phrase length (usize)
; Errors:         none (unknown codes map to 500 phrase)
; Clobbers:       RAX,RDX (volatile return registers)
; Preserves:      RBX,RBP,RDI,RSI,R12-R15 (nonvolatile)
; Locals:         0 (leaf, no frame)
; Max read:       0   Max write: 0 (returns via registers; no pointer derefs)
; ---------------------------------------------------------------------------
reason_phrase:
    cmp     eax, 200
    jne     .n200
    lea     rax, [r200]
    mov     edx, R200_LEN
    ret
.n200:
    cmp     eax, 400
    jne     .n400
    lea     rax, [r400]
    mov     edx, R400_LEN
    ret
.n400:
    cmp     eax, 404
    jne     .n404
    lea     rax, [r404]
    mov     edx, R404_LEN
    ret
.n404:
    cmp     eax, 405
    jne     .n405
    lea     rax, [r405]
    mov     edx, R405_LEN
    ret
.n405:
    cmp     eax, 411
    jne     .n411
    lea     rax, [r411]
    mov     edx, R411_LEN
    ret
.n411:
    cmp     eax, 413
    jne     .n413
    lea     rax, [r413]
    mov     edx, R413_LEN
    ret
.n413:
    cmp     eax, 431
    jne     .n431
    lea     rax, [r431]
    mov     edx, R431_LEN
    ret
.n431:
    cmp     eax, 501
    jne     .n501
    lea     rax, [r501]
    mov     edx, R501_LEN
    ret
.n501:
    cmp     eax, 502
    jne     .n502
    lea     rax, [r502]
    mov     edx, R502_LEN
    ret
.n502:
    cmp     eax, 503
    jne     .n503
    lea     rax, [r503]
    mov     edx, R503_LEN
    ret
.n503:
    lea     rax, [r500]
    mov     edx, R500_LEN
    ret

; ===========================================================================
; http_respond - build and send a complete HTTP/1.1 response
; Purpose:        Builds an HTTP/1.1 response in resp_hdr_buf (status line +
;                 Content-Type + Content-Length + Connection/Cache-Control
;                 headers), then sends headers + body via send_all. Caller
;                 is responsible for closing the socket after return.
; @param[in]      RCX - SOCKET client socket (valid, writable)
; @param[in]      EDX - u16 HTTP status code (200, 400, 404, 405, 411, 413,
;                       431, 500, 501, 502, 503)
; @param[out]     RAX - u32 result: 0=success, 1=send failure
; Inputs:         RCX = client socket (SOCKET), EDX = HTTP status code (u16);
;                 operates on resp_body_ptr/len, resp_ct_ptr/len globals
; Outputs:        RAX = 0 on success, nonzero if send_all failed;
;                 HTTP response sent to client socket
; Errors:         RAX = 1 on send failure (socket error propagated from send_all)
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11
; Preserves:      RBX,RBP,RDI,RSI,R12-R15
; Locals:         32 (shadow only; RBX=socket, R12=status, R13=header cursor,
;                 R14=length scratch saved via push)
; Max read:       resp_ct_len bytes from [resp_ct_ptr], resp_body_len bytes
;                 from [resp_body_ptr], up to CAP_RESP_HDR from [resp_hdr_buf]
;                 (via send_all); static strings from .data
; Max write:      up to CAP_RESP_HDR bytes to [resp_hdr_buf] (via copy_bytes)
; Precond:        resp_body_ptr/len, resp_ct_ptr/len set by caller
;                 (route_request or resp_set_error); resp_hdr_buf valid for
;                 CAP_RESP_HDR bytes
; Stack:          SHADOW 32 + 4 nonvolatile saves (32 bytes),
;                 RSP 16-aligned at CALL
; Modified:       RAX,RCX,RDX,R8,R9,R10,R11
; Initial inputs to registers: socket->RCX, status->EDX
; Register assignments:
;   status_line_phase: R13=resp_hdr_buf cursor; RCX=cursor, RDX=len,
;                      R8=static_str for copy_bytes; EAX=status for
;                      reason_phrase; R14=reason_len
;   headers_phase:     R13=resp_hdr_buf cursor; copy_bytes for Content-Type
;                      header + value, then Content-Length header + value
;   body_phase:        RCX=socket, RDX=resp_hdr_buf/body_ptr,
;                      R8=length for send_all; RAX=0/1 result
; ===========================================================================
global http_respond
http_respond:
    push    rbp
    mov     rbp, rsp
    push    rbx                     ; socket
    push    r12                     ; status code
    push    r13                     ; header cursor
    push    r14                     ; length scratch
    sub     rsp, 32                 ; shadow; aligned
    mov     rbx, rcx
    mov     r12d, edx
    lea     r13, [resp_hdr_buf]
    ; "HTTP/1.1 "
    mov     rcx, r13
    lea     r8,  [s_http_ver]
    mov     edx, S_HTTP_VER_LEN
    call    copy_bytes
    add     r13, S_HTTP_VER_LEN
    ; status code decimal
    mov     ecx, r12d
    mov     rdx, r13
    mov     r8,  8
    call    u32_to_dec
    add     r13, rax
    ; " "
    mov     rcx, r13
    lea     r8,  [s_sp]
    mov     edx, 1
    call    copy_bytes
    add     r13, 1
    ; reason phrase
    mov     eax, r12d
    call    reason_phrase           ; rax=ptr, rdx=len
    mov     r14, rdx                ; save len
    mov     r8,  rax
    mov     rcx, r13
    mov     rdx, r14
    call    copy_bytes
    add     r13, r14
    ; Content-Type header
    mov     rcx, r13
    lea     r8,  [s_ct_hdr]
    mov     edx, S_CT_HDR_LEN
    call    copy_bytes
    add     r13, S_CT_HDR_LEN
    mov     rcx, r13
    mov     r8,  [resp_ct_ptr]
    mov     rdx, [resp_ct_len]
    call    copy_bytes
    add     r13, [resp_ct_len]
    ; Content-Length header
    mov     rcx, r13
    lea     r8,  [s_cl_hdr]
    mov     edx, S_CL_HDR_LEN
    call    copy_bytes
    add     r13, S_CL_HDR_LEN
    mov     ecx, [resp_body_len]    ; value (low 32 bits; body len is small)
    mov     rdx, r13
    mov     r8,  8
    call    u32_to_dec
    add     r13, rax
    ; tail (Connection: close + Cache-Control + blank line)
    mov     rcx, r13
    lea     r8,  [s_tail]
    mov     edx, S_TAIL_LEN
    call    copy_bytes
    add     r13, S_TAIL_LEN
    ; --- send headers ---
    mov     rcx, rbx
    lea     rdx, [resp_hdr_buf]
    mov     r8,  r13
    sub     r8,  rdx                ; header length
    call    send_all
    test    eax, eax
    jnz     .fail
    ; --- send body ---
    mov     rcx, rbx
    mov     rdx, [resp_body_ptr]
    mov     r8,  [resp_body_len]
    call    send_all
    test    eax, eax
    jnz     .fail
    xor     eax, eax
    jmp     .out
.fail:
    mov     eax, 1
.out:
    add     rsp, 32
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret
