; ===========================================================================
; src/http_read.asm - bounded HTTP request reader state machine (PLAN §2.4)
; ===========================================================================
%include "win64.inc"
%include "winsock.inc"
%include "http.inc"
%include "config.inc"

extern recv_buf
extern req_used
extern req_header_end
extern req_has_cl
extern req_has_te
extern req_content_length
extern req_method_ptr
extern req_method_len
extern req_path_ptr
extern req_path_len
extern mem_find
extern http_parse
extern log_err
extern Sleep

default rel

section .data
crlfcrlf:    db 13,10,13,10
CRLFCRLF_LEN equ $-crlfcrlf
s_recvread:  db "recv_read"
S_RECVREAD_LEN equ $-s_recvread

section .text

; ---------------------------------------------------------------------------
; http_read_request - accumulate one complete HTTP/1.1 request.
; Inputs:  RCX = client socket.
; Outputs: RAX = 0 on complete request, or an HTTP status code on failure.
; Clobbers: volatile + saved rbx,r12.
; Alignment: 3 pushes (rbp,rbx,r12) entry≡8 -> ≡0; sub 32 -> ≡0. OK.
; ---------------------------------------------------------------------------
global http_read_request
http_read_request:
    push    rbp
    mov     rbp, rsp
    push    rbx                     ; socket
    push    r12                     ; scratch (cap / n / expected_total)
    sub     rsp, 32                 ; shadow; aligned
    mov     rbx, rcx
    mov     qword [req_used], 0
    mov     qword [req_header_end], 0
    mov     dword [req_has_cl], 0
    mov     dword [req_has_te], 0
    mov     qword [req_content_length], 0
    mov     qword [req_method_ptr], 0
    mov     qword [req_method_len], 0
    mov     qword [req_path_ptr], 0
    mov     qword [req_path_len], 0
    ; ===================== state: reading headers =====================
.hdr_loop:
    mov     rax, CAP_REQUEST
    sub     rax, [req_used]
    jle     .err_431                ; no room left, still no terminator
%if DEBUG
    cmp     rax, DBG_RECV_CHUNK
    jbe     .hcap
    mov     rax, DBG_RECV_CHUNK
.hcap:
%endif
    mov     r12, rax                ; cap
    mov     rcx, rbx
    lea     rdx, [recv_buf]
    add     rdx, [req_used]
    mov     r8,  r12
    xor     r9d, r9d
    call    recv_retry
    mov     r12d, eax               ; n (MUST capture immediately; calls clobber eax)
    test    r12d, r12d
    js      .recv_err
    je      .peer_closed
    add     [req_used], r12
    ; search CRLFCRLF within [0, req_used)
    lea     rcx, [rel recv_buf]
    mov     rdx, [req_used]
    lea     r8,  [crlfcrlf]
    mov     r9,  CRLFCRLF_LEN
    call    mem_find
    test    rax, rax
    js      .no_term
    add     rax, 4
    mov     [req_header_end], rax   ; body starts here
    jmp     .headers_done
.no_term:
    mov     rax, [req_used]
    cmp     rax, CAP_HEADERS
    jge     .err_431
    jmp     .hdr_loop
.headers_done:
    call    http_parse              ; -> eax = 0 or status code
    test    eax, eax
    jnz     .out                    ; propagate parse error code
    ; ===================== state: reading body =====================
    cmp     dword [req_has_te], 0
    jne     .err_501
    cmp     dword [req_has_cl], 0
    je      .complete               ; no body
    mov     rax, [req_header_end]
    add     rax, [req_content_length]
    jc      .err_413
    cmp     rax, CAP_REQUEST
    ja      .err_413
    mov     r12, rax                ; expected_total
.body_loop:
    mov     rax, [req_used]
    cmp     rax, r12
    jae     .complete
    mov     rax, CAP_REQUEST
    sub     rax, [req_used]
%if DEBUG
    cmp     rax, DBG_RECV_CHUNK
    jbe     .bcap
    mov     rax, DBG_RECV_CHUNK
.bcap:
%endif
    mov     rcx, rbx
    lea     rdx, [recv_buf]
    add     rdx, [req_used]
    mov     r8,  rax
    xor     r9d, r9d
    call    recv_retry
    mov     r11d, eax               ; n
    test    r11d, r11d
    js      .recv_err
    je      .peer_closed
    add     [req_used], r11
    jmp     .body_loop
.complete:
    xor     eax, eax
    jmp     .out
.recv_err:
    lea     rcx, [rel s_recvread]
    mov     rdx, S_RECVREAD_LEN
    call    log_err
    mov     eax, HTTP_400
    jmp     .out
.peer_closed:
    mov     eax, HTTP_400
    jmp     .out
.err_431:
    mov     eax, HTTP_431
    jmp     .out
.err_413:
    mov     eax, HTTP_413
    jmp     .out
.err_501:
    mov     eax, HTTP_501
.out:
    add     rsp, 32
    pop     r12
    pop     rbx
    pop     rbp
    ret

; ===========================================================================
; recv_retry - recv with WSAEWOULDBLOCK retry loop (non-blocking socket compat).
; Inputs:  RCX=socket, RDX=buf, R8=len, R9=flags (same as recv).
; Outputs: RAX = bytes read (positive), 0 (EOF), or -1 (error/timeout).
; Retries up to ~5000 times with 1ms Sleep between attempts.
; ===========================================================================
recv_retry:
    push    rbp
    mov     rbp, rsp
    push    rbx                     ; socket
    push    r12                     ; retry counter
    push    r13                     ; buf
    push    r14                     ; len
    push    r15                     ; flags
    sub     rsp, 40                 ; shadow(32)+alignment
    mov     rbx, rcx
    mov     r12, 5000
    mov     r13, rdx
    mov     r14, r8
    mov     r15, r9
.rr_loop:
    mov     rcx, rbx
    mov     rdx, r13
    mov     r8,  r14
    mov     r9,  r15
    call    recv
    test    eax, eax
    jns     .rr_done
    ; Error — check for WSAEWOULDBLOCK
    call    WSAGetLastError
    cmp     eax, WSAEWOULDBLOCK
    jne     .rr_error
    dec     r12
    jz      .rr_error
    mov     ecx, 1
    call    Sleep
    jmp     .rr_loop
.rr_error:
    mov     eax, -1
.rr_done:
    add     rsp, 40
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret
