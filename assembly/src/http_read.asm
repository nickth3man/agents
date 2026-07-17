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
extern mem_find
extern http_parse
extern log_err

default rel

section .data
crlfcrlf:    db 13,10,13,10
CRLFCRLF_LEN equ $-crlfcrlf
s_recvread:  db "recv_read"
S_RECVREAD_LEN equ $-s_recvread
section .text

; ===========================================================================
; read_more_request - incremental non-blocking request reader (state machine)
; Purpose:        Does ONE non-blocking recv into recv_buf, accumulates bytes
;                 into [recv_buf+req_used], detects CRLFCRLF header terminator,
;                 transitions through ST_HEADERS → ST_BODY → ST_COMPLETE, and
;                 enforces CAP_REQUEST/CAP_HEADERS. Never retries with Sleep.
;                 Caller must WSAEnumNetworkEvents after return to reset the
;                 level-triggered event and detect FD_CLOSE.
; @param[in]      RCX - SOCKET client socket (valid, non-zero)
; @param[out]     RAX - u32 result: 0=complete, 1=more, 2=wouldblock, >=400 err
; Inputs:         RCX = client socket (SOCKET)
; Precond:        req_used, req_header_end reflect current accumulation;
;                 recv_buf valid for CAP_REQUEST bytes
; Outputs:        RAX = 0 (request complete — route it)
;                       1 (more data needed — recv returned bytes but request
;                          not yet complete; level-triggered event stays
;                          signaled for subsequent WFMO wake)
;                       2 (WSAEWOULDBLOCK — no more data right now; caller
;                          should WSAEnumNetworkEvents + return to event loop)
;                  >= 400 (HTTP error — clean up and respond)
; Errors:         RAX in {400,413,431,501} on protocol/recv error
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11
; Preserves:      RBX,RBP,RDI,RSI,R12-R15
; Locals:         32 (shadow only; RBX=socket, RSI=n saved via push,
;                 frame via push rbp + mov rbp,rsp + sub 32)
; Max read:       (CAP_REQUEST - req_used) bytes via recv into
;                 [recv_buf + req_used]; up to req_used bytes via mem_find
; Max write:      (CAP_REQUEST - req_used) bytes into recv_buf
; Stack:          SHADOW 32 + 2 pushes (RBX,RSI), RSP 16-aligned at CALL
; Modified:       RAX,RCX,RDX,R8,R9,R10,R11
; Initial inputs to registers: socket->RCX
; Register assignments:
;   recv_phase:   RBX=socket, RCX=socket, RDX=recv_buf+offset,
;                 R8=remaining, R9=0; RSI=recv result after WSARecv
;   scan_phase:   RCX=recv_buf, RDX=req_used, R8=crlfcrlf, R9=4;
;                 RAX=offset from mem_find
;   boundary_phase: RAX=expected_total, compare to req_used;
;                   RAX=0/1/>=400 result
; ===========================================================================
global read_more_request
read_more_request:
    push    rbp
    mov     rbp, rsp
    push    rbx                     ; socket
    push    rsi                     ; n (bytes read)
    sub     rsp, 32                 ; shadow; 3 pushes → RSP≡0 before each CALL
    mov     rbx, rcx                ; save socket

    ; --- Safety guard: reject zero/invalid socket ---
    test    rcx, rcx
    jz      .recv_err

    ; --- Capacity check ---
    mov     rax, CAP_REQUEST
    sub     rax, [rel req_used]
    jle     .err_431

    ; --- Try one non-blocking recv ---
    mov     rcx, rbx
    lea     rdx, [rel recv_buf]
    add     rdx, [rel req_used]
    mov     r8,  rax                ; remaining buffer capacity
    xor     r9d, r9d                ; flags = 0
    call    recv
    mov     esi, eax                ; save recv result

    test    esi, esi
    js      .check_wouldblock
    je      .peer_closed

    ; --- Accumulate bytes ---
    mov    eax, esi
    add    [rel req_used], rax

    ; --- Check if headers are complete yet ---
    cmp    qword [rel req_header_end], 0
    jne    .check_body

    ; Search for CRLFCRLF in [0, req_used)
    lea    rcx, [rel recv_buf]
    mov    rdx, [rel req_used]
    lea    r8,  [rel crlfcrlf]
    mov    r9,  CRLFCRLF_LEN
    call   mem_find
    test   rax, rax
    js     .check_hdr_limit

    ; Headers found: set req_header_end = offset + 4, then parse.
    add    rax, 4
    mov    [rel req_header_end], rax
    call   http_parse
    test   eax, eax
    jnz    .out                    ; propagate parse error

    ; Fall through to check if body is expected.
    ; req_has_cl / req_has_te have been set by http_parse.

.check_body:
    cmp    dword [rel req_has_te], 0
    jne    .err_501
    cmp    dword [rel req_has_cl], 0
    je     .complete               ; no body → request done

    ; Content-Length present: check if body fully received.
    mov    rax, [rel req_header_end]
    add    rax, [rel req_content_length]
    jc     .err_413
    cmp    rax, CAP_REQUEST
    ja     .err_413
    mov    rcx, [rel req_used]
    cmp    rcx, rax
    jae    .complete               ; >= expected_total → done
    ; Need more body data
    mov    eax, 1                   ; more needed
    jmp    .out

.check_hdr_limit:
    ; No CRLFCRLF found yet. Check if headers exceeded maximum.
    mov    rax, [rel req_used]
    cmp    rax, CAP_HEADERS
    jge    .err_431
    mov    eax, 1                  ; more data needed
    jmp    .out

.check_wouldblock:
    call   WSAGetLastError
    cmp    eax, WSAEWOULDBLOCK
    jne    .recv_err
    xor    eax, eax
    add    eax, 2                  ; RAX = 2 (wouldblock)
    jmp    .out

.complete:
    xor    eax, eax                ; RAX = 0 (request complete)
    jmp    .out

.err_431:
    mov    eax, HTTP_431
    jmp    .out
.err_501:
    mov    eax, HTTP_501
    jmp    .out
.err_413:
    mov    eax, HTTP_413
    jmp    .out
.peer_closed:
    mov    eax, HTTP_400
    jmp    .out
.recv_err:
    lea    rcx, [rel s_recvread]
    mov    rdx, S_RECVREAD_LEN
    call   log_err
    mov    eax, HTTP_400
.out:
    add    rsp, 32
    pop    rsi
    pop    rbx
    pop    rbp
    ret
