; ===========================================================================
; src/http_parse.asm - request line + framing parser (PLAN §2.4, §2.6, §7.7)
; ===========================================================================
%include "win64.inc"
%include "http.inc"
%include "config.inc"

extern recv_buf
extern req_used
extern req_header_end
extern req_content_length
extern req_has_cl
extern req_has_te
extern req_method_ptr
extern req_method_len
extern req_path_ptr
extern req_path_len
extern mem_find
extern mem_find_ci
extern bytes_eq
extern parse_u32

default rel

section .data
crlf2:    db 13,10
sp_byte:  db 32
cl_name:  db "content-length:"
CL_NAME_LEN  equ $-cl_name          ; 15
te_name:  db "transfer-encoding:"
TE_NAME_LEN  equ $-te_name          ; 18
http11:   db "HTTP/1.1"
HTTP11_LEN equ $-http11

section .text

; ---------------------------------------------------------------------------
; find_header_ci - find a case-insensitive header name at a line boundary
; Purpose:        Scans the recv_buf from a start offset up to req_header_end,
;                 looking for a header whose name matches the given string
;                 case-insensitively. Validates that the match starts at a line
;                 boundary (preceded by CRLF or at offset 0).
; Inputs:         RCX=name ptr (u8*), RDX=name len (usize),
;                 R8=start offset (usize into recv_buf)
; Outputs:        RAX=offset of match in recv_buf, or -1 if not found
; Errors:         RAX=-1 sentinel (not found)
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11 (volatile; calls mem_find_ci)
; Preserves:      RBX,RBP,RDI,RSI,R12-R15 (nonvolatile; RBX=name ptr,
;                 RSI=name len, R12=cursor, R13=header end)
; Locals:         32 (shadow only; rbx,rsi,r12,r13 saved via push)
; Max read:       up to (req_header_end - R8) bytes from [recv_buf + R8],
;                 bounded by CAP_HEADERS (via mem_find_ci)
; Max write:      0 (no writes through pointer args)
; Precond:        req_header_end valid; recv_buf populated; R8 <= req_header_end
; ---------------------------------------------------------------------------
find_header_ci:
    push    rbp
    mov     rbp, rsp
    push    rbx                     ; name ptr
    push    rsi                     ; name len
    push    r12                     ; search cursor
    push    r13                     ; header end
    sub     rsp, 32
    mov     rbx, rcx
    mov     rsi, rdx
    mov     r12, r8
    mov     r13, [req_header_end]
.search:
    cmp     r12, r13
    jae     .not_found
    lea     rcx, [rel recv_buf]
    add     rcx, r12
    mov     rdx, r13
    sub     rdx, r12
    mov     r8, rbx
    mov     r9, rsi
    call    mem_find_ci
    test    rax, rax
    js      .not_found
    add     r12, rax
    cmp     r12, 2
    jb      .advance
    lea     r10, [rel recv_buf]
    cmp     byte [r10 + r12 - 2], CR
    jne     .advance
    cmp     byte [r10 + r12 - 1], LF
    jne     .advance
    mov     rax, r12
    jmp     .out
.advance:
    inc     r12
    jmp     .search
.not_found:
    mov     rax, -1
.out:
    add     rsp, 32
    pop     r13
    pop     r12
    pop     rsi
    pop     rbx
    pop     rbp
    ret

; ===========================================================================
; http_parse - parse complete HTTP request line + framing headers (state machine)
; Purpose:        Parses the first line (METHOD /path HTTP/1.1) and headers
;                 (Content-Length, Transfer-Encoding) from recv_buf. Sets
;                 req_method_ptr/len, req_path_ptr/len, and the content-length/
;                 transfer-encoding flags. Validates framing rules: no duplicate
;                 CL, no CL+TE, no TE without CL, body <= CAP_CHAT_BODY.
; @param[in]      none (operates on recv_buf, req_used, req_header_end globals)
; @param[out]     RAX - u32 HTTP status: 0=success, >=400=error
; Inputs:         none; operates on recv_buf, req_header_end, req_used globals
; Outputs:        RAX = 0 on success, else HTTP error status;
;                 req_method_ptr/len, req_path_ptr/len, req_content_length,
;                 req_has_cl, req_has_te globals set
; Errors:         RAX in {400,413,501} on malformed/oversized request
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11
; Preserves:      RBX,RBP,RDI,RSI,R12-R15
; Locals:         32 (shadow only; RBX=rli, RSI=cursor, RDI=value_start,
;                 R12=sp1, R13=sp2, R15=cl_offset saved via push)
; Max read:       up to req_header_end (bounded by CAP_HEADERS) bytes from
;                 [recv_buf] for parsing; CAP_CHAT_BODY from req_content_length
;                 boundary check; up to CAP_HEADERS via find_header_ci
; Max write:      0 (writes to req_* globals, not through pointer args)
; Precond:        req_header_end set to offset past CRLFCRLF; recv_buf
;                 populated up to req_header_end; req_used valid
; Stack:          SHADOW 32 + 6 nonvolatile saves (48 bytes),
;                 RSP 16-aligned at CALL
; Modified:       RAX,RCX,RDX,R8,R9,R10,R11
; Initial inputs to registers: none (operates on globals)
; Register assignments:
;   method_phase: RCX=recv_buf, RDX=header_end for mem_find(CRLF);
;                 RBX=rli, R12=sp1, R13=sp2; sets req_method*/req_path*
;   path_phase:   RCX=req_path_ptr, validates '/'; checks HTTP/1.1 via bytes_eq
;   headers_phase: RCX=cl_name, RDX=CL_NAME_LEN, R8=0 for find_header_ci;
;                  R15=CL header offset; RSI=cursor, RDI=value_start for
;                  parse_u32; then same pattern for TE check
;   framing_phase: RAX=content_length vs CAP_CHAT_BODY; RAX=0 or HTTP error
; ===========================================================================
global http_parse
http_parse:
    push    rbp
    mov     rbp, rsp
    push    rbx                     ; request-line end offset (rli)
    push    rsi                     ; value cursor
    push    rdi                     ; value start
    push    r12                     ; sp1
    push    r13                     ; sp2
    push    r15                     ; CL header offset
    sub     rsp, 32                 ; shadow; aligned
    mov     dword [req_has_cl], 0
    mov     dword [req_has_te], 0
    mov     qword [req_content_length], 0
    ; --- end of request line = first CRLF within [0, header_end) ---
    lea     rcx, [rel recv_buf]
    mov     rdx, [req_header_end]
    lea     r8,  [crlf2]
    mov     r9,  2
    call    mem_find
    test    rax, rax
    js      .err400
    mov     rbx, rax                ; rli
    ; --- sp1 = first SP in [0, rli) ---
    lea     rcx, [rel recv_buf]
    mov     rdx, rbx
    lea     r8,  [sp_byte]
    mov     r9,  1
    call    mem_find
    test    rax, rax
    js      .err400
    mov     r12, rax                ; sp1
    ; --- sp2 = next SP in (sp1+1, rli) ---
    lea     rcx, [rel recv_buf]
    add     rcx, r12
    inc     rcx                     ; hay = recv_buf + sp1 + 1
    mov     rdx, rbx
    sub     rdx, r12
    dec     rdx                     ; len = rli - sp1 - 1
    lea     r8,  [sp_byte]
    mov     r9,  1
    call    mem_find
    test    rax, rax
    js      .err400
    add     rax, r12
    inc     rax                     ; sp2 absolute offset
    mov     r13, rax
    ; --- method span = [0, sp1) ---
    lea     rax, [recv_buf]
    mov     [req_method_ptr], rax
    mov     rax, r12
    mov     [req_method_len], rax
    test    rax, rax
    jz      .err400
    ; --- path span = [sp1+1, sp2) ---
    lea     rax, [rel recv_buf]
    add     rax, r12
    inc     rax                     ; path ptr = recv_buf + sp1 + 1
    mov     [req_path_ptr], rax
    mov     rax, r13
    sub     rax, r12
    dec     rax                     ; path len = sp2 - sp1 - 1
    mov     [req_path_len], rax
    cmp     rax, 0
    jle     .err400
    ; path must begin with '/'
    mov     rcx, [req_path_ptr]
    mov     cl, [rcx]
    cmp     cl, '/'
    jne     .err400
    ; --- version span = [sp2+1, rli); accept exactly HTTP/1.1 ---
    lea     rcx, [rel recv_buf]
    add     rcx, r13
    inc     rcx                     ; version ptr = recv_buf + sp2 + 1
    mov     rdx, rbx
    sub     rdx, r13
    dec     rdx                     ; version len = rli - sp2 - 1
    lea     r8, [rel http11]        ; expected string
    mov     r9, HTTP11_LEN          ; expected len
    call    bytes_eq
    test    rax, rax
    jz      .err400
    ; ===================== Content-Length =====================
    lea     rcx, [cl_name]
    mov     rdx, CL_NAME_LEN
    xor     r8d, r8d
    call    find_header_ci
    test    rax, rax
    js      .check_te
    mov     r15, rax                ; CL header offset
    ; duplicate check: another CL after this header
    lea     rcx, [cl_name]
    mov     rdx, CL_NAME_LEN
    lea     r8, [r15 + CL_NAME_LEN]
    call    find_header_ci
    test    rax, rax
    jns     .err400                 ; second CL -> 400 (even if equal)
    ; parse value: [r15+CL_NAME_LEN .. CR)
    lea     rsi, [rel recv_buf]
    add     rsi, r15
    add     rsi, CL_NAME_LEN
    mov     rdi, rsi
.find_cr:
    mov     cl, [rsi]
    cmp     cl, CR
    je      .found_cr
    cmp     cl, LF
    je      .found_cr
    inc     rsi
    jmp     .find_cr
.found_cr:
    ; trim leading spaces from rdi
.trim:
    cmp     rdi, rsi
    jae     .err400                 ; empty value
    mov     cl, [rdi]
    cmp     cl, SP
    jne     .trimmed
    inc     rdi
    jmp     .trim
.trimmed:
    mov     rcx, rdi
    mov     rdx, rsi
    sub     rdx, rdi
    call    parse_u32               ; rax=value, CF on error
    jc      .err400
    mov     [req_content_length], rax
    mov     dword [req_has_cl], 1
.check_te:
    lea     rcx, [te_name]
    mov     rdx, TE_NAME_LEN
    xor     r8d, r8d
    call    find_header_ci
    test    rax, rax
    js      .framing_done
    mov     dword [req_has_te], 1
.framing_done:
    mov     eax, [req_has_cl]
    test    eax, eax
    jnz     .have_cl
    cmp     dword [req_has_te], 0
    jne     .err501                 ; TE without CL -> 501
    xor     eax, eax                ; no body
    jmp     .pout
.have_cl:
    cmp     dword [req_has_te], 0
    jne     .err400                 ; CL + TE -> 400
    mov     rax, [req_content_length]
    cmp     rax, CAP_CHAT_BODY
    ja      .err413
    xor     eax, eax
    jmp     .pout
.err413:
    mov     eax, HTTP_413
    jmp     .pout
.err501:
    mov     eax, HTTP_501
    jmp     .pout
.err400:
    mov     eax, HTTP_400
.pout:
    add     rsp, 32
    pop     r15
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbx
    pop     rbp
    ret
