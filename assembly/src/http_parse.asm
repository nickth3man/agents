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
extern parse_u32

default rel

section .data
crlf2:    db 13,10
sp_byte:  db 32
cl_name:  db "content-length:"
CL_NAME_LEN  equ $-cl_name          ; 15
te_name:  db "transfer-encoding:"
TE_NAME_LEN  equ $-te_name          ; 18
dbg_len:  db "len"
DBG_LEN_LEN equ $-dbg_len

section .text

; ---------------------------------------------------------------------------
; http_parse - parse a complete request held in recv_buf.
; ---------------------------------------------------------------------------
; Precondition: req_header_end set (body start / past CRLFCRLF).
; Inputs:  none (operates on globals).
; Outputs: RAX = 0 on success, else HTTP status (400/413/501).
;          Sets method/path spans and Content-Length / Transfer-Encoding flags.
; Clobbers: volatile + saved rbx,rsi,rdi,r12,r13,r15.
; Alignment: 7 pushes (rbp,rbx,rsi,rdi,r12,r13,r15) entry≡8 -> ≡0; sub 32 -> ≡0.
; ---------------------------------------------------------------------------
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
    ; --- version span = [sp2+1, rli); must start with "HTTP/" ---
    mov     rax, rbx
    sub     rax, r13
    dec     rax                     ; version len
    cmp     rax, 5
    jb      .err400
    lea     r10, [rel recv_buf]
    add     r10, r13
    inc     r10                     ; version ptr = recv_buf + sp2 + 1
    mov     cl, [r10+0]  ; cmp cl,'H' is implicit in next compare
    cmp     cl, 'H'
    jne     .err400
    mov     cl, [r10+1]
    cmp     cl, 'T'
    jne     .err400
    mov     cl, [r10+2]
    cmp     cl, 'T'
    jne     .err400
    mov     cl, [r10+3]
    cmp     cl, 'P'
    jne     .err400
    mov     cl, [r10+4]
    cmp     cl, '/'
    jne     .err400
    ; ===================== Content-Length =====================
    lea     rcx, [rel recv_buf]
    mov     rdx, [req_header_end]
    lea     r8,  [cl_name]
    mov     r9,  CL_NAME_LEN
    call    mem_find_ci
    test    rax, rax
    js      .check_te
    mov     r15, rax                ; CL header offset
    ; duplicate check: another CL after this header
    ; duplicate check: another CL after this header
    ; duplicate check: another CL after this header
    lea     rcx, [rel recv_buf]
    add     rcx, r15
    add     rcx, CL_NAME_LEN        ; hay = recv_buf + cl_off + CL_NAME_LEN
    mov     rdx, [req_header_end]
    sub     rdx, r15
    sub     rdx, CL_NAME_LEN
    lea     r8,  [cl_name]
    mov     r9,  CL_NAME_LEN
    call    mem_find_ci
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
    lea     rcx, [rel recv_buf]
    mov     rdx, [req_header_end]
    lea     r8,  [te_name]
    mov     r9,  TE_NAME_LEN
    call    mem_find_ci
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
