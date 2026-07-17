; ===========================================================================
; src/router.asm - exact endpoint + method dispatch (PLAN §2.6, Milestone 4)
; ===========================================================================
%include "win64.inc"
%include "http.inc"
%include "engine.inc"

extern req_method_ptr
extern req_method_len
extern req_path_ptr
extern req_path_len
extern req_has_cl
extern req_header_end
extern req_content_length
extern recv_buf
extern resp_body_ptr
extern resp_body_len
extern resp_ct_ptr
extern resp_ct_len
extern bytes_eq
extern chat_html
extern chat_html_len
extern build_id
extern build_id_len
extern health_json
extern health_json_len
extern gateway_start
extern gw_state
%ifdef DEV_MODE
extern load_index_html
%endif

default rel

section .data
s_get:     db "GET"
s_post:    db "POST"
s_root:    db "/"
s_version: db "/version"
s_health:  db "/health"
s_chat:    db "/chat"
ct_html:   db "text/html; charset=utf-8"
CT_HTML_LEN   equ $-ct_html
ct_ascii:  db "text/plain; charset=us-ascii"
CT_ASCII_LEN  equ $-ct_ascii
ct_text:   db "text/plain; charset=utf-8"
CT_TEXT_LEN   equ $-ct_text
ct_json:   db "application/json"
CT_JSON_LEN   equ $-ct_json
err_body:  db "error"
ERR_BODY_LEN  equ $-err_body

section .text

; ---------------------------------------------------------------------------
; set_response - store (body_ptr,len) and (ct_ptr,len) into resp_* globals
; Purpose:        Stores the four HTTP response component globals from register
;                 arguments. Leaf helper used by route_request and resp_set_error.
; Inputs:         RCX=body_ptr (u8*), RDX=body_len (usize),
;                 R8=ct_ptr (u8*), R9=ct_len (usize)
; Outputs:        RAX=unmodified; resp_body_ptr, resp_body_len, resp_ct_ptr,
;                 resp_ct_len globals set
; Errors:         none
; Clobbers:       none
; Preserves:      all (including RAX)
; Locals:         0 (leaf, no frame)
; Max read:       0   Max write: 0 (writes to resp_* globals, not via ptr args)
; ---------------------------------------------------------------------------
set_response:
    mov     [resp_body_ptr], rcx
    mov     [resp_body_len], rdx
    mov     [resp_ct_ptr], r8
    mov     [resp_ct_len], r9
    ret

; ---------------------------------------------------------------------------
; resp_set_error - set response globals to the generic error body
; Purpose:        Sets the four resp_* globals to the fixed "error" body with
;                 text/plain content-type. Preserves RAX so the caller can keep
;                 its HTTP status code.
; Inputs:         none (operates on globals; RAX preserved for caller)
; Outputs:        RAX=preserved; resp_body_ptr, resp_body_len, resp_ct_ptr,
;                 resp_ct_len globals set
; Errors:         none
; Clobbers:       R11
; Preserves:      RBX,RBP,RDI,RSI,R12-R15 (nonvolatile); also RAX
; Locals:         0 (leaf, no frame)
; Max read:       0   Max write: 0 (writes to resp_* globals, not via ptr args)
; ---------------------------------------------------------------------------
global resp_set_error
resp_set_error:
    lea     r11, [rel err_body]
    mov     [resp_body_ptr], r11
    mov     qword [resp_body_len], ERR_BODY_LEN
    lea     r11, [rel ct_text]
    mov     [resp_ct_ptr], r11
    mov     qword [resp_ct_len], CT_TEXT_LEN
    ret

; ---------------------------------------------------------------------------
; path_is_known - test if req_path matches one of the four known paths
; Purpose:        Returns 1 if req_path is "/", "/version", "/health", or
;                 "/chat"; 0 otherwise. Used by route_request to distinguish
;                 404 from 405 without duplicating path comparison logic.
; Inputs:         none (operates on req_path_ptr, req_path_len globals)
; Outputs:        RAX=1 if path is known, 0 if unknown
; Errors:         none
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11 (volatile; calls bytes_eq)
; Preserves:      RBX,RBP,RDI,RSI,R12-R15 (nonvolatile)
; Locals:         32 (shadow only; push rbp + sub 32)
; Max read:       req_path_len bytes from [req_path_ptr] (via bytes_eq)
; Max write:      0 (writes only to RAX return register)
; Precond:        req_path_ptr, req_path_len valid (set by http_parse)
; ---------------------------------------------------------------------------
path_is_known:
    push    rbp
    mov     rbp, rsp
    sub     rsp, 32
    mov     rcx, [rel req_path_ptr]
    mov     rdx, [rel req_path_len]
    lea     r8,  [rel s_root]
    mov     r9,  1
    call    bytes_eq
    test    eax, eax
    jnz     .yes
    mov     rcx, [rel req_path_ptr]
    mov     rdx, [rel req_path_len]
    lea     r8,  [rel s_version]
    mov     r9,  8
    call    bytes_eq
    test    eax, eax
    jnz     .yes
    mov     rcx, [rel req_path_ptr]
    mov     rdx, [rel req_path_len]
    lea     r8,  [rel s_health]
    mov     r9,  7
    call    bytes_eq
    test    eax, eax
    jnz     .yes
    mov     rcx, [rel req_path_ptr]
    mov     rdx, [rel req_path_len]
    lea     r8,  [rel s_chat]
    mov     r9,  5
    call    bytes_eq
    test    eax, eax
    jnz     .yes
    xor     eax, eax
    jmp     .pout
.yes:
    mov     eax, 1
.pout:
    add     rsp, 32
    pop     rbp
    ret

; ===========================================================================
; route_request - dispatch parsed request to matching response handler
; Purpose:        Reads req_* globals (populated by http_read/http_parse),
;                 matches method + path, sets resp_* globals, and either
;                 returns synchronously (static routes) or kicks the async
;                 gateway for POST /chat. Single entry point for all HTTP
;                 response selection.
; @param[in]      none (operates on req_* globals)
; @param[out]     RAX - u16 HTTP status (200, 404, 405, 411, 503)
; Inputs:         none; operates on req_method_ptr, req_method_len,
;                 req_path_ptr, req_path_len, req_has_cl,
;                 req_content_length, req_header_end, recv_buf globals
; Outputs:        RAX = HTTP status; resp_body_ptr/len, resp_ct_ptr/len set
; Errors:         RAX in {404,405,411,503}; err_body set via set_response
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11
; Preserves:      RBX,RBP,RDI,RSI,R12-R15
; Locals:         32 (shadow only; push rbp + sub 32)
; Max read:       req_path_len bytes from [req_path_ptr], req_method_len from
;                 [req_method_ptr] (via bytes_eq); CAP_CHAT_BODY from
;                 [recv_buf + req_header_end] (forwarded to gateway_start)
; Max write:      0 (writes go to resp_* globals, not caller buffers)
; Precond:        http_parse complete; req_path_len, req_method_len valid;
;                 req_header_end set; recv_buf populated
; Stack:          SHADOW 32, single CALL to gateway_start at aligned RSP
; Modified:       RAX,RCX,RDX,R8,R9,R10,R11
; Initial inputs to registers: none (operates on globals)
; Register assignments:
;   match_phase:  RCX=method_ptr, RDX=method_len, R8=candidate, R9=cand_len
;                 (for bytes_eq); then RCX=path_ptr, RDX=path_len
;   route_phase:  RCX=body_ptr, RDX=body_len, R8=ct_ptr, R9=ct_len
;                 (for set_response); or RCX=body_ptr, RDX=body_len
;                 for gateway_start; RAX=HTTP status for error paths
; ===========================================================================
global route_request
route_request:
    push    rbp
    mov     rbp, rsp
    sub     rsp, 32
    ; --- method GET? ---
    mov     rcx, [rel req_method_ptr]
    mov     rdx, [rel req_method_len]
    lea     r8,  [rel s_get]
    mov     r9,  3
    call    bytes_eq
    test    eax, eax
    jnz     .get
    ; --- method POST? ---
    mov     rcx, [rel req_method_ptr]
    mov     rdx, [rel req_method_len]
    lea     r8,  [rel s_post]
    mov     r9,  4
    call    bytes_eq
    test    eax, eax
    jnz     .post
    ; --- other method: 405 on known path, else 404 ---
    call    path_is_known
    test    eax, eax
    jnz     .meth_not_allowed
    jmp     .notfound
.get:
    mov     rcx, [rel req_path_ptr]
    mov     rdx, [rel req_path_len]
    lea     r8,  [rel s_root]
    mov     r9,  1
    call    bytes_eq
    test    eax, eax
    jnz     .serve_root
    mov     rcx, [rel req_path_ptr]
    mov     rdx, [rel req_path_len]
    lea     r8,  [rel s_version]
    mov     r9,  8
    call    bytes_eq
    test    eax, eax
    jnz     .serve_version
    ; GET /health -> JSON status
    mov     rcx, [rel req_path_ptr]
    mov     rdx, [rel req_path_len]
    lea     r8,  [rel s_health]
    mov     r9,  7
    call    bytes_eq
    test    eax, eax
    jnz     .serve_health
    ; GET /chat -> 405; GET /<unknown> -> 404
    mov     rcx, [rel req_path_ptr]
    mov     rdx, [rel req_path_len]
    lea     r8,  [rel s_chat]
    mov     r9,  5
    call    bytes_eq
    test    eax, eax
    jnz     .meth_not_allowed
    jmp     .notfound
.post:
    mov     rcx, [rel req_path_ptr]
    mov     rdx, [rel req_path_len]
    lea     r8,  [rel s_chat]
    mov     r9,  5
    call    bytes_eq
    test    eax, eax
    jnz     .do_chat
    ; POST on / or /version -> 405; else 404
    call    path_is_known
    test    eax, eax
    jnz     .meth_not_allowed
    jmp     .notfound
.do_chat:
    cmp     dword [rel req_has_cl], 0
    je      .length_required
    ; message body = recv_buf[header_end .. header_end+content_length)
    lea     rcx, [rel recv_buf]
    add     rcx, [rel req_header_end]        ; msg ptr (two-step: ASLR-safe)
    mov     rdx, [rel req_content_length]    ; msg len
    call    gateway_start                    ; async start; returns 0 if accepted
    test    eax, eax
    jnz     .server_busy                     ; gateway busy → 503
    ; Gateway started asynchronously. Response will come later when the
    ; WinHTTP call chain completes. Do NOT set resp_* here — start.asm
    ; detects gw_state != GW_IDLE and defers the HTTP response.
    mov     eax, HTTP_200                    ; placeholder; start.asm will NOT send this
    jmp     .rout
    ; ---- 200 responses ----
.serve_root:
%ifdef DEV_MODE
    call    load_index_html          ; RAX=ptr, RDX=len, CF=1=error
    jc      .serve_root_embedded
    mov     rcx, rax                 ; body ptr from load_index_html
    ; RDX already = bytes read (body len)
    lea     r8,  [rel ct_html]
    mov     r9,  CT_HTML_LEN
    call    set_response
    mov     eax, HTTP_200
    jmp     .rout
.serve_root_embedded:
%endif
    lea     rcx, [rel chat_html]
    mov     rdx, [rel chat_html_len]
    lea     r8,  [rel ct_html]
    mov     r9,  CT_HTML_LEN
    call    set_response
    mov     eax, HTTP_200
    jmp     .rout
.serve_version:
    lea     rcx, [rel build_id]
    mov     rdx, [rel build_id_len]
    lea     r8,  [rel ct_ascii]
    mov     r9,  CT_ASCII_LEN
    call    set_response
    mov     eax, HTTP_200
    jmp     .rout
.serve_health:
    lea     rcx, [rel health_json]
    mov     rdx, [rel health_json_len]
    lea     r8,  [rel ct_json]
    mov     r9,  CT_JSON_LEN
    call    set_response
    mov     eax, HTTP_200
    jmp     .rout
    ; ---- error responses (set err body; eax already set) ----
.notfound:
    mov     eax, HTTP_404
    jmp     .send_err
.meth_not_allowed:
    mov     eax, HTTP_405
    jmp     .send_err
.server_busy:
    mov     eax, HTTP_503
    jmp     .send_err
.length_required:
    mov     eax, HTTP_411
    jmp     .send_err
.send_err:
    lea     rcx, [rel err_body]
    mov     rdx, ERR_BODY_LEN
    lea     r8,  [rel ct_text]
    mov     r9,  CT_TEXT_LEN
    call    set_response
.rout:
    add     rsp, 32
    pop     rbp
    ret
