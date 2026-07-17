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
; set_response - fill the four response globals from register args. Leaf.
; Inputs:  RCX=body_ptr, RDX=body_len, R8=ct_ptr, R9=ct_len. Preserves RAX.
; ---------------------------------------------------------------------------
set_response:
    mov     [resp_body_ptr], rcx
    mov     [resp_body_len], rdx
    mov     [resp_ct_ptr], r8
    mov     [resp_ct_len], r9
    ret

; ---------------------------------------------------------------------------
; resp_set_error - set response globals to the generic error body. Preserves
; EAX (so the caller can keep its status code).
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
; path_is_known - 1 if path is /, /version, or /chat, else 0. Returns RAX.
; Alignment: push rbp + sub 32 -> aligned.
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

; ---------------------------------------------------------------------------
; route_request - dispatch a parsed request to the right response.
; Inputs:  none (operates on req_* globals, set by http_read/parse).
; Outputs: RAX = HTTP status code; resp_* globals set for http_respond.
; Clobbers: volatile. Alignment: push rbp + sub 32 -> aligned.
; ---------------------------------------------------------------------------
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
