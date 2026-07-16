; ===========================================================================
; engine_gateway.asm - native OpenRouter HTTPS chat-completions client.
; TLS, authentication, JSON encoding/decoding, and response extraction all
; execute in this Assembly module through the documented Windows WinHTTP API.
; ===========================================================================
%include "win64.inc"
%include "winapi.inc"
%include "winhttp.inc"
%include "http.inc"
%include "config.inc"

extern mem_find
extern gateway_req
extern gateway_resp
extern gateway_draft
extern gateway_api_key
extern gateway_model
extern gateway_headers_w
extern gw_used
extern decode_target_ptr
extern decode_target_cap
extern resp_body_ptr
extern resp_body_len
extern resp_ct_ptr
extern resp_ct_len

default rel

section .data
env_api_key: db "OPENROUTER_API_KEY",0
env_model:   db "OPENROUTER_MODEL",0

; WinHTTP requires UTF-16 strings.
ua_w:   dw 'a','s','m','-','c','h','a','t','/','1','.','0',0
host_w: dw 'o','p','e','n','r','o','u','t','e','r','.','a','i',0
post_w: dw 'P','O','S','T',0
path_w: dw '/','a','p','i','/','v','1','/','c','h','a','t','/','c','o','m','p','l','e','t','i','o','n','s',0

auth_prefix: db "Authorization: Bearer "
AUTH_PREFIX_LEN equ $-auth_prefix
auth_suffix: db 13,10,"Content-Type: application/json",13,10
AUTH_SUFFIX_LEN equ $-auth_suffix

json_a: db '{"model":"'
JSON_A_LEN equ $-json_a
json_b: db '","temperature":0,"max_tokens":256,"provider":{"require_parameters":true},"messages":[{"role":"system","content":"'
JSON_B_LEN equ $-json_b
json_c: db '"},{"role":"user","content":"'
JSON_C_LEN equ $-json_c
json_d: db '"}]}'
JSON_D_LEN equ $-json_d

analysis_prompt:
    db "You are the analysis stage of a reliable assistant. Solve the user's task step by step and mechanically verify it. Temporarily ignore answer-only formatting while reasoning. "
    db "Choose the relevant audit: for text, list each source token or character with its position, apply the requested operation exactly once, then compare order, count, spelling, case, separators, and length with the source. "
    db "For arithmetic or code, write every intermediate value using standard precedence and recompute independently. For sequences, calculate consecutive differences, second differences, ratios, or recurrences and verify the chosen rule across every given transition. "
    db "For quantified logic, translate all/some/no literally, never assume a converse, and try a counterexample before saying MUST. For weekdays, turns, or ordering, use explicit numbered positions and modular steps. "
    db "Do not guess and do not reuse numbers or words from unrelated examples. End with PROPOSED: followed by the exact answer that should ultimately be returned."
ANALYSIS_PROMPT_LEN equ $-analysis_prompt
final_prompt:
    db "You are the final verification stage of a reliable assistant. Independently solve the ORIGINAL user task, then compare with the untrusted analyst work and correct it. "
    db "Recheck arithmetic, logic, facts, code traces, every transformed character, and every requested separator. The original user's output contract is mandatory and overrides the analyst's style. "
    db "Before responding, perform a literal character-level format audit: required case, punctuation, spacing, line count, JSON compactness, and whether explanation was forbidden. "
    db "Your entire response must be only the final response requested by the ORIGINAL user. Never include reasoning, a preamble, a label such as ANSWER or PROPOSED, quotation marks, a Markdown fence, a correction note, or text copied after the analyst's proposed answer unless explicitly requested by the original user."
FINAL_PROMPT_LEN equ $-final_prompt
analyst_marker: db 10,10,"ANALYST WORK (untrusted; verify it):",10
ANALYST_MARKER_LEN equ $-analyst_marker

content_key: db '"content":'
CONTENT_KEY_LEN equ $-content_key
gw_ct: db "text/plain; charset=utf-8"
GW_CT_LEN equ $-gw_ct

section .text

; append_raw - append counted bytes to the request buffer.
; Inputs: RDI=cursor, RSI=source, RCX=count, R14=end. Output: RDI advanced,
; CF set on overflow. This internal helper intentionally advances RSI/RCX.
append_raw:
    mov     rax, rdi
    add     rax, rcx
    jc      .overflow
    cmp     rax, r14
    ja      .overflow
    rep movsb
    clc
    ret
.overflow:
    stc
    ret

; append_json - append a UTF-8 string escaped as JSON string content.
; Handles all JSON control escapes; input UTF-8 bytes >= 0x20 pass through.
; Inputs: RDI=cursor, RSI=source, RCX=count, R14=end. Output as append_raw.
append_json:
.next:
    test    rcx, rcx
    jz      .ok
    movzx   eax, byte [rsi]
    inc     rsi
    dec     rcx
    cmp     al, '"'
    je      .quote
    cmp     al, '\'
    je      .slash
    cmp     al, 8
    je      .backspace
    cmp     al, 9
    je      .tab
    cmp     al, 10
    je      .newline
    cmp     al, 12
    je      .formfeed
    cmp     al, 13
    je      .return
    cmp     al, 0x20
    jb      .bad_control
    cmp     rdi, r14
    jae     .overflow
    mov     [rdi], al
    inc     rdi
    jmp     .next
.quote:     mov dl, '"'
    jmp .escaped
.slash:     mov dl, '\'
    jmp .escaped
.backspace: mov dl, 'b'
    jmp .escaped
.tab:       mov dl, 't'
    jmp .escaped
.newline:   mov dl, 'n'
    jmp .escaped
.formfeed:  mov dl, 'f'
    jmp .escaped
.return:    mov dl, 'r'
.escaped:
    mov     rax, rdi
    add     rax, 2
    cmp     rax, r14
    ja      .overflow
    mov     byte [rdi], '\'
    mov     [rdi+1], dl
    add     rdi, 2
    jmp     .next
.bad_control:
    ; Encode uncommon C0 controls as \u00XX.
    mov     rax, rdi
    add     rax, 6
    cmp     rax, r14
    ja      .overflow
    mov     byte [rdi], '\'
    mov     byte [rdi+1], 'u'
    mov     byte [rdi+2], '0'
    mov     byte [rdi+3], '0'
    mov     edx, eax
    and     edx, 15
    shr     eax, 4
    and     eax, 15
    cmp     al, 9
    jbe     .hi_digit
    add     al, 'a'-10
    jmp     .hi_store
.hi_digit:  add al, '0'
.hi_store:  mov [rdi+4], al
    mov     eax, edx
    cmp     al, 9
    jbe     .lo_digit
    add     al, 'a'-10
    jmp     .lo_store
.lo_digit:  add al, '0'
.lo_store:  mov [rdi+5], al
    add     rdi, 6
    jmp     .next
.ok:
    clc
    ret
.overflow:
    stc
    ret

; append_wide - widen counted ASCII bytes into a UTF-16 header buffer.
; Inputs: RDI=wide cursor, RSI=ASCII, RCX=count, R14=wide-buffer byte end.
append_wide:
.loop:
    test    rcx, rcx
    jz      .ok
    mov     rax, rdi
    add     rax, 2
    cmp     rax, r14
    ja      .overflow
    movzx   eax, byte [rsi]
    mov     [rdi], ax
    inc     rsi
    add     rdi, 2
    dec     rcx
    jmp     .loop
.ok: clc
    ret
.overflow: stc
    ret

; hex_nibble - convert one ASCII hex digit. CF set if invalid.
hex_nibble:
    cmp     al, '0'
    jb      .bad
    cmp     al, '9'
    jbe     .digit
    or      al, 0x20
    cmp     al, 'a'
    jb      .bad
    cmp     al, 'f'
    ja      .bad
    sub     al, 'a'-10
    clc
    ret
.digit:
    sub     al, '0'
    clc
    ret
.bad:
    stc
    ret

; decode_content - extract and JSON-decode choices[0].message.content.
; Input: gateway_resp/gw_used. Output: RAX=0 and resp globals, else 1.
decode_content:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    rsi
    push    rdi
    push    r12
    push    r13
    push    r14
    sub     rsp, 32                 ; six saved regs leave RSP call-aligned
    lea     rcx, [rel gateway_resp]
    mov     rdx, [rel gw_used]
    lea     r8,  [rel content_key]
    mov     r9,  CONTENT_KEY_LEN
    call    mem_find
    test    rax, rax
    js      .fail
    lea     rsi, [rel gateway_resp]
    add     rsi, rax
    add     rsi, CONTENT_KEY_LEN
    lea     r14, [rel gateway_resp]
    add     r14, [rel gw_used]
.skip_ws:
    cmp     rsi, r14
    jae     .fail
    mov     al, [rsi]
    cmp     al, ' '
    je      .ws
    cmp     al, 9
    je      .ws
    cmp     al, 10
    je      .ws
    cmp     al, 13
    jne     .expect_quote
.ws: inc rsi
    jmp .skip_ws
.expect_quote:
    cmp     byte [rsi], '"'
    jne     .fail
    inc     rsi
    mov     rdi, [rel decode_target_ptr]
    mov     r12, rdi                 ; decoded start
    mov     r13, rdi
    add     r13, [rel decode_target_cap]
.decode:
    cmp     rsi, r14
    jae     .fail
    movzx   eax, byte [rsi]
    inc     rsi
    cmp     al, '"'
    je      .done
    cmp     al, '\'
    jne     .emit_byte
    cmp     rsi, r14
    jae     .fail
    movzx   eax, byte [rsi]
    inc     rsi
    cmp     al, '"'
    je      .emit_byte
    cmp     al, '\'
    je      .emit_byte
    cmp     al, '/'
    je      .emit_byte
    cmp     al, 'b'
    je      .esc_b
    cmp     al, 'f'
    je      .esc_f
    cmp     al, 'n'
    je      .esc_n
    cmp     al, 'r'
    je      .esc_r
    cmp     al, 't'
    je      .esc_t
    cmp     al, 'u'
    jne     .fail
    ; Decode one BMP \uXXXX value to UTF-8 (surrogates become '?').
    mov     ecx, 4
    xor     ebx, ebx
.hex_loop:
    cmp     rsi, r14
    jae     .fail
    mov     al, [rsi]
    inc     rsi
    call    hex_nibble
    jc      .fail
    shl     ebx, 4
    movzx   eax, al
    or      ebx, eax
    dec     ecx
    jnz     .hex_loop
    cmp     ebx, 0xD800
    jb      .utf8
    cmp     ebx, 0xDFFF
    ja      .utf8
    mov     al, '?'
    jmp     .emit_byte
.utf8:
    cmp     ebx, 0x7F
    ja      .utf8_two
    mov     eax, ebx
    jmp     .emit_byte
.utf8_two:
    cmp     ebx, 0x7FF
    ja      .utf8_three
    mov     rax, rdi
    add     rax, 2
    cmp     rax, r13
    ja      .fail
    mov     eax, ebx
    shr     eax, 6
    or      al, 0xC0
    mov     [rdi], al
    mov     eax, ebx
    and     al, 0x3F
    or      al, 0x80
    mov     [rdi+1], al
    add     rdi, 2
    jmp     .decode
.utf8_three:
    mov     rax, rdi
    add     rax, 3
    cmp     rax, r13
    ja      .fail
    mov     eax, ebx
    shr     eax, 12
    or      al, 0xE0
    mov     [rdi], al
    mov     eax, ebx
    shr     eax, 6
    and     al, 0x3F
    or      al, 0x80
    mov     [rdi+1], al
    mov     eax, ebx
    and     al, 0x3F
    or      al, 0x80
    mov     [rdi+2], al
    add     rdi, 3
    jmp     .decode
.esc_b: mov al, 8
    jmp .emit_byte
.esc_f: mov al, 12
    jmp .emit_byte
.esc_n: mov al, 10
    jmp .emit_byte
.esc_r: mov al, 13
    jmp .emit_byte
.esc_t: mov al, 9
.emit_byte:
    cmp     rdi, r13
    jae     .fail
    mov     [rdi], al
    inc     rdi
    jmp     .decode
.done:
    ; Match the former relay's strip(): trim outer ASCII whitespace only.
.trim_left:
    cmp     r12, rdi
    jae     .fail
    mov     al, [r12]
    cmp     al, ' '
    je      .left_one
    cmp     al, 9
    je      .left_one
    cmp     al, 10
    je      .left_one
    cmp     al, 13
    jne     .trim_right
.left_one: inc r12
    jmp .trim_left
.trim_right:
    cmp     rdi, r12
    jbe     .fail
    mov     al, [rdi-1]
    cmp     al, ' '
    je      .right_one
    cmp     al, 9
    je      .right_one
    cmp     al, 10
    je      .right_one
    cmp     al, 13
    jne     .success
.right_one: dec rdi
    jmp .trim_right
.success:
    mov     [rel resp_body_ptr], r12
    mov     rax, rdi
    sub     rax, r12
    mov     [rel resp_body_len], rax
    lea     rax, [rel gw_ct]
    mov     [rel resp_ct_ptr], rax
    mov     qword [rel resp_ct_len], GW_CT_LEN
    xor     eax, eax
    jmp     .out
.fail:
    mov     eax, 1
.out:
    add     rsp, 32
    pop     r14
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbx
    pop     rbp
    ret

; gateway_generate - call OpenRouter and expose the decoded model answer.
; Inputs: RCX=user UTF-8 pointer, RDX=length.
; Output: RAX=0 success, HTTP_502 failure. No non-LLM answer path exists.
global gateway_generate
gateway_generate:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    rsi
    push    rdi
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 136                ; seven saved regs require +8 alignment
    mov     [rsp+96], rcx           ; user pointer
    mov     [rsp+104], rdx          ; user length
    mov     qword [rsp+64], 0       ; stage: 0=analysis, 1=final
    xor     ebx, ebx                ; session handle
    xor     r12d, r12d              ; connection handle
    xor     r15d, r15d              ; request handle

    ; Configuration is inherited from run.ps1, which reads the shared .env.
    lea     rcx, [rel env_api_key]
    lea     rdx, [rel gateway_api_key]
    mov     r8d, CAP_API_KEY
    call    GetEnvironmentVariableA
    test    eax, eax
    jz      .fail
    cmp     eax, CAP_API_KEY
    jae     .fail
    mov     [rsp+112], rax          ; key length
    lea     rcx, [rel env_model]
    lea     rdx, [rel gateway_model]
    mov     r8d, CAP_MODEL
    call    GetEnvironmentVariableA
    test    eax, eax
    jz      .fail
    cmp     eax, CAP_MODEL
    jae     .fail
    mov     [rsp+120], rax          ; model length

    ; Construct the OpenRouter request body with strict JSON escaping.
    lea     rdi, [rel gateway_req]
    lea     r14, [rel gateway_req]
    add     r14, CAP_GATEWAY_REQ
    lea     rsi, [rel json_a]
    mov     ecx, JSON_A_LEN
    call    append_raw
    jc      .fail
    lea     rsi, [rel gateway_model]
    mov     rcx, [rsp+120]
    call    append_json
    jc      .fail
    lea     rsi, [rel json_b]
    mov     ecx, JSON_B_LEN
    call    append_raw
    jc      .fail
    lea     rsi, [rel analysis_prompt]
    mov     ecx, ANALYSIS_PROMPT_LEN
    call    append_json
    jc      .fail
    lea     rsi, [rel json_c]
    mov     ecx, JSON_C_LEN
    call    append_raw
    jc      .fail
    mov     rsi, [rsp+96]
    mov     rcx, [rsp+104]
    call    append_json
    jc      .fail
    lea     rsi, [rel json_d]
    mov     ecx, JSON_D_LEN
    call    append_raw
    jc      .fail
    lea     rax, [rel gateway_req]
    sub     rdi, rax
    mov     r13, rdi                ; request body byte length
    lea     rax, [rel gateway_draft]
    mov     [rel decode_target_ptr], rax
    mov     qword [rel decode_target_cap], CAP_GATEWAY_DRAFT

    ; Build UTF-16 Authorization and Content-Type headers.
    lea     rdi, [rel gateway_headers_w]
    lea     r14, [rel gateway_headers_w]
    add     r14, CAP_AUTH_WCHARS*2
    lea     rsi, [rel auth_prefix]
    mov     ecx, AUTH_PREFIX_LEN
    call    append_wide
    jc      .fail
    lea     rsi, [rel gateway_api_key]
    mov     rcx, [rsp+112]
    call    append_wide
    jc      .fail
    lea     rsi, [rel auth_suffix]
    mov     ecx, AUTH_SUFFIX_LEN
    call    append_wide
    jc      .fail
    lea     rax, [rel gateway_headers_w]
    sub     rdi, rax
    shr     rdi, 1
    mov     [rsp+88], rdi           ; header character count

.start_http:
    lea     rcx, [rel ua_w]
    mov     edx, WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY
    xor     r8d, r8d
    xor     r9d, r9d
    mov     qword [rsp+32], 0
    call    WinHttpOpen
    test    rax, rax
    jz      .fail
    mov     rbx, rax
    mov     rcx, rbx
    mov     edx, 10000
    mov     r8d, 10000
    mov     r9d, 60000
    mov     qword [rsp+32], 60000
    call    WinHttpSetTimeouts
    test    eax, eax
    jz      .fail
    mov     rcx, rbx
    lea     rdx, [rel host_w]
    mov     r8d, 443
    xor     r9d, r9d
    call    WinHttpConnect
    test    rax, rax
    jz      .fail
    mov     r12, rax
    mov     rcx, r12
    lea     rdx, [rel post_w]
    lea     r8,  [rel path_w]
    xor     r9d, r9d
    mov     qword [rsp+32], 0
    mov     qword [rsp+40], 0
    mov     qword [rsp+48], WINHTTP_FLAG_SECURE
    call    WinHttpOpenRequest
    test    rax, rax
    jz      .fail
    mov     r15, rax
    mov     rcx, r15
    lea     rdx, [rel gateway_headers_w]
    mov     r8d, [rsp+88]
    lea     r9,  [rel gateway_req]
    mov     [rsp+32], r13
    mov     [rsp+40], r13
    mov     qword [rsp+48], 0
    call    WinHttpSendRequest
    test    eax, eax
    jz      .fail
    mov     rcx, r15
    xor     edx, edx
    call    WinHttpReceiveResponse
    test    eax, eax
    jz      .fail

    ; Require a numeric HTTP 200 before accepting response content.
    mov     dword [rsp+80], 4        ; status buffer size
    mov     dword [rsp+84], 0        ; status value
    mov     rcx, r15
    mov     edx, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER
    xor     r8d, r8d
    lea     r9, [rsp+84]
    lea     rax, [rsp+80]
    mov     [rsp+32], rax
    mov     qword [rsp+40], 0
    call    WinHttpQueryHeaders
    test    eax, eax
    jz      .fail
    cmp     dword [rsp+84], 200
    jne     .fail

    mov     qword [rel gw_used], 0
.read:
    mov     rax, CAP_GATEWAY_RESP
    sub     rax, [rel gw_used]
    jz      .fail
    mov     rcx, r15
    lea     rdx, [rel gateway_resp]
    add     rdx, [rel gw_used]
    mov     r8d, eax
    lea     r9, [rsp+76]             ; DWORD bytes read
    mov     dword [rsp+76], 0
    call    WinHttpReadData
    test    eax, eax
    jz      .fail
    mov     eax, [rsp+76]
    test    eax, eax
    jz      .read_done
    add     [rel gw_used], rax
    jmp     .read
.read_done:
    call    decode_content
    test    eax, eax
    jnz     .fail
    cmp     qword [rsp+64], 0
    jne     .final_complete

    ; Preserve the first model's analysis, close its WinHTTP handles, then
    ; construct a second real LLM request for independent final verification.
    mov     rax, [rel resp_body_len]
    mov     [rsp+72], rax           ; draft length
    mov     rcx, r15
    call    WinHttpCloseHandle
    xor     r15d, r15d
    mov     rcx, r12
    call    WinHttpCloseHandle
    xor     r12d, r12d
    mov     rcx, rbx
    call    WinHttpCloseHandle
    xor     ebx, ebx
    mov     qword [rsp+64], 1

    lea     rdi, [rel gateway_req]
    lea     r14, [rel gateway_req]
    add     r14, CAP_GATEWAY_REQ
    lea     rsi, [rel json_a]
    mov     ecx, JSON_A_LEN
    call    append_raw
    jc      .fail
    lea     rsi, [rel gateway_model]
    mov     rcx, [rsp+120]
    call    append_json
    jc      .fail
    lea     rsi, [rel json_b]
    mov     ecx, JSON_B_LEN
    call    append_raw
    jc      .fail
    lea     rsi, [rel final_prompt]
    mov     ecx, FINAL_PROMPT_LEN
    call    append_json
    jc      .fail
    lea     rsi, [rel json_c]
    mov     ecx, JSON_C_LEN
    call    append_raw
    jc      .fail
    mov     rsi, [rsp+96]
    mov     rcx, [rsp+104]
    call    append_json
    jc      .fail
    lea     rsi, [rel analyst_marker]
    mov     ecx, ANALYST_MARKER_LEN
    call    append_json
    jc      .fail
    lea     rsi, [rel gateway_draft]
    mov     rcx, [rsp+72]
    call    append_json
    jc      .fail
    lea     rsi, [rel json_d]
    mov     ecx, JSON_D_LEN
    call    append_raw
    jc      .fail
    lea     rax, [rel gateway_req]
    sub     rdi, rax
    mov     r13, rdi
    lea     rax, [rel gateway_req]
    mov     [rel decode_target_ptr], rax
    mov     qword [rel decode_target_cap], CAP_GATEWAY_REQ
    jmp     .start_http

.final_complete:
    xor     r13d, r13d               ; final status success
    jmp     .cleanup
.fail:
    mov     r13d, HTTP_502
.cleanup:
    test    r15, r15
    jz      .close_connect
    mov     rcx, r15
    call    WinHttpCloseHandle
.close_connect:
    test    r12, r12
    jz      .close_session
    mov     rcx, r12
    call    WinHttpCloseHandle
.close_session:
    test    rbx, rbx
    jz      .out
    mov     rcx, rbx
    call    WinHttpCloseHandle
.out:
    mov     eax, r13d
    add     rsp, 136
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbx
    pop     rbp
    ret
