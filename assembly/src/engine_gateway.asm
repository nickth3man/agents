; ===========================================================================
; src/engine_gateway.asm - local LLM relay client (PLAN §2.10, Milestone 8)
; ===========================================================================
%include "win64.inc"
%include "winsock.inc"
%include "http.inc"
%include "config.inc"

extern copy_bytes
extern u32_to_dec
extern mem_find
extern bytes_eq
extern send_all
extern gateway_req
extern gateway_resp
extern gw_sock
extern gw_used
extern gw_header_end
extern resp_body_ptr
extern resp_body_len
extern resp_ct_ptr
extern resp_ct_len

default rel

section .data
gw_req_pfx:
    db "POST /generate HTTP/1.1",13,10
    db "Host: 127.0.0.1",13,10
    db "Content-Type: text/plain",13,10
    db "Content-Length: "
GW_REQ_PFX_LEN equ $-gw_req_pfx
gw_req_sfx: db 13,10,"Connection: close",13,10,13,10
GW_REQ_SFX_LEN equ $-gw_req_sfx
gw_crlf:    db 13,10,13,10
gw_ok_status: db "HTTP/1.1 200"
GW_OK_STATUS_LEN equ $-gw_ok_status
gw_ct:      db "text/plain; charset=utf-8"
GW_CT_LEN   equ $-gw_ct
gw_timeout_ms: dd 60000              ; long recv timeout for model latency
relay_addr:
    dw  AF_INET
    dw  PORT_8081_NET                ; htons(8081)
    dd  IP_127_0_0_1_NET            ; 127.0.0.1
    dq  0

section .text

; ---------------------------------------------------------------------------
; gateway_generate - send the user message to the relay, read back the reply.
; ---------------------------------------------------------------------------
; Inputs:  RCX = message ptr (in recv_buf), RDX = message length.
; Outputs: RAX = 0 on success (resp_* globals set to the model reply),
;          or HTTP_502 on any failure (relay down / timeout / malformed).
; Clobbers: volatile + saved rbx,r12,r13.
; Alignment: 4 pushes (rbp,rbx,r12,r13) entry≡8 -> ≡8; sub 40 -> ≡0. OK.
; Max read: CAP_GATEWAY_RESP (16 KiB). Max write: bounded relay reply.
; ---------------------------------------------------------------------------
global gateway_generate
gateway_generate:
    push    rbp
    mov     rbp, rsp
    push    rbx                     ; message ptr
    push    r12                     ; message len
    push    r13                     ; request-header cursor
    sub     rsp, 40                 ; shadow(32)+8; aligned
    mov     rbx, rcx
    mov     r12, rdx
    mov     qword [gw_used], 0
    mov     qword [gw_header_end], 0
    ; --- outbound socket ---
    mov     rcx, AF_INET
    mov     rdx, SOCK_STREAM
    mov     r8,  IPPROTO_TCP
    call    socket
    cmp     rax, INVALID_SOCKET
    je      .fail502
    mov     [gw_sock], rax
    ; --- long recv timeout (model latency) ---
    mov     rcx, rax
    mov     edx, SOL_SOCKET
    mov     r8d, SO_RCVTIMEO
    lea     r9,  [gw_timeout_ms]
    mov     dword [rsp+0x20], 4
    call    setsockopt
    ; --- connect to relay 127.0.0.1:8081 ---
    mov     rcx, [gw_sock]
    lea     rdx, [relay_addr]
    mov     r8,  SOCKADDR_IN_SIZE
    call    connect
    test    eax, eax
    jnz     .fail502_close
    ; --- build request headers into gateway_req ---
    lea     r13, [rel gateway_req]
    mov     rcx, r13
    lea     r8,  [gw_req_pfx]
    mov     rdx, GW_REQ_PFX_LEN
    call    copy_bytes
    add     r13, GW_REQ_PFX_LEN
    ; Content-Length decimal (message length)
    mov     ecx, r12d
    mov     rdx, r13
    mov     r8,  8
    call    u32_to_dec
    add     r13, rax
    ; suffix (Connection: close + blank line)
    mov     rcx, r13
    lea     r8,  [gw_req_sfx]
    mov     rdx, GW_REQ_SFX_LEN
    call    copy_bytes
    add     r13, GW_REQ_SFX_LEN
    ; --- send headers ---
    lea     r11, [rel gateway_req]
    mov     r8,  r13
    sub     r8,  r11                ; header length
    mov     rcx, [gw_sock]
    mov     rdx, r11
    call    send_all
    test    eax, eax
    jnz     .fail502_close
    ; --- send message body ---
    mov     rcx, [gw_sock]
    mov     rdx, rbx
    mov     r8,  r12
    call    send_all
    test    eax, eax
    jnz     .fail502_close
    ; --- read relay reply until it closes (bounded) ---
.gw_read:
    mov     rax, CAP_GATEWAY_RESP
    sub     rax, [gw_used]
    jle     .gw_read_done           ; buffer full
    mov     rcx, [gw_sock]
    lea     rdx, [gateway_resp]
    add     rdx, [gw_used]
    mov     r8,  rax
    xor     r9d, r9d
    call    recv
    mov     r11d, eax
    test    r11d, r11d
    js      .fail502_close          ; recv error (timeout / reset)
    je      .gw_read_done           ; relay closed -> response complete
    add     [gw_used], r11
    jmp     .gw_read
.gw_read_done:
    ; Reject relay HTTP errors instead of forwarding their text as a 200 reply.
    cmp     qword [gw_used], GW_OK_STATUS_LEN
    jb      .fail502_close
    lea     rcx, [rel gateway_resp]
    mov     rdx, GW_OK_STATUS_LEN
    lea     r8,  [gw_ok_status]
    mov     r9,  GW_OK_STATUS_LEN
    call    bytes_eq
    test    eax, eax
    jz      .fail502_close
    ; locate header/body split
    lea     rcx, [rel gateway_resp]
    mov     rdx, [gw_used]
    lea     r8,  [gw_crlf]
    mov     r9,  4
    call    mem_find
    test    rax, rax
    js      .fail502_close          ; no CRLFCRLF -> malformed
    add     rax, 4
    mov     [gw_header_end], rax
    ; resp_body = gateway_resp + header_end ; len = gw_used - header_end
    lea     r11, [rel gateway_resp]
    add     r11, [gw_header_end]
    mov     [resp_body_ptr], r11
    mov     rax, [gw_used]
    sub     rax, [gw_header_end]
    mov     [resp_body_len], rax
    lea     r11, [gw_ct]
    mov     [resp_ct_ptr], r11
    mov     qword [resp_ct_len], GW_CT_LEN
    mov     rcx, [gw_sock]
    call    closesocket
    xor     eax, eax                ; success
    jmp     .out
.fail502_close:
    mov     rcx, [gw_sock]
    call    closesocket
.fail502:
    mov     eax, HTTP_502
.out:
    add     rsp, 40
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret
