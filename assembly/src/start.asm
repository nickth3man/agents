; ===========================================================================
; src/start.asm - program entry + polling event loop (PLAN §2.3, concurrent)
; ===========================================================================
; 32-slot polling model: WFMO on {listen_event, gw_event} only.  No per-slot
; WSA events.  READING slots are polled every 100ms via non-blocking recv.
; Gateway completion is signaled via gw_event (auto-reset).  Timeouts use
; GetTickCount64 deadlines stored in each slot.
; ---------------------------------------------------------------------------
%include "win64.inc"
%include "winapi.inc"
%include "winsock.inc"
%include "http.inc"
%include "config.inc"
%include "generated/version.inc"

default rel

extern net_init
extern net_shutdown
extern apply_timeouts
extern read_more_request
extern route_request
extern resp_set_error
extern http_respond
extern gateway_start
extern gateway_advance
extern log_err
extern log_str
extern log_request
extern listen_sock
extern client_slots
extern req_id
extern req_used
extern req_header_end
extern req_has_cl
extern req_has_te
extern req_content_length
extern req_method_ptr
extern req_method_len
extern req_path_ptr
extern req_path_len
extern resp_body_len
extern req_cl_count
extern debug_canaries_init
extern debug_canaries_check
extern gw_state
extern gw_event
extern listen_event
extern wsaevents
extern gw_pending_slot
extern WSAGetLastError
extern GetTickCount64
extern recv_buf

section .data
banner:
    db  "[startup] build=", BUILD_ID, " listen=127.0.0.1:8080", 10
BANNER_LEN  equ $-banner
s_accept:   db "accept"
S_ACCEPT_LEN equ $-s_accept
s_event:    db "WSAEnumNetworkEvents"
S_EVENT_LEN equ $-s_event
s_client_timeout: db "client timeout"
S_CLIENT_TIMEOUT_LEN equ $-s_client_timeout
s_pool_full: db "pool full"
S_POOL_FULL_LEN equ $-s_pool_full
req_post:   db "POST"
REQ_POST_LEN equ $-req_post
req_path_chat: db "/chat"
REQ_PATH_CHAT_LEN equ $-req_path_chat

section .bss
fionbio_arg:   resd 1                    ; ioctlsocket(FIONBIO) argument
poll_slot_idx: resd 1                    ; current slot index during poll round
gw_status:     resd 1                    ; gateway completion status code

section .text

; ===========================================================================
; Slot address helper — r10 = client_slots + ecx * CLIENT_SLOT_SIZE
; Clobbers: RAX (offset).
; ===========================================================================
%macro slot_addr 0
    mov     eax, CLIENT_SLOT_SIZE
    mul     ecx
    lea     r10, [client_slots]
    add     r10, rax
%endmacro

; Reset the parser state owned by a client slot. The socket, lifecycle state,
; request ID, and deadline are managed separately by the caller.
%macro clear_slot_request 1
    mov     qword [%1 + ClientSlot.req_used], 0
    mov     qword [%1 + ClientSlot.req_header_end], 0
    mov     dword [%1 + ClientSlot.req_has_cl], 0
    mov     dword [%1 + ClientSlot.req_has_te], 0
    mov     qword [%1 + ClientSlot.req_content_length], 0
    mov     dword [%1 + ClientSlot.req_cl_count], 0
    mov     qword [%1 + ClientSlot.req_method_ptr], 0
    mov     qword [%1 + ClientSlot.req_method_len], 0
    mov     qword [%1 + ClientSlot.req_path_ptr], 0
    mov     qword [%1 + ClientSlot.req_path_len], 0
%endmacro

; ---------------------------------------------------------------------------
; start - true OS entry point (never returns).
; ---------------------------------------------------------------------------
global start
start:
    and     rsp, -16
    sub     rsp, 128                     ; scratch workspace
    lea     rcx, [banner]
    mov     rdx, BANNER_LEN
    call    log_str
    call    debug_canaries_init
    call    net_init

    ; --- Create Winsock event for listen socket (FD_ACCEPT) ---
    call    WSACreateEvent
    mov     [listen_event], rax
    mov     rcx, [listen_sock]
    mov     rdx, rax
    mov     r8d, FD_ACCEPT
    call    WSAEventSelect

    ; --- Create auto-reset event for gateway completion ---
    xor     ecx, ecx                     ; lpEventAttributes = NULL
    xor     edx, edx                     ; bManualReset = FALSE
    xor     r8d, r8d                     ; bInitialState = FALSE
    xor     r9d, r9d                     ; lpName = NULL
    call    CreateEventW
    mov     [gw_event], rax

    ; --- Initial state ---
    mov     dword [gw_state], GW_IDLE
    mov     dword [gw_pending_slot], -1

    ; --- Initialize all client slots to FREE ---
    xor     ecx, ecx
.init_slots:
    slot_addr
    mov     qword [r10 + ClientSlot.sock], 0
    mov     dword [r10 + ClientSlot.state], CS_FREE
    inc     ecx
    cmp     ecx, MAX_CLIENTS
    jb      .init_slots

; ========================== EVENT LOOP ===================================
.event_loop:
    ; Build handle array on stack: [rsp+32]=listen, [rsp+40]=gw
    mov     rax, [listen_event]
    mov     [rsp+32], rax
    mov     rax, [gw_event]
    mov     [rsp+40], rax

    ; Compute timeout: 100ms if any READING slot, else INFINITE
    mov     dword [rsp+48], INFINITE     ; default
    xor     ecx, ecx
.tmo_scan:
    slot_addr
    cmp     dword [r10 + ClientSlot.state], CS_READING
    jne     .tmo_next
    mov     dword [rsp+48], 100          ; poll interval
    jmp     .tmo_done
.tmo_next:
    inc     ecx
    cmp     ecx, MAX_CLIENTS
    jb      .tmo_scan
.tmo_done:

    ; WaitForMultipleObjects(2, handles, FALSE, timeout)
    mov     ecx, 2
    lea     rdx, [rsp+32]
    xor     r8d, r8d
    mov     r9d, [rsp+48]
    call    WaitForMultipleObjects
    cmp     eax, WAIT_OBJECT_0
    je      .handle_accept
    cmp     eax, WAIT_OBJECT_0 + 1
    je      .handle_gateway
    cmp     eax, WAIT_TIMEOUT
    je      .poll_clients
    jmp     .event_loop                  ; WAIT_FAILED — retry

; ====================== ACCEPT ==========================================
.handle_accept:
    mov     rcx, [listen_sock]
    mov     rdx, [listen_event]
    lea     r8,  [wsaevents]
    call    WSAEnumNetworkEvents
    test    eax, eax
    jnz     .event_fail

.accept_loop:
    xor     edx, edx
    xor     r8d, r8d
    mov     rcx, [listen_sock]
    call    accept
    cmp     rax, INVALID_SOCKET
    jne     .have_client
    call    WSAGetLastError
    cmp     eax, WSAEWOULDBLOCK
    je      .poll_clients                ; no more pending accepts
    lea     rcx, [s_accept]
    mov     rdx, S_ACCEPT_LEN
    call    log_err
    jmp     .poll_clients

.have_client:
    mov     rbx, rax                     ; preserve new socket

    ; ----- find first FREE slot -----
    xor     ecx, ecx
.find_slot:
    slot_addr
    cmp     dword [r10 + ClientSlot.state], CS_FREE
    je      .slot_found
    inc     ecx
    cmp     ecx, MAX_CLIENTS
    jb      .find_slot

    ; ----- pool full -----
    mov     rcx, rbx
    call    closesocket
    lea     rcx, [s_pool_full]
    mov     rdx, S_POOL_FULL_LEN
    call    log_err
    jmp     .accept_loop

.slot_found:
    ; ECX = slot index, r10 = slot address
    mov     [r10 + ClientSlot.sock], rbx
    mov     dword [r10 + ClientSlot.state], CS_READING

    inc     qword [req_id]
    mov     rax, [req_id]
    mov     [r10 + ClientSlot.req_id], rax

    clear_slot_request r10

    ; deadline
    mov     r12, r10                         ; WinAPI calls may clobber r10
    call    GetTickCount64
    add     rax, CLIENT_TIMEOUT_MS
    mov     [r12 + ClientSlot.deadline_tick], rax

    ; clear inherited WSA events, set non-blocking
    mov     rcx, rbx
    xor     edx, edx
    xor     r8d, r8d
    call    WSAEventSelect
    mov     rcx, rbx
    call    apply_timeouts
    mov     rcx, rbx
    mov     edx, FIONBIO
    lea     r8,  [fionbio_arg]
    mov     dword [r8], 1
    call    ioctlsocket

    jmp     .accept_loop

; ====================== POLL READING SLOTS ==============================
.poll_clients:
    xor     ecx, ecx
.poll_loop:
    mov     [poll_slot_idx], ecx
    slot_addr
    cmp     dword [r10 + ClientSlot.state], CS_READING
    jne     .poll_next

    ; ---- slot_to_globals: copy slot recv_buf / req_* into globals ----
    mov     ecx, [poll_slot_idx]
    call    slot_to_globals

    ; ---- read_more_request(slot.sock) ----
    ; (operates on global recv_buf/req_*)
    mov     ecx, [poll_slot_idx]
    slot_addr
    mov     rcx, [r10 + ClientSlot.sock]
    call    read_more_request
    mov     [rsp+0x50], eax              ; save return code

    ; ---- globals_to_slot: copy back ----
    mov     ecx, [poll_slot_idx]
    call    globals_to_slot

    ; ---- dispatch result ----
    mov     eax, [rsp+0x50]
    cmp     eax, 0
    je      .poll_complete
    cmp     eax, 1
    je      .poll_next                   ; more data needed
    cmp     eax, 2
    je      .poll_next                   ; WSAEWOULDBLOCK
    ; >= 400 error
    mov     ecx, [poll_slot_idx]
    call    error_and_free_slot
    jmp     .poll_next

.poll_complete:
    ; request fully read.  Disassociate WSA events, restore blocking.
    mov     ecx, [poll_slot_idx]
    call    disassociate_slot

    ; detect if /chat starts an async gateway round
    mov     eax, [gw_state]
    mov     [rsp+0x58], eax              ; pre-route gw_state

    call    debug_canaries_check
    test    eax, eax
    jnz     .poll_canary

    call    route_request                ; returns status in eax
    mov     [rsp+0x50], eax              ; status code

    call    debug_canaries_check
    test    eax, eax
    jnz     .poll_canary

    ; Did route_request start an async gateway?
    mov     edx, [gw_state]
    cmp     edx, [rsp+0x58]
    je      .poll_sync_respond

    ; ---- async /chat deferral ----
    mov     eax, [poll_slot_idx]
    mov     [gw_pending_slot], eax
    mov     ecx, [poll_slot_idx]
    slot_addr
    mov     dword [r10 + ClientSlot.state], CS_CHAT_PENDING

    call    gateway_advance
    test    eax, eax
    jnz     .gateway_respond
    jmp     .poll_next

.poll_sync_respond:
    mov     ecx, [poll_slot_idx]
    mov     edx, [rsp+0x50]              ; status code
    call    respond_and_free_slot
    jmp     .poll_next

.poll_canary:
    mov     eax, HTTP_500
    mov     ecx, [poll_slot_idx]
    call    error_and_free_slot

.poll_next:
    mov     ecx, [poll_slot_idx]
    inc     ecx
    cmp     ecx, MAX_CLIENTS
    jb      .poll_loop

    ; after all slots polled, check timeouts
    call    scan_timeouts
    jmp     .event_loop

; ====================== FATAL ==========================================
.event_fail:
    lea     rcx, [s_event]
    mov     rdx, S_EVENT_LEN
    call    log_err
    mov     rcx, 1
    call    net_shutdown
    ud2

; ====================== GATEWAY EVENT ====================================
.handle_gateway:
    call    gateway_advance
    test    eax, eax
    jz      .event_loop                   ; still in progress

.gateway_respond:
    mov     [gw_status], eax              ; persist

    ; locate pending slot
    mov     ecx, [gw_pending_slot]
    cmp     ecx, -1
    je      .event_loop
    slot_addr

    ; restore req_* from slot for logging
    mov     rax, [r10 + ClientSlot.req_id]
    mov     [req_id], rax
    mov     rax, [r10 + ClientSlot.req_content_length]
    mov     [req_content_length], rax
    lea     rax, [req_post]
    mov     [req_method_ptr], rax
    mov     qword [req_method_len], REQ_POST_LEN
    lea     rax, [req_path_chat]
    mov     [req_path_ptr], rax
    mov     qword [req_path_len], REQ_PATH_CHAT_LEN

    ; http_respond(slot.sock, gw_status)
    mov     rcx, [r10 + ClientSlot.sock]
    mov     edx, [gw_status]
    call    http_respond

    ; log_request(status, resp_body_len)
    mov     ecx, [gw_status]
    mov     rdx, [resp_body_len]
    call    log_request

    ; free slot
    mov     ecx, [gw_pending_slot]
    call    free_slot
    mov     dword [gw_pending_slot], -1

    jmp     .event_loop

; ======================================================================
;  HELPERS
; ======================================================================

; -----------------------------------------------------------------------
; slot_to_globals(ecx = slot index)
; Copy slot.recv_buf → global recv_buf and slot.req_* → global req_*,
; adjusting pointer fields from slot-space to global-space.
; -----------------------------------------------------------------------
slot_to_globals:
    push    rbp
    mov     rbp, rsp
    push    rbx
    sub     rsp, 32

    slot_addr                              ; r10 = slot addr

    ; delta = &recv_buf - &slot.recv_buf
    lea     rax, [recv_buf]
    lea     rdx, [r10 + ClientSlot.recv_buf]
    sub     rax, rdx
    mov     rbx, rax                       ; save delta

    ; copy recv_buf (req_used bytes)
    mov     rcx, [r10 + ClientSlot.req_used]
    test    rcx, rcx
    jz      .sg_copy_done
    lea     rdi, [recv_buf]
    lea     rsi, [r10 + ClientSlot.recv_buf]
    rep movsb
.sg_copy_done:

    ; scalar req_* fields
    mov     rax, [r10 + ClientSlot.req_used]
    mov     [req_used], rax
    mov     rax, [r10 + ClientSlot.req_header_end]
    mov     [req_header_end], rax
    mov     rax, [r10 + ClientSlot.req_content_length]
    mov     [req_content_length], rax
    mov     eax, [r10 + ClientSlot.req_has_cl]
    mov     [req_has_cl], eax
    mov     eax, [r10 + ClientSlot.req_has_te]
    mov     [req_has_te], eax
    mov     eax, [r10 + ClientSlot.req_cl_count]
    mov     [req_cl_count], eax
    mov     rax, [r10 + ClientSlot.req_method_len]
    mov     [req_method_len], rax
    mov     rax, [r10 + ClientSlot.req_path_len]
    mov     [req_path_len], rax

    ; pointer req_method_ptr: adjust by delta
    mov     rcx, [r10 + ClientSlot.req_method_ptr]
    test    rcx, rcx
    jz      .sg_skip_mptr
    add     rcx, rbx
    mov     [req_method_ptr], rcx
    jmp     .sg_mptr_done
.sg_skip_mptr:
    mov     qword [req_method_ptr], 0
.sg_mptr_done:

    ; pointer req_path_ptr: adjust by delta
    mov     rcx, [r10 + ClientSlot.req_path_ptr]
    test    rcx, rcx
    jz      .sg_skip_pptr
    add     rcx, rbx
    mov     [req_path_ptr], rcx
    jmp     .sg_pptr_done
.sg_skip_pptr:
    mov     qword [req_path_ptr], 0
.sg_pptr_done:

    add     rsp, 32
    pop     rbx
    pop     rbp
    ret

; -----------------------------------------------------------------------
; globals_to_slot(ecx = slot index)
; Reverse of slot_to_globals: copy global → slot, adjust pointers back.
; -----------------------------------------------------------------------
globals_to_slot:
    push    rbp
    mov     rbp, rsp
    push    rbx
    sub     rsp, 32

    slot_addr

    ; delta = &slot.recv_buf - &recv_buf  (inverse of above)
    lea     rax, [r10 + ClientSlot.recv_buf]
    lea     rdx, [recv_buf]
    sub     rax, rdx
    mov     rbx, rax

    ; copy recv_buf back
    mov     rcx, [req_used]
    test    rcx, rcx
    jz      .gs_copy_done
    lea     rdi, [r10 + ClientSlot.recv_buf]
    lea     rsi, [recv_buf]
    rep movsb
.gs_copy_done:

    ; scalar fields
    mov     rax, [req_used]
    mov     [r10 + ClientSlot.req_used], rax
    mov     rax, [req_header_end]
    mov     [r10 + ClientSlot.req_header_end], rax
    mov     rax, [req_content_length]
    mov     [r10 + ClientSlot.req_content_length], rax
    mov     eax, [req_has_cl]
    mov     [r10 + ClientSlot.req_has_cl], eax
    mov     eax, [req_has_te]
    mov     [r10 + ClientSlot.req_has_te], eax
    mov     eax, [req_cl_count]
    mov     [r10 + ClientSlot.req_cl_count], eax
    mov     rax, [req_method_len]
    mov     [r10 + ClientSlot.req_method_len], rax
    mov     rax, [req_path_len]
    mov     [r10 + ClientSlot.req_path_len], rax

    ; pointer req_method_ptr: adjust back
    mov     rcx, [req_method_ptr]
    test    rcx, rcx
    jz      .gs_skip_mptr
    add     rcx, rbx
    mov     [r10 + ClientSlot.req_method_ptr], rcx
    jmp     .gs_mptr_done
.gs_skip_mptr:
    mov     qword [r10 + ClientSlot.req_method_ptr], 0
.gs_mptr_done:

    ; pointer req_path_ptr: adjust back
    mov     rcx, [req_path_ptr]
    test    rcx, rcx
    jz      .gs_skip_pptr
    add     rcx, rbx
    mov     [r10 + ClientSlot.req_path_ptr], rcx
    jmp     .gs_pptr_done
.gs_skip_pptr:
    mov     qword [r10 + ClientSlot.req_path_ptr], 0
.gs_pptr_done:

    add     rsp, 32
    pop     rbx
    pop     rbp
    ret

; -----------------------------------------------------------------------
; disassociate_slot(ecx = slot index)
; Clear WSAEventSelect on slot's socket and restore blocking mode.
; Does NOT close socket or change state.
; -----------------------------------------------------------------------
disassociate_slot:
    push    rbp
    mov     rbp, rsp
    push    rbx
    sub     rsp, 40

    slot_addr
    mov     rbx, r10
    mov     rcx, [rbx + ClientSlot.sock]
    test    rcx, rcx
    jz      .ds_ret
    xor     edx, edx
    xor     r8d, r8d
    call    WSAEventSelect
    mov     rcx, [rbx + ClientSlot.sock]
    mov     edx, FIONBIO
    lea     r8,  [fionbio_arg]
    mov     dword [r8], 0
    call    ioctlsocket
.ds_ret:
    add     rsp, 40
    pop     rbx
    pop     rbp
    ret

; -----------------------------------------------------------------------
; free_slot(ecx = slot index)
; shutdown + closesocket + zero fields + state = FREE.
; -----------------------------------------------------------------------
free_slot:
    push    rbp
    mov     rbp, rsp
    push    rbx
    sub     rsp, 40

    slot_addr
    mov     rbx, r10
    mov     rcx, [rbx + ClientSlot.sock]
    test    rcx, rcx
    jz      .fs_no_sock
    mov     edx, SD_SEND
    call    shutdown
    mov     rcx, [rbx + ClientSlot.sock]
    call    closesocket
.fs_no_sock:
    mov     qword [rbx + ClientSlot.sock], 0
    mov     dword [rbx + ClientSlot.state], CS_FREE
    clear_slot_request rbx

    add     rsp, 40
    pop     rbx
    pop     rbp
    ret

; -----------------------------------------------------------------------
; respond_and_free_slot(ecx = slot index, edx = HTTP status)
; http_respond + log_request + free_slot.
; -----------------------------------------------------------------------
respond_and_free_slot:
    push    rbp
    mov     rbp, rsp
    push    rbx                              ; slot index
    push    r12                              ; status code
    sub     rsp, 32

    mov     rbx, rcx
    mov     r12d, edx

    slot_addr
    mov     rcx, [r10 + ClientSlot.sock]
    mov     edx, r12d
    call    http_respond

    mov     ecx, r12d
    mov     rdx, [resp_body_len]
    call    log_request

    mov     ecx, ebx
    call    free_slot

    add     rsp, 32
    pop     r12
    pop     rbx
    pop     rbp
    ret

; -----------------------------------------------------------------------
; error_and_free_slot(ecx = slot index, eax = HTTP status)
; disassociate + resp_set_error + http_respond + log_request + free_slot.
; -----------------------------------------------------------------------
error_and_free_slot:
    push    rbp
    mov     rbp, rsp
    push    rbx                              ; slot index
    push    r12                              ; status code
    sub     rsp, 32

    mov     rbx, rcx
    mov     r12d, eax

    mov     ecx, ebx
    call    disassociate_slot

    mov     eax, r12d
    call    resp_set_error

    mov     ecx, ebx
    slot_addr
    mov     rcx, [r10 + ClientSlot.sock]
    mov     edx, r12d
    call    http_respond

    mov     ecx, r12d
    mov     rdx, [resp_body_len]
    call    log_request

    mov     ecx, ebx
    call    free_slot

    add     rsp, 32
    pop     r12
    pop     rbx
    pop     rbp
    ret

; -----------------------------------------------------------------------
; scan_timeouts - iterate READING slots; respond 400 on deadline expiry.
; -----------------------------------------------------------------------
scan_timeouts:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    sub     rsp, 32

    call    GetTickCount64
    mov     r12, rax                         ; preserved across logging/response calls

    xor     ebx, ebx
.st_loop:
    mov     ecx, ebx
    slot_addr
    cmp     dword [r10 + ClientSlot.state], CS_READING
    jne     .st_next

    cmp     r12, [r10 + ClientSlot.deadline_tick]
    jb      .st_next

    ; timeout
    lea     rcx, [s_client_timeout]
    mov     rdx, S_CLIENT_TIMEOUT_LEN
    call    log_err

    mov     ecx, ebx
    mov     eax, HTTP_400
    call    error_and_free_slot

.st_next:
    inc     ebx
    cmp     ebx, MAX_CLIENTS
    jb      .st_loop

    add     rsp, 32
    pop     r12
    pop     rbx
    pop     rbp
    ret
