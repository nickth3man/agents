; ===========================================================================
; src/assets.asm - embedded immutable data: chat page + build ID (PLAN §2.8)
; ===========================================================================
%include "win64.inc"
%include "generated/version.inc"

default rel

section .data
; --- embedded browser page (build-time inclusion; never read at runtime) ----
global chat_html
chat_html:
    incbin  "web/chat.html"
global chat_html_len
chat_html_len dq $ - chat_html        ; assembly-time length, no runtime scan

; --- embedded build id (PLAN §2.9) -----------------------------------------
global build_id
build_id:
    db      BUILD_ID
global build_id_len
build_id_len dq $ - build_id

; --- /health endpoint body (static JSON: status + build + listen) -----------
global health_json
health_json:
    db      '{"status":"ok","build":"', BUILD_ID, '","listen":"127.0.0.1:8080"}'
global health_json_len
health_json_len dq $ - health_json

%ifdef DEV_MODE

%include "config.inc"

; --- WinAPI declarations for disk read ------------------------------------
extern CreateFileA
extern ReadFile
extern GetFileSizeEx
extern CloseHandle

; --- WinAPI constants -----------------------------------------------------
%define GENERIC_READ            0x80000000
%define FILE_SHARE_READ         0x00000001
%define OPEN_EXISTING           3
%define FILE_ATTRIBUTE_NORMAL   0x80

section .data
index_path: db "web\chat.html",0

section .bss
index_buf: resb CAP_INDEX

section .text
; ---------------------------------------------------------------------------
; load_index_html - load web/chat.html from disk into index_buf (DEV_MODE)
; Purpose:        Reads the web/chat.html file from disk into the static
;                 index_buf. Falls back safely on any error. Only available
;                 in DEV_MODE builds.
; Inputs:         none (operates on index_path and index_buf globals)
; Outputs:        RAX=index_buf ptr (u8*) on success, RDX=bytes read (usize);
;                 CF=0 on success, CF=1 on error
; Errors:         CF=1 on file-not-found, empty-file, too-large (>CAP_INDEX),
;                 or read-failure; RAX,RDX undefined on error
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11 (volatile)
; Preserves:      RBX,RBP,RDI,RSI,R12-R15 (nonvolatile; RBX=hFile, R12=size)
; Locals:         64 (32 shadow + 24 stack args + 8 temp)
; Max read:       CAP_INDEX bytes from disk via ReadFile into [index_buf]
; Max write:      CAP_INDEX bytes to [index_buf] via ReadFile
; Precond:        DEV_MODE enabled; index_buf in .bss sized CAP_INDEX
; ---------------------------------------------------------------------------
global load_index_html
load_index_html:
    push    rbp
    mov     rbp, rsp
    push    rbx                     ; save hFile
    push    r12                     ; save file size / bytes read
    sub     rsp, 64                 ; shadow(32)+3 stack args(24)+temp(8); aligned
    ; Step 1: CreateFileA(index_path, GENERIC_READ, FILE_SHARE_READ, NULL,
    ;                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL)
    lea     rcx, [rel index_path]
    mov     edx, GENERIC_READ
    mov     r8d, FILE_SHARE_READ
    xor     r9d, r9d                ; lpSecurityAttributes = NULL
    mov     qword [rsp+32], OPEN_EXISTING
    mov     qword [rsp+40], FILE_ATTRIBUTE_NORMAL
    mov     qword [rsp+48], 0       ; hTemplateFile = NULL
    call    CreateFileA
    cmp     rax, -1                 ; INVALID_HANDLE_VALUE
    je      .err_ret
    mov     rbx, rax                ; hFile
    ; Step 2: GetFileSizeEx(hFile, &size)
    mov     rcx, rbx
    lea     rdx, [rsp+56]           ; LARGE_INTEGER output
    call    GetFileSizeEx
    test    eax, eax
    jz      .err_close
    mov     r12, [rsp+56]           ; file size
    test    r12, r12
    jz      .err_close              ; empty file
    cmp     r12, CAP_INDEX
    ja      .err_close              ; too large
    ; Step 3: ReadFile(hFile, index_buf, size, &bytesRead, NULL)
    mov     rcx, rbx                ; hFile
    lea     rdx, [rel index_buf]
    mov     r8d, r12d               ; nNumberOfBytesToRead
    lea     r9,  [rsp+56]           ; lpNumberOfBytesRead (DWORD)
    mov     qword [rsp+32], 0       ; lpOverlapped = NULL
    call    ReadFile
    test    eax, eax
    jz      .err_close              ; ReadFile failed
    mov     rdx, [rsp+56]           ; bytesRead (zero-extended from DWORD)
    cmp     rdx, r12
    jne     .err_close              ; short read
    ; Step 4: CloseHandle(hFile); success
    mov     rcx, rbx
    call    CloseHandle
    lea     rax, [rel index_buf]
    clc                             ; CF=0 = success
    jmp     .out
.err_close:
    mov     rcx, rbx
    call    CloseHandle
.err_ret:
    stc                             ; CF=1 = error
.out:
    add     rsp, 64
    pop     r12
    pop     rbx
    pop     rbp
    ret

%endif  ; DEV_MODE
