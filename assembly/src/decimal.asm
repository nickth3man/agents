; ===========================================================================
; src/decimal.asm - checked decimal formatting (PLAN §2.7, supports logging)
; ===========================================================================
; Unsigned 32-bit -> ASCII decimal. Parsing lands in Milestone 3.
; Leaf routine (no API calls).
; ---------------------------------------------------------------------------

%include "win64.inc"

default rel

section .text

; ---------------------------------------------------------------------------
; u32_to_dec - format a 32-bit unsigned integer as ASCII decimal
; Purpose:        Converts an unsigned 32-bit integer to its decimal ASCII
;                 representation at [RDX]. Returns bytes written, or 0 if the
;                 buffer is too small. Leaf (no API calls).
; Inputs:         ECX=value (u32), RDX=out buffer ptr (u8*),
;                 R8=buffer capacity (usize)
; Outputs:        RAX=bytes written (usize), or 0 on capacity error
; Errors:         RAX=0 on buffer too small (R8 < digit count)
; Clobbers:       RAX,RCX,RDX,R8,R9,R10,R11 (volatile)
; Preserves:      RBX,RBP,RDI,RSI,R12-R15 (nonvolatile; RBX=out ptr,
;                 RSI=cursor saved via push)
; Locals:         32 (digit scratch buffer; leaf, no shadow)
; Max read:       0   Max write: up to 10 bytes to [RDX] (checked vs R8)
; ---------------------------------------------------------------------------
global u32_to_dec
u32_to_dec:
    push    rbp
    mov     rbp, rsp
    push    rbx                     ; preserve (hold original out ptr)
    push    rsi                     ; preserve (hold out cursor)
    sub     rsp, 32                 ; 16-byte digit scratch + slack
    test    r8, r8                  ; cap == 0 ?
    jz      .fail
    mov     rbx, rdx                ; rbx = original out (for length calc)
    mov     rsi, rdx                ; rsi = cursor
    lea     r10, [rbp-32]           ; r10 = scratch base ([rsp..rsp+15])
    xor     r9d, r9d                ; r9 = digit count
    mov     eax, ecx                ; eax = value
    test    eax, eax
    jnz     .convert
    mov     byte [r10], '0'         ; value == 0 special case
    mov     r9d, 1
    jmp     .emit
.convert:
    mov     r11d, 10                ; divisor
.cloop:
    test    eax, eax
    jz      .emit
    xor     edx, edx
    div     r11d                    ; edx = eax%10, eax = eax/10 (unsigned)
    add     dl, '0'
    mov     [r10 + r9], dl          ; LSB-first at scratch[count]
    inc     r9d
    jmp     .cloop
.emit:
    cmp     r8d, r9d                ; cap >= count ?
    jb      .fail
    mov     r8, r9                  ; r8 = remaining digits (cap no longer needed)
    lea     rcx, [r10 + r9 - 1]     ; rcx -> scratch[count-1] (MSB)
.eloop:
    test    r8, r8
    jz      .done
    mov     al, [rcx]
    mov     [rsi], al
    inc     rsi
    dec     rcx
    dec     r8
    jmp     .eloop
.done:
    mov     rax, rsi
    sub     rax, rbx                ; length written
    add     rsp, 32
    pop     rsi
    pop     rbx
    pop     rbp
    ret
.fail:
    xor     eax, eax
    add     rsp, 32
    pop     rsi
    pop     rbx
    pop     rbp
    ret

; ---------------------------------------------------------------------------
; parse_u32 - parse an unsigned decimal ASCII span into a 32-bit value
; Purpose:        Parses a decimal ASCII string of length RDX at [RCX] into
;                 a 32-bit unsigned integer. Rejects empty strings, strings
;                 longer than 10 digits, non-digit characters, and values
;                 exceeding 0xFFFFFFFF. Leaf (no API calls).
; Inputs:         RCX=ptr to ASCII decimal string (u8*), RDX=len (usize, >=1)
; Outputs:        RAX=parsed value (u32) on success; CF=0 on success, CF=1 on error
; Errors:         CF=1 on empty span, >10 digits, non-digit byte, or
;                 value > 0xFFFFFFFF
; Clobbers:       RAX,RCX,R8,R9 (volatile)
; Preserves:      RBX,RBP,RDI,RSI,R12-R15 (nonvolatile; none touched)
; Locals:         0 (leaf, no frame, no shadow)
; Max read:       up to RDX bytes from [RCX] (input string; max 10)
; Max write:      0 (returns value in RAX; no writes through ptr args)
; ---------------------------------------------------------------------------
global parse_u32
parse_u32:
    test    rdx, rdx
    jz      .p_err
    cmp     rdx, 10
    ja      .p_err                  ; >10 digits cannot fit a u32
    mov     r8, rcx                 ; r8 = cursor
    mov     r9, rdx                 ; r9 = remaining
    xor     rax, rax                ; value = 0
.p_loop:
    test    r9, r9
    jz      .p_ok
    mov     cl, [r8]                ; rcx is free (ptr saved in r8)
    cmp     cl, '0'
    jb      .p_err
    cmp     cl, '9'
    ja      .p_err
    sub     cl, '0'
    imul    rax, rax, 10
    movzx   rcx, cl                 ; rcx = digit
    add     rax, rcx
    inc     r8
    dec     r9
    jmp     .p_loop
.p_ok:
    mov     rcx, 0xFFFFFFFF
    cmp     rax, rcx
    ja      .p_err
    clc
    ret
.p_err:
    xor     eax, eax                ; zero first (xor CLEARS CF!)
    stc                             ; then set carry (must be LAST flag op)
    ret
