; ===========================================================================
; src/decimal.asm - checked decimal formatting (PLAN §2.7, supports logging)
; ===========================================================================
; Unsigned 32-bit -> ASCII decimal. Parsing lands in Milestone 3.
; Leaf routine (no API calls).
; ---------------------------------------------------------------------------

%include "win64.inc"

section .text

; ---------------------------------------------------------------------------
; u32_to_dec - format a 32-bit unsigned integer as ASCII decimal.
; ---------------------------------------------------------------------------
; Inputs:  ECX = value (unsigned 32-bit), RDX = out buffer ptr,
;          R8  = buffer capacity (bytes).
; Outputs: RAX = bytes written (0 on capacity error; nothing written).
; Clobbers: RAX,RCX,RDX,R8,R9,R10,R11.  Preserves: RBX,RSI (saved).
; Max write: up to 10 bytes into [RDX..RDX+len).
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
