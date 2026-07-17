; ===========================================================================
; src/text.asm - bounded byte utilities (PLAN §2.7, §7.2)
; ===========================================================================
%include "win64.inc"

default rel

section .text

; ---------------------------------------------------------------------------
; copy_bytes - bounded byte-by-byte copy (tiny memcpy).
; Purpose:        Copies RDX bytes from [R8] to [RCX] in a byte-by-byte loop.
;                 Leaf function (no CALLs). RCX and R8 advanced past the copied
;                 region on return.
; Inputs:         RCX=u8* dst, RDX=usize len, R8=u8* src
; Outputs:        none (no meaningful RAX return; RCX,R8 advanced past end)
; Errors:         none
; Clobbers:       RAX,RCX,RDX,R8 (volatile; RAX holds last byte copied)
; Preserves:      RBX,RBP,RDI,RSI,R9,R10,R11,R12-R15 (all nonvolatile +
;                 untouched volatiles)
; Locals:         0 (leaf; no calls, no frame)
; Max read:       RDX bytes from [R8]
; Max write:      RDX bytes to [RCX]
; Precond:        RDX <= CAP_REQUEST (or caller's buffer bound); [RCX,RDX) and
;                 [R8,RDX) must not overlap; RDX=0 is a no-op
; ---------------------------------------------------------------------------
global copy_bytes
copy_bytes:
    test    rdx, rdx
    jz      .done
.loop:
    mov     al, [r8]
    mov     [rcx], al
    inc     rcx
    inc     r8
    dec     rdx
    jnz     .loop
.done:
    ret

; ---------------------------------------------------------------------------
; bytes_eq - exact equality of two counted byte spans.
; Purpose:        Compares up to min(RDX,R9) bytes from [RCX] and [R8] using
;                 repe cmpsb after checking length equality. Returns 1 if
;                 lengths match AND all bytes equal. Leaf function.
; Inputs:         RCX=u8* ptr1, RDX=usize len1, R8=u8* ptr2, R9=usize len2
; Outputs:        RAX=1 (equal) or 0 (not equal)
; Errors:         none
; Clobbers:       RCX,RDX,R8 (and saved RSI,RDI on the equal path)
; Preserves:      RBX,RBP,RDI,RSI,R12-R15 (all nonvolatile; RSI,RDI saved)
; Locals:         0 (leaf; no calls, no frame)
; Max read:       min(RDX,R9) bytes from [RCX]; min(RDX,R9) bytes from [R8]
; Max write:      0 (none)
; Precond:        RCX,R8 valid for min(RDX,R9) bytes; RDX,R9 may be 0
; ---------------------------------------------------------------------------
global bytes_eq
bytes_eq:
    cmp     rdx, r9
    jne     .ne
    push    rsi
    push    rdi
    mov     rsi, rcx
    mov     rdi, r8
    mov     rcx, rdx                ; count for repe cmpsb
    repe    cmpsb
    sete    al
    movzx   rax, al
    pop     rdi
    pop     rsi
    ret
.ne:
    xor     eax, eax
    ret

; ---------------------------------------------------------------------------
; mem_find - exact substring search.
; Purpose:        Searches [RCX] for the first occurrence of [R8]. Returns the
;                 byte offset of the match or -1. Leaf function.
; Inputs:         RCX=u8* haystack ptr, RDX=usize haystack len, R8=u8* needle
;                 ptr, R9=usize needle len
; Outputs:        RAX = byte offset (0-indexed) of first match, or -1 if not
;                 found
; Errors:         none (RAX=-1 signals not-found, not an error)
; Clobbers:       volatile + saved rbx,rsi
; Preserves:      RBX,RBP,RDI,RSI,R12-R15 (all nonvolatile; RBX,RSI saved)
; Locals:         0 (leaf; no calls, no frame)
; Max read:       RDX bytes from [RCX] (haystack, worst-case full scan); R9
;                 bytes from [R8] (needle)
; Max write:      0 (none)
; Precond:        RCX valid for RDX bytes; R8 valid for R9 bytes; R9=0 is a
;                 no-find (needle of length zero trivially absent)
; ---------------------------------------------------------------------------
global mem_find
mem_find:
    push    rbx
    push    rsi
    mov     rsi, rcx                 ; preserve haystack base
    test    r9, r9
    jz      .nf
    cmp     rdx, r9
    jb      .nf                      ; haystack shorter than needle
    mov     r10, rdx
    sub     r10, r9                  ; r10 = max_i (hay_len - needle_len)
    xor     r11, r11                 ; i = 0
.loop_i:
    cmp     r11, r10
    ja      .nf
    lea     rdx, [rsi + r11]         ; hay + i
    xor     eax, eax                 ; j = 0
.loop_j:
    cmp     rax, r9
    jae     .found
    mov     bl, [rdx + rax]          ; hay[i+j]
    mov     cl, [r8 + rax]           ; needle[j]
    cmp     bl, cl
    jne     .next_i
    inc     rax
    jmp     .loop_j
.next_i:
    inc     r11
    jmp     .loop_i
.found:
    mov     rax, r11
    pop     rsi
    pop     rbx
    ret
.nf:
    mov     rax, -1
    pop     rsi
    pop     rbx
    ret

; ---------------------------------------------------------------------------
; mem_find_ci - case-insensitive (ASCII) substring search.
; Purpose:        Searches [RCX] for the first case-insensitive match of [R8]:
;                 uppercase A-Z is folded to lowercase before comparison.
;                 Same interface and loop structure as mem_find.
; Inputs:         RCX=u8* haystack ptr, RDX=usize haystack len, R8=u8* needle
;                 ptr, R9=usize needle len
; Outputs:        RAX = byte offset (0-indexed) of first match, or -1 if not
;                 found
; Errors:         none (RAX=-1 signals not-found)
; Clobbers:       volatile + saved rbx,rsi
; Preserves:      RBX,RBP,RDI,RSI,R12-R15 (all nonvolatile; RBX,RSI saved)
; Locals:         0 (leaf; no calls, no frame)
; Max read:       RDX bytes from [RCX] (haystack, worst-case full scan); R9
;                 bytes from [R8] (needle)
; Max write:      0 (none)
; Precond:        RCX valid for RDX bytes; R8 valid for R9 bytes; comparison
;                 is ASCII-only (A-Z folded to lowercase)
; ---------------------------------------------------------------------------
global mem_find_ci
mem_find_ci:
    push    rbx
    push    rsi
    mov     rsi, rcx
    test    r9, r9
    jz      .nf
    cmp     rdx, r9
    jb      .nf
    mov     r10, rdx
    sub     r10, r9
    xor     r11, r11
.loop_i:
    cmp     r11, r10
    ja      .nf
    lea     rdx, [rsi + r11]
    xor     eax, eax
.loop_j:
    cmp     rax, r9
    jae     .found
    mov     bl, [rdx + rax]
    mov     cl, [r8 + rax]
    cmp     bl, 'A'
    jb      .c1
    cmp     bl, 'Z'
    ja      .c1
    add     bl, 0x20
.c1:
    cmp     cl, 'A'
    jb      .c2
    cmp     cl, 'Z'
    ja      .c2
    add     cl, 0x20
.c2:
    cmp     bl, cl
    jne     .next_i
    inc     rax
    jmp     .loop_j
.next_i:
    inc     r11
    jmp     .loop_i
.found:
    mov     rax, r11
    pop     rsi
    pop     rbx
    ret
.nf:
    mov     rax, -1
    pop     rsi
    pop     rbx
    ret
