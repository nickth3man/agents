; ===========================================================================
; src/text.asm - bounded byte utilities (PLAN §2.7, §7.2)
; ===========================================================================
%include "win64.inc"

default rel

section .text

; ---------------------------------------------------------------------------
; copy_bytes - tiny memcpy. rcx=dst, r8=src, rdx=len. Leaf (no calls).
; Clobbers: rax,rcx,rdx,r8. Returns: rcx/r8 advanced past end.
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
; Inputs:  RCX = ptr1, RDX = len1, R8 = ptr2, R9 = len2.
; Outputs: RAX = 1 if equal (same length and bytes), else 0.
; Clobbers: RCX,RDX,R8 (and saved RSI,RDI on the equal path). Leaf.
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
; Inputs:  RCX = haystack ptr, RDX = haystack len, R8 = needle ptr,
;          R9 = needle len.
; Outputs: RAX = byte offset of first match, or -1 if not found.
; Clobbers: volatile + saved rbx,rsi.
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
; Inputs/Outputs: same as mem_find.
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
