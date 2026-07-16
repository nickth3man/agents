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
