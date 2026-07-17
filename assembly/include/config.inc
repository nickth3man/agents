; ===========================================================================
; config.inc - build/run configuration: port, capacities, build switches
; ===========================================================================
; Single source of truth for tunable limits (PLAN §2.7) and build-mode flags
; (PLAN §5.1, §2.5 debug chunk caps). Values are assembly-time constants.
; ---------------------------------------------------------------------------

%ifndef CONFIG_INC
%define CONFIG_INC

; --- Network ---------------------------------------------------------------
%define LISTEN_PORT      8080          ; loopback only (PLAN §1.1)

; --- Buffer capacities (bytes) - PLAN §2.7 --------------------------------
%define CAP_REQUEST      8192          ; request buffer: headers + body
%define CAP_HEADERS      6144          ; max header bytes
%define CAP_CHAT_BODY    2048          ; max /chat body
%define CAP_RESP_HDR     512           ; response-header workspace
%define CAP_INDEX        65536         ; index.html disk-read buffer (DEV_MODE)
%define CAP_LOG_LINE     512           ; one structured log line
%define CAP_GATEWAY_REQ  16384         ; OpenRouter JSON request / decoded reply
%define CAP_GATEWAY_RESP 65536         ; bounded OpenRouter JSON response
%define CAP_GATEWAY_DRAFT 16384        ; first LLM call's analysis
%define CAP_API_KEY      1024          ; OpenRouter credential from environment
%define CAP_MODEL        256           ; model identifier from environment
%define CAP_AUTH_WCHARS  1024          ; UTF-16 request headers

; Debug guards placed immediately around critical static buffers.
%define CANARY_VALUE     0xA55AA55AA55AA55A

; --- Build modes -----------------------------------------------------------
; DEBUG=1 enables forced-small-chunk I/O caps (PLAN §2.5).
; Set DEBUG=0 via command-line -DDEBUG=0 for a release build.
%ifndef DEBUG
%define DEBUG 1
%endif

; Forced small chunk caps (debug only) to exercise split-header/body paths.
%if DEBUG
%define DBG_RECV_CHUNK   7
%define DBG_SEND_CHUNK   13
%endif

; --- Async gateway state machine values (see engine_gateway.asm) ------------
%define GW_IDLE            0
%define GW_OPEN_SESSION    1
%define GW_OPEN_REQUEST    2
%define GW_SEND_PENDING    3
%define GW_RECV_PENDING    4
%define GW_READ_PENDING    5
%define GW_DECODE          6
%define GW_TERMINAL        7

; gateway_advance return values
%define GW_RET_PROGRESS    0
%define GW_RET_DONE        200
%define GW_RET_FAIL        502

; --- Concurrent client handling (PLAN §2.3, per-slot WFMO) ------------------
%define MAX_CLIENTS       32
%define CS_FREE           0
%define CS_READING        1
%define CS_CHAT_PENDING   2
%define CLIENT_TIMEOUT_MS 8000

; --- Per-slot client state (must be 8-byte aligned, TOTAL a multiple of 8) ---
; Each slot has its own socket, WSA event, deadline, request state, and
; guard-protected recv_buf.  The global recv_buf / req_* variables serve as
; the "active slot" scratch — data is copied in before read_more_request /
; route_request and copied out afterwards via copy_slot_*/copy_globals_*
; helpers in start.asm.
struc ClientSlot
  .sock             resq 1   ; 0
  .event            resq 1   ; 8
  .state            resd 1   ; 16 (CS_FREE/CS_READING/CS_CHAT_PENDING)
  .pad              resd 1   ; 20 (align to 8)
  .deadline_tick    resq 1   ; 24 (GetTickCount64 + CLIENT_TIMEOUT_MS)
  .req_id           resq 1   ; 32
  .req_used         resq 1   ; 40
  .req_header_end   resq 1   ; 48
  .req_content_length resq 1 ; 56
  .req_method_ptr   resq 1   ; 64 (pointer into .recv_buf)
  .req_method_len   resq 1   ; 72
  .req_path_ptr     resq 1   ; 80 (pointer into .recv_buf)
  .req_path_len     resq 1   ; 88
  .req_has_cl       resd 1   ; 96
  .req_has_te       resd 1   ; 100
  .req_cl_count     resd 1   ; 104
  .pad2             resd 1   ; 108 (align to 8)
  .guard_lo         resq 1   ; 112
  .recv_buf         resb CAP_REQUEST  ; 120
  .guard_hi         resq 1   ; 120 + CAP_REQUEST
endstruc

CLIENT_SLOT_SIZE equ ClientSlot_size

%endif ; CONFIG_INC
