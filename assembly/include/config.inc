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

%endif ; CONFIG_INC
