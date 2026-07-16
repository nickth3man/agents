; ===========================================================================
; include/engine.inc - stable response-engine boundary (PLAN §2.10)
; ===========================================================================
%ifndef ENGINE_INC
%define ENGINE_INC

; gateway_generate
;   Input:  RCX = message pointer, RDX = message byte length.
;   Output: RAX = 0 on success; response pointer/length globals are populated.
;           RAX = HTTP_502 on relay/connect/timeout/framing/capacity failure.
;   Bounds: input <= CAP_CHAT_BODY; response <= CAP_GATEWAY_RESP.
extern gateway_generate

%endif ; ENGINE_INC
