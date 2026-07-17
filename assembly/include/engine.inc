; ===========================================================================
; include/engine.inc - async Gateway interface (PLAN §2.10, async rewrite)
; ===========================================================================
%ifndef ENGINE_INC
%define ENGINE_INC

; gateway_start - initiate an async /chat gateway round.
;   Input:  RCX = message pointer, RDX = message byte length.
;   Output: RAX = 0 on success (gateway now busy), 1 on failure/busy.
;   The caller MUST check gw_state after this to determine if a response
;   was set. On success, gw_state becomes GW_OPEN_SESSION.
extern gateway_start

; gateway_advance - called from the event loop after gw_event signals,
;   or immediately after gateway_start to begin the WinHTTP chain.
;   Input:  none (works on gw_state globals).
;   Output: RAX = 0 if still in progress, 200 on success (resp_* set),
;           502 on failure.
;   After RAX != 0, the caller should http_respond on gw_pending_slot socket, RAX.
extern gateway_advance

; gw_callback - WinHTTP asynchronous status callback.
;   Registered once with WinHttpSetStatusCallback on hSession at session
;   creation. Running on a WinHTTP worker thread — only writes DWORD
;   shared state and calls SetEvent. Must preserve non-volatile regs.
extern gw_callback

%endif ; ENGINE_INC
