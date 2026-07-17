; ===========================================================================
; include/http.inc - HTTP subset constants (PLAN §2.4-2.6, Milestone 3+)
; ===========================================================================
%ifndef HTTP_INC
%define HTTP_INC

; --- Response status codes -------------------------------------------------
%define HTTP_200   200
%define HTTP_400   400
%define HTTP_404   404
%define HTTP_405   405
%define HTTP_411   411
%define HTTP_413   413
%define HTTP_431   431
%define HTTP_500   500
%define HTTP_501   501
%define HTTP_502   502
%define HTTP_503   503

; --- Reader state machine (PLAN §2.4) --------------------------------------
%define ST_HEADERS    0     ; accumulating until CRLFCRLF
%define ST_BODY       1     ; accumulating until header_end + content_length
%define ST_COMPLETE   2     ; ready to route

; --- Byte constants --------------------------------------------------------
%define CR    13
%define LF    10
%define SP    32

%endif ; HTTP_INC
