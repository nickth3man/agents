; ===========================================================================
; winhttp.inc - Windows HTTP Services declarations used by the LLM client.
; ===========================================================================
%ifndef WINHTTP_INC
%define WINHTTP_INC

extern WinHttpOpen
extern WinHttpSetTimeouts
extern WinHttpConnect
extern WinHttpOpenRequest
extern WinHttpSendRequest
extern WinHttpReceiveResponse
extern WinHttpQueryHeaders
extern WinHttpReadData
extern WinHttpCloseHandle

%define WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY 4
%define WINHTTP_FLAG_SECURE                 0x00800000
%define WINHTTP_QUERY_STATUS_CODE           19
%define WINHTTP_QUERY_FLAG_NUMBER           0x20000000

%endif
