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
extern WinHttpSetStatusCallback  ; HINTERNET, WINHTTP_STATUS_CALLBACK, DWORD flags, DWORD_PTR -> old callback

%define WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY 4
%define WINHTTP_FLAG_SECURE                 0x00800000
%define WINHTTP_FLAG_ASYNC                  0x10000000

; --- Status query codes ----------------------------------------------------
%define WINHTTP_QUERY_STATUS_CODE           19
%define WINHTTP_QUERY_FLAG_NUMBER           0x20000000

; --- Async callback notification flags ------------------------------------
%define WINHTTP_CALLBACK_FLAG_ALL_NOTIFICATIONS      0xFFFFFFFF
%define WINHTTP_CALLBACK_STATUS_HANDLE_CREATED       0x00000400
%define WINHTTP_CALLBACK_STATUS_HANDLE_CLOSING       0x00000800
%define WINHTTP_CALLBACK_STATUS_RESPONSE_RECEIVED    0x00000080   ; informational
%define WINHTTP_CALLBACK_STATUS_HEADERS_AVAILABLE    0x00020000   ; ReceiveResponse done
%define WINHTTP_CALLBACK_STATUS_DATA_AVAILABLE       0x00040000   ; QueryDataAvailable done
%define WINHTTP_CALLBACK_STATUS_READ_COMPLETE        0x00080000   ; ReadData done
%define WINHTTP_CALLBACK_STATUS_REQUEST_ERROR        0x00200000
%define WINHTTP_CALLBACK_STATUS_SENDREQUEST_COMPLETE 0x00400000
%define WINHTTP_CALLBACK_STATUS_SECURE_FAILURE       0x00010000

; --- Sentinel --------------------------------------------------------------
%define WINHTTP_INVALID_STATUS_CALLBACK 0xFFFFFFFFFFFFFFFF

%endif
