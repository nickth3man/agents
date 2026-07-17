; ===========================================================================
; winapi.inc - Kernel32 declarations + constants (PLAN §2.2, M1)
; ===========================================================================
; Only the documented Windows API DLLs are imported (PLAN §1.1):
;   kernel32.dll - process exit, console handles, file/logging primitives
;   ws2_32.dll   - added in Milestone 2 (declared in winsock.inc)
; No C runtime is linked. No import libraries are used; GoLink resolves
; externals directly from the listed DLLs' export tables.
; ---------------------------------------------------------------------------

%ifndef WINAPI_INC
%define WINAPI_INC

%include "win64.inc"

; --- Kernel32 imported functions ------------------------------------------
extern GetStdHandle       ; DWORD  nStdHandle (rcx)            -> RAX = handle
extern WriteFile          ; HANDLE,ptr,DWORD,ptr DWORD,ptr OVERLAPPED
extern ExitProcess        ; UINT uExitCode (rcx)               -> does not return
extern GetEnvironmentVariableA ; LPCSTR name, LPSTR buffer, DWORD size

; --- Async event / wait primitives (kernel32) ------------------------------
extern CreateEventW          ; LPSECURITY_ATTRIBUTES, BOOL manualReset, BOOL initState, LPCWSTR name -> HANDLE
extern SetEvent              ; HANDLE -> BOOL
extern ResetEvent            ; HANDLE -> BOOL
extern WaitForMultipleObjects  ; DWORD count, HANDLE* handles, BOOL waitAll, DWORD ms -> DWORD
extern GetLastError          ; () -> DWORD error code
extern Sleep                 ; DWORD milliseconds -> VOID

; --- Standard handle pseudo-values (passed to GetStdHandle) ----------------
%define STD_INPUT_HANDLE  -10
%define STD_OUTPUT_HANDLE -11
%define STD_ERROR_HANDLE  -12

; --- Generic status --------------------------------------------------------
%define STATUS_SUCCESS 0
%define STATUS_FAIL    1

; --- Error codes from winerror.h ------------------------------------------
%define ERROR_IO_PENDING 997          ; 0x3E5

; --- WaitForMultipleObjects return values ----------------------------------
%define INFINITE        0xFFFFFFFF
%define WAIT_OBJECT_0   0
%define WAIT_TIMEOUT    0x00000102
%define WAIT_FAILED     0xFFFFFFFF

%endif ; WINAPI_INC
