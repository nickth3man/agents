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
extern GetTickCount64        ; () -> ULONGLONG
extern Sleep                 ; DWORD milliseconds -> VOID

; --- File I/O (kernel32) - used by DEV_MODE disk-served assets --------------
extern CreateFileA           ; LPCSTR name, DWORD access, DWORD share, LPSEC, DWORD disp, DWORD attr, HANDLE tmpl -> HANDLE
extern ReadFile              ; HANDLE, LPVOID buf, DWORD nBytes, LPDWORD bytesRead, LPVOID overlapped -> BOOL
extern GetFileSizeEx         ; HANDLE, PLARGE_INTEGER li -> BOOL
extern CloseHandle           ; HANDLE hObject -> BOOL

; --- Process management (kernel32) - used by the asm test harness -----------
extern CreateProcessW        ; LPCSTR app, LPWSTR cmd, LPSEC, LPSEC, BOOL inh, DWORD flags, LPVOID env, LPCWSTR cwd, LPSTARTUPINFOW, LPPROCESS_INFORMATION -> BOOL
extern TerminateProcess      ; HANDLE hProcess, UINT uExitCode -> BOOL
extern WaitForSingleObject   ; HANDLE hHandle, DWORD milliseconds -> DWORD
extern GetExitCodeProcess    ; HANDLE hProcess, LPDWORD lpExitCode -> BOOL

; --- CreateFileA access/share/disposition constants -------------------------
%define GENERIC_READ             0x80000000
%define FILE_SHARE_READ          0x00000001
%define OPEN_EXISTING            3
%define FILE_ATTRIBUTE_NORMAL    0x00000080
%define INVALID_HANDLE_VALUE     -1

; --- CreateProcessW creation flags ------------------------------------------
%define CREATE_NO_WINDOW         0x08000000
%define DETACHED_PROCESS         0x00000008

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
