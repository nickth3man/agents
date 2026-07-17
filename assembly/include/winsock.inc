; ===========================================================================
; winsock.inc - Winsock 2.2 constants + struct layouts (PLAN §2.3, Milestone 2)
; ===========================================================================
%ifndef WINSOCK_INC
%define WINSOCK_INC

%include "win64.inc"

; --- ws2_32.dll imported functions ----------------------------------------
extern WSAStartup           ; RCX=WORD wVersionRequired, RDX=&WSADATA -> RAX=int
extern WSACleanup           ; () -> RAX=int (0 ok, SOCKET_ERROR fail)
extern WSAGetLastError      ; () -> RAX=int error code
extern socket               ; RCX=af RDX=type R8=protocol -> RAX=SOCKET
extern setsockopt           ; RCX=sock RDX=level R8=optname R9=optval [rsp+32]=optlen
extern bind                 ; RCX=sock RDX=&sockaddr R8=namelen -> RAX=int
extern listen               ; RCX=sock RDX=backlog -> RAX=int
extern accept               ; RCX=sock RDX=&addr R8=&addrlen -> RAX=SOCKET
extern connect              ; RCX=sock RDX=&sockaddr R8=len -> RAX=int
extern recv                 ; RCX=sock RDX=buf R8=len R9=flags -> RAX=int
extern send                 ; RCX=sock RDX=buf R8=len R9=flags -> RAX=int
extern closesocket          ; RCX=sock -> RAX=int
extern ioctlsocket          ; RCX=sock, RDX=cmd (FIONBIO), R8=&u_long arg -> int
extern shutdown             ; RCX=sock RDX=how -> RAX=int
extern htons                ; RCX=u_short -> RAX=u_short (network order)

; --- Families / types / protocols -----------------------------------------
%define AF_INET         2
%define SOCK_STREAM     1
%define IPPROTO_TCP     6

; --- Socket option levels / names -----------------------------------------
%define SOL_SOCKET          0xffff
%define SO_REUSEADDR        0x0004
%define SO_EXCLUSIVEADDRUSE 0xFFFFFFFB      ; ((int)(~SO_REUSEADDR)) = -5
%define SO_RCVTIMEO         0x1006          ; DWORD ms (5th arg style: via setsockopt)
%define SO_SNDTIMEO         0x1005          ; DWORD ms

; --- Winsock event objects (async-accept support) -------------------------
extern WSACreateEvent          ; () -> WSAEVENT
extern WSAEnumNetworkEvents    ; SOCKET, WSAEVENT, LPWSANETWORKEVENTS -> int
extern WSAEventSelect          ; SOCKET, WSAEVENT, LONG events -> int

; --- FD_XXX event masks for WSAEventSelect ---------------------------------
%define FD_READ             0x0001
%define FD_WRITE            0x0002
%define FD_ACCEPT           0x0008
%define FD_CLOSE            0x0020

; --- ioctlsocket commands --------------------------------------------------
%define FIONBIO             0x8004667E      ; set/clear non-blocking mode

; --- WSANETWORKEVENTS struct sizes (for .bss reservation) ------------------
;   struct { long lNetworkEvents; int iErrorCode[FD_MAX_EVENTS=10]; }
%define WSANETWORKEVENTS_SIZE 44     ; 4 + 10*4

; --- Error codes -----------------------------------------------------------
%define WSAEWOULDBLOCK     10035

; --- Winsock version + sentinels ------------------------------------------
%define WINSOCK_VER         0x0202          ; MAKEWORD(2,2)
%define INVALID_SOCKET      0xFFFFFFFFFFFFFFFF
%define SOCKET_ERROR        -1              ; returned as int (EAX)

; shutdown() "how" values
%define SD_RECV             0
%define SD_SEND             1
%define SD_BOTH             2

; --- Common WSAGetLastError codes -----------------------------------------
%define WSAETIMEDOUT        10060
%define WSAEADDRINUSE       10048
%define WSAECONNRESET       10054

; --- sockaddr_in layout (16 bytes) ----------------------------------------
%define SIN_FAMILY_OFF      0               ; WORD  (ADDRESS_FAMILY)
%define SIN_PORT_OFF        2               ; WORD  (network byte order)
%define SIN_ADDR_OFF        4               ; DWORD (network byte order)
%define SIN_ZERO_OFF        8               ; 8 bytes padding
%define SOCKADDR_IN_SIZE    16

; Loopback address constants, already in network byte order:
;   port 8080 = 0x1F90 -> htons -> store WORD 0x901F (memory: 1F 90)
;   ip 127.0.0.1 -> store DWORD 0x0100007F (memory: 7F 00 00 01 = 127.0.0.1)
%define PORT_8080_NET       0x901F
%define IP_127_0_0_1_NET    0x0100007F

; Relay (PLAN §2.10, Milestone 8): localhost text/plain -> OpenRouter.
%define RELAY_PORT          8081
%define PORT_8081_NET       0x911F      ; htons(8081)

; --- WSADATA layout (Win64, 402 bytes total) ------------------------------
%define WSADATA_SIZE        402

%endif ; WINSOCK_INC
