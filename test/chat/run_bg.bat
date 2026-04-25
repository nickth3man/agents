@echo off
cd /d "C:\Users\nicolas\Documents\GitHub\agents\test\chat"
start /b "" ".venv\Scripts\python.exe" "main.py" > "server.log" 2>&1
