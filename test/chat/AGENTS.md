*Use `rg` in bash instead of the glob or grep tools.*

## Running the Debate Server in Background

To start the PocketFlow Gradio debate server and keep it running in the background (so you can continue using the terminal):

```bash
# Start server with nohup (redirects output to server.log)
nohup .venv/Scripts/python.exe main.py > server.log 2>&1 < /dev/null &

# Verify it's running
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:7860/
# Should return: 200

# Check logs
tail -f server.log
```

**Key fixes required for background operation:**
1. **SSE server thread must be non-daemon** (`daemon=False` in `start_sse_server()`) - otherwise the thread exits when main thread returns
2. **Keep-alive loop after `demo.launch()`** - prevents main thread from exiting immediately
3. **Use `nohup`** - ignores SIGHUP so process survives terminal closure

**To stop the server:**
```bash
# Find and kill python processes on ports 7860-7865
for pid in $(netstat -ano 2>/dev/null | grep -E "LISTEN.*786[0-9]" | awk '{print $NF}' | sort -u); do
  taskkill //F //PID $pid 2>/dev/null
done
```