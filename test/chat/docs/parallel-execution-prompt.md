# Implementation Prompt: Parallel Model Execution & Lane-Aware Streaming

## Executive Summary

Transform the sequential PocketFlow debate pipeline into a **parallel-first architecture** where Model A, B, and C execute concurrently, streaming tokens to their respective UI lanes in real-time. Reduce wall-clock time from **51.8s to ~22s** (theoretical max: max(2.4, 19.2, 4.6) + 15.9 + 8.8 ≈ 27.5s, but with streaming overlap, perceived latency drops dramatically).

**Current Pipeline (Sequential):**
```
Prepare (0.7s) → Model A (2.4s) → Model B (19.2s) → Model C (4.6s) → Debate (15.9s) → Judge (8.8s) → SendFinal
Total: ~51.8s
```

**Target Pipeline (Parallel Models + Streaming):**
```
Prepare (0.7s) → [Model A ‖ Model B ‖ Model C] (max ~19.2s, all stream concurrently)
                → Debate (15.9s) → Judge (8.8s) → SendFinal
Total: ~44s (but UI sees tokens from all 3 models within seconds)
```

---

## Architecture Overview

### Core Design Principles

1. **Lane-Aware Streaming Protocol**: Each token is tagged with a `lane` identifier so the SSE handler can route it to the correct UI lane without relying on a single global `currentPhase`.
2. **PocketFlow Node Preservation**: Keep the `Node` framework intact. Introduce a single `ParallelModelsNode` that internally orchestrates concurrent execution using `ThreadPoolExecutor`.
3. **Trace Context Propagation**: Serialize OpenTelemetry context before thread submission and re-attach in each model worker.
4. **Graceful Degradation**: If Model B fails after 19s, Models A and C results are still collected and passed to Debate Round (which already handles missing responses with `"No response"`).
5. **Zero Breaking Changes**: The existing `call_llm_stream()` signature is extended, not replaced. Fallback behavior remains identical when `lane_id=None`.

### File Change Matrix

| File | Changes | Lines |
|------|---------|-------|
| `flow.py` | Restructure graph: sequential models → single parallel node | ~10 |
| `nodes.py` | Add `ParallelModelsNode`; extract model configs; keep existing nodes as mixins | ~120 |
| `utils/call_llm.py` | Add `lane_id` parameter; wrap tokens with lane routing prefix | ~15 |
| `main.py` | Update SSE handler for lane parsing; update UI JS for multi-lane tokens; update phase maps | ~80 |

---

## 1. Lane-Aware Streaming Protocol

### 1.1 Token Wrapping Convention

Introduce a **lane routing prefix** that wraps each token before it enters the shared `stream_queue`:

```python
# In utils/call_llm.py
LANE_TOKEN_PREFIX = "\x00LANE:"
LANE_TOKEN_SUFFIX = "\x00"
```

When `call_llm_stream(..., lane_id="model_a")` is called, each token is wrapped as:
```
\x00LANE:model_a\x00This is the actual token content...
```

### 1.2 SSE Handler Parsing

In `main.py`, extend the SSE handler to detect lane-wrapped tokens:

```python
# Add alongside existing PHASE_SENTINEL constants
LANE_TOKEN_PREFIX = "\x00LANE:"
LANE_TOKEN_SUFFIX = "\x00"

# Inside the token processing loop:
if isinstance(token, str) and token.startswith(LANE_TOKEN_PREFIX):
    rest = token[len(LANE_TOKEN_PREFIX):]
    sep_idx = rest.find(LANE_TOKEN_SUFFIX)
    if sep_idx != -1:
        lane = rest[:sep_idx]
        actual_token = rest[sep_idx + len(LANE_TOKEN_SUFFIX):]
        payload = json.dumps({"token": actual_token, "lane": lane})
        self.wfile.write(f"data: {payload}\n\n".encode())
        self.wfile.flush()
        continue
```

### 1.3 Frontend Lane Routing

In the SSE_JS, modify the `onmessage` handler to use lane-aware appending:

```javascript
source.onmessage = function(e) {
  try {
    var d = JSON.parse(e.data);
    if (d.token != null) {
      if (d.lane) {
        appendTokenToLane(d.lane, d.token);
        setLane(d.lane, 'active');
      } else {
        appendToken(d.token);
      }
      setStatus('Live · streaming', 'live');
    }
  } catch(_) {}
};
```

Add the `appendTokenToLane` function:

```javascript
function appendTokenToLane(lane, token) {
  var id = LANE_IDS[lane];
  if (!id) return;
  var el = document.getElementById(id);
  if (!el) return;
  var s = el.querySelector('.lane-stream');
  if (!s) return;
  s.textContent += token;
  s.scrollTop = s.scrollHeight;
}
```

**Critical**: Modify `startPhase` to support multiple simultaneous active phases. Remove the auto-deactivation of the previous phase:

```javascript
function startPhase(phase) {
  if (phase === 'reset') {
    clearAll();
    return;
  }
  if (_runComplete) {
    clearAll();
    _runComplete = false;
  }
  // REMOVED: do NOT deactivate previous phases here
  setLane(phase, 'active');
}
```

Add explicit `endPhase` support via a new SSE event type:

```javascript
source.addEventListener('end_phase', function(e) {
  try {
    var d = JSON.parse(e.data);
    if (d.phase) setLane(d.phase, 'done');
  } catch(_) {}
});
```

And in the Python SSE handler, send end_phase events:
```python
# When ending a phase:
payload = json.dumps({"phase": phase_name})
self.wfile.write(f"event: end_phase\ndata: {payload}\n\n".encode())
self.wfile.flush()
```

---

## 2. ParallelModelsNode Implementation

### 2.1 Node Design

Create a new node class `ParallelModelsNode` in `nodes.py` that replaces the sequential chain of `ModelAResponse >> ModelBResponse >> ModelCResponse`.

```python
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from pocketflow import Node
from opentelemetry import context

from utils.call_llm import call_llm_stream
from utils.conversation import load_conversation, save_conversation
from utils.observability import get_tracer, logger as log, serialize_trace_context, extract_trace_context


class ParallelModelsNode(Node):
    """
    Execute Model A, B, and C in parallel using ThreadPoolExecutor.
    
    Each model streams tokens to the UI via lane-aware stream queues.
    Results are collected and saved to conversation state before returning.
    """
    
    # Reuse the system prompts from existing model nodes
    MODEL_CONFIGS = {
        "model_a": {
            "system_prompt": ModelAResponse.SYSTEM_PROMPT,
            "prompt_builder": lambda ctx: (
                f"{ctx}\n\n### YOUR TASK\n"
                "Provide your best answer to the user's question. Be direct, practical, and actionable.\n"
                "Keep your response concise but complete."
            ),
        },
        "model_b": {
            "system_prompt": ModelBResponse.SYSTEM_PROMPT,
            "prompt_builder": lambda ctx: (
                f"{ctx}\n\n### YOUR TASK\n"
                "Provide your best answer to the user's question, but be sure to:\n"
                "1. Identify key assumptions you're making\n"
                "2. Point out edge cases or exceptions the user should consider\n"
                "3. Highlight any risks or caveats\n"
                "4. Suggest what information might be missing for a complete answer\n\n"
                "Keep your response concise but thorough."
            ),
        },
        "model_c": {
            "system_prompt": ModelCResponse.SYSTEM_PROMPT,
            "prompt_builder": lambda ctx: (
                f"{ctx}\n\n### YOUR TASK\n"
                "Provide your best answer to the user's question from a unique or alternative perspective.\n"
                "Consider:\n"
                "1. Are there non-obvious approaches the user hasn't considered?\n"
                "2. Can you reframe the problem in a useful way?\n"
                "3. What creative solutions exist beyond the conventional answer?\n"
                "4. How might different contexts or stakeholders view this differently?\n\n"
                "Keep your response concise but insightful."
            ),
        },
    }
    
    def prep(self, shared):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        stream_queue = shared.get("stream_queue")
        return (session["debate_context"], stream_queue)
    
    def exec(self, prep_res):
        debate_context, stream_queue = prep_res
        tracer = get_tracer()
        
        # Serialize trace context for thread propagation
        carrier = serialize_trace_context()
        
        # Container for results with thread-safe access
        results = {}
        results_lock = threading.Lock()
        
        def run_single_model(carrier, model_key, system_prompt, prompt):
            """Worker function executed in a thread pool."""
            # Re-attach OpenTelemetry trace context
            ctx = extract_trace_context(carrier)
            token = context.attach(ctx)
            
            try:
                with tracer.start_as_current_span(f"model_{model_key}") as span:
                    span.set_attribute("model.key", model_key)
                    
                    # Send phase activation sentinel for this model's lane
                    if stream_queue is not None:
                        from main import PHASE_SENTINEL_PREFIX, PHASE_SENTINEL_SUFFIX
                        stream_queue.put(
                            f"{PHASE_SENTINEL_PREFIX}{model_key}{PHASE_SENTINEL_SUFFIX}"
                        )
                    
                    # Call LLM with lane-aware streaming
                    result = call_llm_stream(
                        prompt,
                        system_prompt=system_prompt,
                        stream_queues=[stream_queue] if stream_queue else None,
                        lane_id=model_key,
                    )
                    
                    with results_lock:
                        results[model_key] = {"status": "success", "content": result}
                    
                    log.info(f"model_{model_key}_complete", response_chars=len(result))
                    return model_key, result
                    
            except Exception as exc:
                log.error(f"model_{model_key}_failed", error=type(exc).__name__, exc_info=exc)
                with results_lock:
                    results[model_key] = {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                return model_key, None
            finally:
                context.detach(token)
        
        # Submit all three models to the thread pool concurrently
        # Use a local executor with 3 workers (one per model)
        executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="parallel_model")
        futures = {}
        
        for model_key, config in self.MODEL_CONFIGS.items():
            prompt = config["prompt_builder"](debate_context)
            future = executor.submit(
                run_single_model,
                carrier,
                model_key,
                config["system_prompt"],
                prompt,
            )
            futures[future] = model_key
        
        # Wait for all futures to complete (success or failure)
        for future in as_completed(futures):
            model_key, result = future.result()
            log.info("parallel_model_finished", model=model_key, success=result is not None)
        
        executor.shutdown(wait=False)
        return results
    
    def post(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        
        # Save all successful model responses
        success_count = 0
        for model_key, result in exec_res.items():
            if result["status"] == "success":
                session["model_responses"][model_key] = result["content"]
                session["debate_transcript"].append({
                    "speaker": model_key,
                    "content": result["content"],
                })
                success_count += 1
            else:
                # Graceful degradation: leave missing model response as empty
                # DebateRound already handles missing keys with .get('model_x', 'No response')
                log.warning(
                    "model_response_missing",
                    model=model_key,
                    error=result.get("error"),
                )
        
        save_conversation(conversation_id, session)
        
        flow_log = shared["flow_queue"]
        flow_log.put(f"✅ Parallel models complete ({success_count}/3 succeeded)")
        
        return "default"
```

### 2.2 Important Implementation Notes

**Thread Pool Choice**: Use a **local** `ThreadPoolExecutor(max_workers=3)` inside `exec()`, not the global `chatflow_thread_pool`. This ensures:
- The flow thread pool doesn't get saturated by model workers
- Each parallel execution gets its own dedicated pool
- Clean shutdown per execution

**Trace Context**: The `run_single_model` function manually handles OTel context because it runs in a thread submitted to a *new* executor. The existing `@propagate_trace_context` decorator is designed for the flow runner thread pool, not internal node pools.

**Import Handling**: The import of `PHASE_SENTINEL_PREFIX` from `main` inside `run_single_model` avoids circular import issues (since `nodes.py` is already imported by `main.py` via `flow.py`). Alternatively, define the constants in a shared `utils/constants.py`.

---

## 3. Flow Graph Restructuring

### 3.1 Modified flow.py

Replace the sequential model chain with a single parallel node:

```python
from pocketflow import Flow

from nodes import (
    PrepareDebate,
    ParallelModelsNode,  # NEW
    DebateRound,
    Judge,
    SendFinalAnswer,
)


def create_flow():
    prepare_debate = PrepareDebate()
    parallel_models = ParallelModelsNode()  # Replaces Model A → B → C
    debate_round = DebateRound()
    judge = Judge()
    send_final_answer = SendFinalAnswer()

    prepare_debate >> parallel_models  # NEW: single parallel step
    parallel_models >> debate_round
    debate_round >> judge
    judge >> send_final_answer

    return Flow(start=prepare_debate)
```

### 3.2 Backward Compatibility

Keep the existing model node classes (`ModelAResponse`, `ModelBResponse`, `ModelCResponse`) in `nodes.py` but do **not** wire them into the flow. They serve as:
- Configuration holders (`SYSTEM_PROMPT` references)
- Fallback options if parallel execution needs to be disabled
- Historical reference for prompt templates

---

## 4. call_llm.py Modifications

### 4.1 Extended Signature

Add `lane_id` parameter to `call_llm_stream`:

```python
def call_llm_stream(
    message: str,
    system_prompt: str | None = None,
    stream_queues: list[Queue] | None = None,
    lane_id: str | None = None,  # NEW
) -> str:
```

### 4.2 Lane-Aware Token Wrapping

In the streaming loop, wrap tokens when `lane_id` is provided:

```python
# Near the top of call_llm_stream, define constants:
LANE_TOKEN_PREFIX = "\x00LANE:"
LANE_TOKEN_SUFFIX = "\x00"

# Inside the chunk processing loop:
for chunk in stream:
    text = parse_chunk(chunk)
    if text:
        accumulated.append(text)
        token_count += 1
        if stream_queues:
            for q in stream_queues:
                if lane_id:
                    q.put(f"{LANE_TOKEN_PREFIX}{lane_id}{LANE_TOKEN_SUFFIX}{text}")
                else:
                    q.put(text)
```

### 4.3 Logging Enhancement

Add lane info to the stream start log:

```python
log.info(
    "llm_stream_start",
    model=model,
    prompt_chars=prompt_len,
    lane_id=lane_id,  # NEW
)
```

---

## 5. main.py Modifications

### 5.1 SSE Handler Updates

Add lane token parsing alongside existing phase sentinel parsing:

```python
# Add constants near PHASE_SENTINEL definitions
LANE_TOKEN_PREFIX = "\x00LANE:"
LANE_TOKEN_SUFFIX = "\x00"

# Inside SSEHandler.do_GET() token processing loop:
try:
    token = q.get(timeout=60)
    if token is None:
        log.info("sse_done", sent_tokens=token_count)
        self.wfile.write(b"event: done\ndata: \n\n")
        self.wfile.flush()
        break
    
    token_count += 1
    
    # Handle phase sentinels (unchanged behavior)
    if isinstance(token, str) and token.startswith(PHASE_SENTINEL_PREFIX):
        phase = token[len(PHASE_SENTINEL_PREFIX):].rstrip(PHASE_SENTINEL_SUFFIX)
        payload = json.dumps({"phase": phase})
        self.wfile.write(f"event: phase\ndata: {payload}\n\n".encode())
        self.wfile.flush()
        continue
    
    # NEW: Handle lane-wrapped tokens
    if isinstance(token, str) and token.startswith(LANE_TOKEN_PREFIX):
        rest = token[len(LANE_TOKEN_PREFIX):]
        sep_idx = rest.find(LANE_TOKEN_SUFFIX)
        if sep_idx != -1:
            lane = rest[:sep_idx]
            actual_token = rest[sep_idx + len(LANE_TOKEN_SUFFIX):]
            payload = json.dumps({"token": actual_token, "lane": lane})
            self.wfile.write(f"data: {payload}\n\n".encode())
            self.wfile.flush()
            continue
    
    # Fallback: plain token (backward compatible)
    data = json.dumps({"token": token})
    self.wfile.write(f"data: {data}\n\n".encode())
    self.wfile.flush()
    
except Exception:
    break
```

### 5.2 Phase Map Updates

Update `PHASE_MAP` and `NEXT_PHASE` to reflect the new parallel structure:

```python
PHASE_MAP = {
    "🎯 Preparing debate context...": "prepare",
    # REMOVED: individual model completion messages
    "✅ Parallel models complete (3/3 succeeded)": "after_parallel",
    "✅ Parallel models complete (2/3 succeeded)": "after_parallel",
    "✅ Parallel models complete (1/3 succeeded)": "after_parallel",
    "🔄 Debate round completed - models critiqued each other": "after_round",
    "⚖️ Judge has synthesized the final answer": "after_judge",
}

NEXT_PHASE = {
    "prepare": "parallel",  # NEW: all three models start simultaneously
    "after_parallel": "round",
    "after_round": "judge",
    "after_judge": "done",
}
```

### 5.3 run_debate() Adjustments

The `run_debate()` function needs minimal changes since the flow log mechanism is unchanged. However, update the initial phase markers:

```python
# In run_debate(), replace:
# stream_queue.put(f"{PHASE_SENTINEL_PREFIX}model_a{PHASE_SENTINEL_SUFFIX}")
# With:
stream_queue.put(f"{PHASE_SENTINEL_PREFIX}reset{PHASE_SENTINEL_SUFFIX}")
# The parallel models will send their own phase markers when they start
```

### 5.4 UI JavaScript Updates

Inside `SSE_JS`, make these modifications:

1. **Add `appendTokenToLane`**:
```javascript
function appendTokenToLane(lane, token) {
  var id = LANE_IDS[lane];
  if (!id) return;
  var el = document.getElementById(id);
  if (!el) return;
  var s = el.querySelector('.lane-stream');
  if (!s) return;
  s.textContent += token;
  s.scrollTop = s.scrollHeight;
}
```

2. **Modify `startPhase`** (remove auto-deactivation):
```javascript
function startPhase(phase) {
  if (phase === 'reset') {
    clearAll();
    return;
  }
  if (_runComplete) {
    clearAll();
    _runComplete = false;
  }
  setLane(phase, 'active');
}
```

3. **Modify `onmessage`**:
```javascript
source.onmessage = function(e) {
  try {
    var d = JSON.parse(e.data);
    if (d.token != null) {
      if (d.lane) {
        appendTokenToLane(d.lane, d.token);
        setLane(d.lane, 'active');
      } else {
        appendToken(d.token);
      }
      setStatus('Live · streaming', 'live');
    }
  } catch(_) {}
};
```

4. **Add `end_phase` event listener**:
```javascript
source.addEventListener('end_phase', function(e) {
  try {
    var d = JSON.parse(e.data);
    if (d.phase) setLane(d.phase, 'done');
  } catch(_) {}
});
```

5. **Update `done` handler** to mark all remaining active phases as done:
```javascript
source.addEventListener('done', function() {
  ORDER.forEach(function(p) {
    setLane(p, 'done');
  });
  currentPhase = null;
  _runComplete = true;
  setStatus('Court adjourned', 'idle');
  source.close(); source = null;
  setTimeout(function() {
    setStatus('Awaiting query', 'idle');
    connect();
  }, 1800);
});
```

---

## 6. Thread Synchronization Strategy

### 6.1 Data Flow Diagram

```
[Flow Thread]
    │
    ▼
[ParallelModelsNode.prep()] ──→ loads debate_context
    │
    ▼
[ParallelModelsNode.exec()]
    │
    ├── serialize trace context (carrier)
    │
    ├── ThreadPoolExecutor(max_workers=3)
    │   ├── Thread-A: run_single_model(carrier, "model_a", ...)
    │   │   ├── attach OTel context
    │   │   ├── send PHASE:model_a sentinel
    │   │   ├── call_llm_stream(..., lane_id="model_a")
    │   │   │   └── puts \x00LANE:model_a\x00<token> into stream_queue
    │   │   └── save result to thread-safe `results` dict
    │   │
    │   ├── Thread-B: run_single_model(carrier, "model_b", ...)
    │   │   └── (same pattern)
    │   │
    │   └── Thread-C: run_single_model(carrier, "model_c", ...)
    │       └── (same pattern)
    │
    ├── as_completed(futures) waits for all three
    ├── executor.shutdown(wait=False)
    │
    └── returns results dict
    │
    ▼
[ParallelModelsNode.post()] ──→ saves results to session, logs completion
    │
    ▼
[DebateRound] ──→ reads model_responses from session
```

### 6.2 Synchronization Primitives

| Primitive | Purpose | Location |
|-----------|---------|----------|
| `threading.Lock()` | Protect `results` dict across model threads | `ParallelModelsNode.exec()` |
| `as_completed(futures)` | Barrier: wait for all models before proceeding | `ParallelModelsNode.exec()` |
| `Queue` (stream_queue) | Lock-free token passing to SSE handler | Shared across all threads |
| `context.attach()/detach()` | OTel trace context per thread | `run_single_model()` |

### 6.3 Race Condition Mitigations

1. **Stream Queue Ordering**: The `stream_queue` is a `Queue` which is thread-safe. Tokens from Models A, B, C will be interleaved in the queue in arrival order. The SSE handler processes them serially, and the `lane` metadata ensures correct UI routing regardless of interleaving.

2. **Session State Consistency**: Each model thread reads `debate_context` (immutable for the duration of parallel execution). Writes to `session["model_responses"]` happen in `post()`, which runs in the **flow thread** (single-threaded), so no lock is needed for session writes.

3. **Phase Sentinel Timing**: Phase sentinels are sent by each model thread BEFORE streaming begins. This ensures the UI lane is marked `active` before tokens arrive. There is a small race where tokens might arrive before the phase sentinel if the SSE handler reads faster than the model thread sends the sentinel, but the UI `appendTokenToLane` marks the lane active on first token anyway (defensive programming).

---

## 7. Error Handling

### 7.1 Partial Failure Scenarios

| Scenario | Behavior |
|----------|----------|
| Model A succeeds, B fails, C succeeds | Debate Round receives A and C, B shows `"No response"`. Flow continues. |
| All three models fail | Debate Round receives all `"No response"`. Judge may produce a generic answer. Log error. |
| Model B hangs (timeout) | `call_llm_stream` has `LLM_TIMEOUT=60s`. After timeout, exception is caught, result marked error. Other models proceed. |
| SSE disconnects mid-stream | Existing error handling in `SSEHandler` catches exception and breaks loop. Models continue in background. |
| ThreadPoolExecutor rejects task | Extremely unlikely with max_workers=3 and only 3 tasks. If it happens, exception propagates to flow level and is caught by `_flow_runner`. |

### 7.2 Retry Logic Preservation

The existing retry logic in `call_llm_stream` is **unchanged**. Each model independently retries up to `MAX_RETRIES` times with exponential backoff. A failure in Model B's retries does not affect Model A or C.

### 7.3 Observability on Failure

```python
# In run_single_model exception handler:
log.error(
    "model_failed",
    model=model_key,
    error=type(exc).__name__,
    exc_info=exc,
)
```

This ensures:
- The failing model is identified in logs
- `conversation_id` is present via existing structlog context binding
- The trace span records the exception

---

## 8. Testing Plan

### 8.1 Unit Tests

```python
# tests/test_parallel_models.py

def test_parallel_models_all_succeed():
    """Verify all three model responses are collected."""
    node = ParallelModelsNode()
    shared = create_mock_shared()
    # Mock call_llm_stream to return quickly
    results = node.exec(("debate context", None))
    assert len(results) == 3
    assert all(r["status"] == "success" for r in results.values())

def test_parallel_models_partial_failure():
    """Verify graceful degradation when one model fails."""
    # Mock call_llm_stream to raise for model_b only
    results = node.exec(("debate context", None))
    assert results["model_a"]["status"] == "success"
    assert results["model_b"]["status"] == "error"
    assert results["model_c"]["status"] == "success"

def test_lane_token_wrapping():
    """Verify tokens are wrapped with lane prefix."""
    q = Queue()
    call_llm_stream("test", stream_queues=[q], lane_id="model_a")
    # Verify queue contents have \x00LANE:model_a\x00 prefix
```

### 8.2 Integration Tests

1. **End-to-end streaming**: Submit a query, verify all three lanes show "On the floor" simultaneously within 2 seconds.
2. **Token routing**: Verify Model A tokens only appear in lane-a, Model B in lane-b, etc.
3. **Phase transitions**: After all models complete, verify round lane becomes active and model lanes show "Yielded".
4. **Failure injection**: Temporarily break Model B's API key, verify flow completes with A and C.

### 8.3 Performance Benchmarks

```python
# Measure wall-clock time before/after
# Target: < 30s for full pipeline (vs 51.8s baseline)
# Target: All three lanes active within 3s of submission
```

---

## 9. File-by-File Implementation Checklist

### nodes.py
- [ ] Import `ThreadPoolExecutor`, `as_completed`, `threading`
- [ ] Import OTel `context`
- [ ] Import `serialize_trace_context`, `extract_trace_context` from observability
- [ ] Define `ParallelModelsNode` class with `MODEL_CONFIGS`
- [ ] Implement `prep()` to load debate context and stream queue
- [ ] Implement `exec()` with local ThreadPoolExecutor
- [ ] Implement `post()` to save results and log completion
- [ ] Keep `ModelAResponse`, `ModelBResponse`, `ModelCResponse` classes for reference

### flow.py
- [ ] Import `ParallelModelsNode`
- [ ] Remove imports of individual model response nodes
- [ ] Replace `prepare_debate >> model_a >> model_b >> model_c >> debate_round`
  with `prepare_debate >> parallel_models >> debate_round`

### utils/call_llm.py
- [ ] Add `lane_id: str | None = None` parameter to `call_llm_stream`
- [ ] Define `LANE_TOKEN_PREFIX` and `LANE_TOKEN_SUFFIX` constants
- [ ] Wrap tokens with lane prefix when `lane_id` is provided
- [ ] Add `lane_id` to `llm_stream_start` log

### main.py
- [ ] Define `LANE_TOKEN_PREFIX` and `LANE_TOKEN_SUFFIX` constants
- [ ] Update SSE handler to parse lane-wrapped tokens
- [ ] Update `PHASE_MAP` to map parallel completion message
- [ ] Update `NEXT_PHASE`: `prepare` → `parallel`
- [ ] Update SSE_JS:
  - [ ] Add `appendTokenToLane()` function
  - [ ] Modify `startPhase()` to not auto-deactivate previous phases
  - [ ] Add `end_phase` event listener
  - [ ] Update `onmessage` to handle `d.lane`
  - [ ] Update `done` event handler to mark all lanes done
- [ ] Update `run_debate()` initial phase: send `reset` instead of `model_a`

---

## 10. Rollback Plan

If issues arise, revert to sequential execution by:

1. Restoring `flow.py` to wire `model_a >> model_b >> model_c`
2. Keeping `ParallelModelsNode` in `nodes.py` (unused)
3. The `lane_id` parameter in `call_llm_stream` is backward-compatible (defaults to `None`)
4. The SSE handler still supports plain tokens (fallback path)

No database migrations or external state changes required.

---

## 11. Common Pitfalls & Solutions

| Pitfall | Solution |
|---------|----------|
| Circular import `main.py` ↔ `nodes.py` | Import `PHASE_SENTINEL_PREFIX` inside `run_single_model` function body, not at module level. Or extract constants to `utils/constants.py`. |
| Model tokens interleaving unpredictably | The `lane` wrapping ensures correct routing regardless of interleaving. The Queue is FIFO per producer. |
| UI showing only one active lane | Ensure `startPhase` no longer calls `setLane(previous, 'done')`. Each lane stays active until explicitly ended. |
| Trace spans not connected | Verify `serialize_trace_context()` is called in the flow thread (where trace context exists) and `extract_trace_context()` + `context.attach()` in each worker thread. |
| Thread pool exhaustion | Use a local `ThreadPoolExecutor(max_workers=3)` in `ParallelModelsNode`, not the global `chatflow_thread_pool`. |
| Memory leak from thread pools | Call `executor.shutdown(wait=False)` after `as_completed` finishes. `wait=False` returns immediately; worker threads finish naturally. |
| SSE handler not seeing lane tokens | Verify the token wrapping format exactly matches the parsing regex: `\x00LANE:{lane_id}\x00{token}`. |

---

## 12. Verification Commands

After implementation, verify with:

```bash
# 1. Start server
python main.py

# 2. Check SSE endpoint responds
curl -N http://localhost:7861/stream

# 3. Submit a test query via Gradio UI
# 4. Observe:
#    - All three lanes show "On the floor" within 3 seconds
#    - Tokens stream into correct lanes simultaneously
#    - Round lane activates after all models complete
#    - Judge lane activates after round completes
#    - Final answer appears in chat

# 5. Check logs for parallel execution
rg "parallel_model_finished" debate.log
rg "model_.*_failed" debate.log

# 6. Check OpenTelemetry spans
# Look for three sibling spans: model_model_a, model_model_b, model_model_c
# under the flow_runner span
```

---

## Appendix A: Current Code References

### A.1 Existing Node Base Class
```python
class Node(BaseNode):
    def __init__(self, max_retries=1, wait=0):
        super().__init__()
        self.max_retries, self.wait = max_retries, wait
```

### A.2 Existing call_llm_stream Signature
```python
def call_llm_stream(
    message: str,
    system_prompt: str | None = None,
    stream_queues: list[Queue] | None = None,
) -> str:
```

### A.3 Existing SSE Phase Constants
```python
PHASE_SENTINEL_PREFIX = "\x00PHASE:"
PHASE_SENTINEL_SUFFIX = "\x00"
```

### A.4 Existing Trace Propagation Decorator
```python
def propagate_trace_context(func):
    @functools.wraps(func)
    def wrapper(carrier, *args, **kwargs):
        ctx = extract_trace_context(carrier)
        token = context.attach(ctx)
        try:
            return func(*args, **kwargs)
        finally:
            context.detach(token)
    return wrapper
```

---

## Appendix B: Expected Log Output (Success Case)

```json
{"event": "run_debate_start", "message_preview": "Explain quantum computing", "conversation_id": "uuid-123"}
{"event": "flow_run_start", "conversation_id": "uuid-123"}
{"event": "llm_stream_start", "model": "anthropic/claude-3.5-sonnet", "prompt_chars": 450, "lane_id": "model_a"}
{"event": "llm_stream_start", "model": "anthropic/claude-3.5-sonnet", "prompt_chars": 520, "lane_id": "model_b"}
{"event": "llm_stream_start", "model": "anthropic/claude-3.5-sonnet", "prompt_chars": 480, "lane_id": "model_c"}
{"event": "llm_stream_done", "model": "anthropic/claude-3.5-sonnet", "latency_ms": 2400, "tokens": 180, "lane_id": "model_a"}
{"event": "model_model_a_complete", "response_chars": 450}
{"event": "llm_stream_done", "model": "anthropic/claude-3.5-sonnet", "latency_ms": 19200, "tokens": 950, "lane_id": "model_b"}
{"event": "model_model_b_complete", "response_chars": 2100}
{"event": "llm_stream_done", "model": "anthropic/claude-3.5-sonnet", "latency_ms": 4600, "tokens": 320, "lane_id": "model_c"}
{"event": "model_model_c_complete", "response_chars": 680}
{"event": "parallel_model_finished", "model": "model_a", "success": true}
{"event": "parallel_model_finished", "model": "model_c", "success": true}
{"event": "parallel_model_finished", "model": "model_b", "success": true}
{"event": "llm_stream_start", "model": "anthropic/claude-3.5-sonnet", "prompt_chars": 3800}
{"event": "llm_stream_done", "model": "anthropic/claude-3.5-sonnet", "latency_ms": 15900, "tokens": 780}
{"event": "llm_stream_start", "model": "anthropic/claude-3.5-sonnet", "prompt_chars": 5200}
{"event": "llm_stream_done", "model": "anthropic/claude-3.5-sonnet", "latency_ms": 8800, "tokens": 420}
{"event": "flow_run_done", "conversation_id": "uuid-123"}
{"event": "run_debate_done", "conversation_id": "uuid-123"}
```

---

*This prompt is self-contained and assumes only the context provided in the Current Architecture section. All file paths are relative to the project root (`test/chat/`).*
