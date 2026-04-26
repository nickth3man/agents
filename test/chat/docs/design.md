# Design Doc: Multi-Model Debate Chat

> Please DON'T remove notes for AI

## Requirements

> Notes for AI: Keep it simple and clear.
> If the requirements are abstract, write concrete user stories

- **US-1**: As a user, I want to ask any question in a chat interface and receive one clear final answer.
- **US-2**: As a user, I want multiple AI models to debate possible answers before the system responds, so the final answer is more robust.
- **US-3**: As a user, I want the system to use only LLM calls and no external APIs or tools, so all reasoning happens through model-to-model conversation.
- **US-4**: As a user, I want a judge model to review the debate and produce the final answer, so the response is concise, balanced, and directly addresses my query.
- **US-5**: As a user, I want to see the high-level debate progress in the flow visualization, without exposing unnecessary private reasoning.
- **US-6**: As a user, I want to see live token streaming in a side panel as the debate progresses.

## Flow Design

> Notes for AI:
> 1. Consider the design patterns of agent, map-reduce, rag, and workflow. Apply them if they fit.
> 2. Present a concise, high-level description of the workflow.

### Applicable Design Pattern:

**Parallel Multi-Agent Debate with Reduce**: The system runs three LLM-backed model nodes **concurrently** via `asyncio.gather`, each producing a candidate answer with a distinct persona. A debate round node then evaluates all three responses on four structured dimensions. A final judge node extracts the strongest claims, resolves contradictions using the debate critique, and synthesises one calibrated user-facing answer.

**Map step**: `ParallelModels` maps the same debate context to three independent model calls concurrently.

**Reduce step**: The `Judge` node reduces the three responses + debate critique into one final answer using extractive-then-synthesise mode.

### Flow high-level Design:

1. **Prepare Debate Node**: Reads the user query and chat history, then initialises the debate state.
2. **Parallel Models Node**: Runs Model A, B, and C concurrently via `asyncio.gather`. Each streams tokens in real time to its own UI lane.
3. **Debate Round Node**: Evaluates all three model responses in a single LLM call using four structured dimensions (Accuracy, Completeness, Actionability, Intellectual Honesty). Closes with "Key Takeaways for Judge".
4. **Judge Node**: Reads model responses and debate critique, extracts the strongest claim from each model, resolves contradictions, and writes one calibrated final answer.
5. **Send Final Answer Node**: Sends the judge's final answer to the Gradio UI and ends the flow turn.

```mermaid
flowchart TD
    PrepareDebate[Prepare Debate Node] --> ParallelModels

    subgraph ParallelModels[Parallel Models Node — asyncio.gather]
        ModelA[Model A\nPragmatic Senior Engineer]
        ModelB[Model B\nCautious Technical Lead]
        ModelC[Model C\nUnconventional Systems Thinker]
    end

    ParallelModels --> DebateRound[Debate Round Node]
    DebateRound --> Judge[Judge Node]
    Judge --> SendFinalAnswer[Send Final Answer Node]
```

## Utility Functions

> Notes for AI:
> 1. Understand the utility function definition thoroughly by reviewing the doc.
> 2. Include only the necessary utility functions, based on nodes in the flow.

1. **Call LLM** (`utils/call_llm.py`)
   - Uses the OpenAI SDK pointed at OpenRouter (`https://openrouter.ai/api/v1`).
   - Reads `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, and `APP_NAME` from environment variables.
   - `call_llm_stream(message, system_prompt, token_queue)` → `str`: streams tokens into `token_queue` in real time via `\x00PHASE:name\x00` sentinels, returns the full accumulated response.
   - Retries on `APIConnectionError`, `APIStatusError` (429/500/502/503), `APITimeoutError`, and base `APIError` (e.g. mid-stream disconnects) with exponential backoff.
   - Emits an OpenTelemetry span per call with `llm.prompt_chars`, `llm.response_chars`, `llm.latency_ms`.
   - Used by all debate model nodes, the debate round node, and the judge node.

2. **Async Call LLM** (`utils/call_llm_async.py`)
   - Async wrapper around the same OpenRouter endpoint.
   - `call_llm_async(message, system_prompt)` → `str`: non-streaming, awaitable. Used by test scripts.

3. **Conversation Management** (`utils/conversation.py`)
   - In-memory cache (`conversation_cache` dict) keyed by `conversation_id` with 30-minute TTL.
   - `load_conversation(conversation_id)` → dict: returns the session dict or `{}` if not found.
   - `save_conversation(conversation_id, session)`: writes session back to the cache.
   - Used to persist debate state across nodes within the same request.

4. **Format Chat History** (`utils/format_chat_history.py`)
   - *Input*: history (list of dicts with `role` and `content`)
   - *Output*: formatted string (`"role: content"` lines joined by newline, or `"No history"`)
   - Filters out assistant messages that start with flow-visualisation emoji prefixes (`🤔`, `➡️`, `⬅️`).
   - Used by `PrepareDebate` to build the debate context passed to all downstream nodes.

5. **Observability** (`utils/observability.py`)
   - Configures `structlog` to write structured JSON logs to `debate.log`.
   - Sets up an OpenTelemetry `TracerProvider` with `ConsoleSpanExporter` (replace with OTLP for production).
   - Exports `logger` and `get_tracer()` for use by nodes.

## Prompts

All system and user prompts are stored as plain-text files in `prompts/` and loaded at module startup by `nodes.py` using `_prompt(name)`.

| File | Role | Key instructions |
|---|---|---|
| `model_a_system.txt` | Pragmatic Senior Engineer | Lead with recommendation; bias for proven tech; defend positions |
| `model_a_user.txt` | Model A task | 1-sentence recommendation, top 2-3 reasons, one wrong-when, 24-hour action; ≤250 words |
| `model_b_system.txt` | Cautious Technical Lead | Surface hidden assumptions; end with clear recommendation + caveat |
| `model_b_user.txt` | Model B task | 3 hidden assumptions, 2 overlooked risks, 1 missing piece, recommendation + caveat; ≤350 words |
| `model_c_system.txt` | Unconventional Systems Thinker | Question framing; ground alternatives in real conditions |
| `model_c_user.txt` | Model C task | Reframe, non-obvious alternative, assumption critique, recommendation; ≤300 words |
| `debate_round_system.txt` | Debate Facilitator | Evaluate on Accuracy/Completeness/Actionability/Intellectual Honesty; close with Key Takeaways |
| `judge_system.txt` | Judge synthesiser | Extractive-then-synthesise; anti-sycophancy; length calibration by question complexity |

## Node Design

### Shared Store

> Notes for AI: Try to minimise data redundancy

```python
shared = {
    "conversation_id": str,   # Unique UUID for the conversation session
    "history": list,          # Prior messages: [{"role": "user"/"assistant", "content": str}]
    "query": str,             # The current user input
    "queue": Queue,           # Queue for the final answer (read by Gradio)
    "flow_queue": Queue,      # Queue for flow-progress log messages (None = flow done)
    "stream_queue": Queue,    # Queue for live token streaming to the SSE server / Gradio panel
}
```

**State Management Note**: Debate-specific state is stored in the conversation session keyed by `conversation_id`, not duplicated in `shared`. The session accumulates keys as the flow progresses.

Session state after a full flow run:

```python
session = {
    "debate_context": str,         # Formatted chat history + current query + date
    "model_responses": {
        "model_a": str,            # Pragmatic Senior Engineer response (or "Error: ...")
        "model_b": str,            # Cautious Technical Lead response (or "Error: ...")
        "model_c": str,            # Unconventional Systems Thinker response (or "Error: ...")
    },
    "debate_transcript": [
        {"speaker": "model_a", "content": str},
        {"speaker": "model_b", "content": str},
        {"speaker": "model_c", "content": str},
        {"speaker": "debate_round", "content": str},
    ],
    "debate_critique": str,        # Full critique output from the Debate Round node
    "judge_answer": str,           # Final synthesised answer from the Judge node
    "action_result": str,          # Copy of judge_answer saved after SendFinalAnswer
}
```

### Node Steps

> Notes for AI: All nodes are `AsyncNode` from PocketFlow (async lifecycle: `prep_async`, `exec_async`, `post_async`).

1. **Prepare Debate Node** (`PrepareDebate`)
   - *Purpose*: Initialise the debate for the current user query.
   - *Type*: `AsyncNode`
   - *Steps*:
     - *prep_async*: Read `conversation_id`, `history`, and `query` from `shared`. Load existing session.
     - *exec_async*: Build `debate_context` string: formatted chat history + current user query + current date.
     - *post_async*: Save `debate_context`, empty `model_responses` dict, empty `debate_transcript` list, and `judge_answer = None` to the session. Put `"🎯 Preparing debate context..."` into `flow_queue`. Return `"default"`.

2. **Parallel Models Node** (`ParallelModels`)
   - *Purpose*: Run Model A, B, and C concurrently, streaming tokens to their respective UI lanes.
   - *Type*: `AsyncNode`
   - *Steps*:
     - *prep_async*: Load `debate_context` from session. Read `stream_queue` from `shared`.
     - *exec_async*: Call `asyncio.gather` with `return_exceptions=True` across three `_run_model` coroutines. Each emits a `\x00PHASE:model_x\x00` sentinel before streaming, then calls `call_llm_stream`. Exceptions are caught and stored as `"Error: ..."` strings.
     - *post_async*: Save all three responses to `model_responses` and append to `debate_transcript`. Put three `"✅ Model X has responded"` messages into `flow_queue`. Return `"default"`.

3. **Debate Round Node** (`DebateRound`)
   - *Purpose*: Produce a structured critique of all three model responses in a single LLM call.
   - *Type*: `AsyncNode`
   - *Steps*:
     - *prep_async*: Load `debate_context` and `model_responses` from session. Read `stream_queue`.
     - *exec_async*: Substitute `"Error: ..."` responses with `"[This model failed to respond — skip in critique]"`. Call `call_llm_stream` with all three responses and the `debate_round_system.txt` system prompt.
     - *post_async*: Append to `debate_transcript`. Save critique as `debate_critique` in session. Put `"🔄 Debate round completed"` into `flow_queue`. Return `"default"`.

4. **Judge Node** (`Judge`)
   - *Purpose*: Synthesise model responses and debate critique into one final user-facing answer.
   - *Type*: `AsyncNode`
   - *Steps*:
     - *prep_async*: Load `debate_context`, `model_responses`, and `debate_critique` from session. Read `stream_queue`.
     - *exec_async*: Substitute `"Error: ..."` responses with `"[This model failed to respond]"`. Call `call_llm_stream` with the `judge_system.txt` system prompt. Length calibration in the system prompt caps output at 200/500/700 words depending on question complexity.
     - *post_async*: Save result as `judge_answer` in session. Put `"⚖️ Judge has synthesised the final answer"` into `flow_queue`. Return `"default"`.

5. **Send Final Answer Node** (`SendFinalAnswer`)
   - *Purpose*: Deliver the judge's answer to the UI and signal end-of-flow.
   - *Type*: `AsyncNode`
   - *Steps*:
     - *prep_async*: Put `None` into `flow_queue` (signals flow done to Gradio). If `stream_queue` is present, put `None` into it. Load `judge_answer` from session and read `queue` from `shared`.
     - *exec_async*: Put `judge_answer` then `None` into `queue` (the Gradio chat queue).
     - *post_async*: Save `judge_answer` as `action_result` in session. Return `"done"`.

## UI & Streaming (`main.py` + `frontend/`)

The UI is built with **Gradio** (`gr.Blocks`) and served at the default port. A separate **SSE server** runs on port `7861` to push live token streaming to the browser. Static assets (CSS, JS, HTML partials) are extracted to `frontend/` and loaded at startup.

### Frontend files

| File | Purpose |
|---|---|
| `frontend/styles.css` | All UI styles: layout, debate lanes, typography, dark/light theme |
| `frontend/sse.js` | SSE `EventSource` client; phase state machine that routes tokens to the correct lane |
| `frontend/masthead.html` | Newspaper-style masthead; `{ISSUE_DATE}` placeholder replaced at startup |
| `frontend/floor.html` | Five debate lane `<div>`s (lane-a, lane-b, lane-c, lane-round, lane-judge) |
| `frontend/chat_header.html` | "Chamber — Question & Verdict" section header |

### SSE Server (`SSEHandler`)
- Listens on `GET /stream`.
- Reads tokens from the global `_active_stream_queue`.
- Phase sentinels (`\x00PHASE:name\x00`) cause the JS client to switch its active lane.
- Sends each non-sentinel token as `data: {"token": "..."}` SSE events; sends `event: done` on `None`.
- Started as a daemon thread at module load time.

### Gradio Layout
- **Left column (scale=2)**: `gr.Chatbot` + text input + Send button + Clear button.
- **Right column (scale=1)**: "Live Stream" panel — HTML injected via `head=SSE_JS` connects to the SSE server and routes tokens to the correct lane in real time.

### Flow Callbacks
- `add_user_message(message, history)`: appends the user turn and clears the input box (runs without queue).
- `run_debate(history, uuid_state)`: generator function.
  1. Extracts `query` and `conversation_id` (from `uuid_state`).
  2. Creates `chat_queue`, `flow_queue`, and `stream_queue`.
  3. Sets `_active_stream_queue = stream_queue` (picked up by the SSE server).
  4. Builds `shared` dict and submits `chat_flow.run(shared)` to a `ThreadPoolExecutor` (max 5 workers).
  5. Yields placeholder `"... *Debating* ..."` immediately, then polls `flow_queue` at 50 ms intervals.
  6. Once `flow_queue` signals done (`None`), reads the final answer from `chat_queue` and appends it as `"### Final Answer\n\n{final}"`.
- Clear button resets the chatbot and generates a new `uuid.uuid4()` conversation ID.


## New Feature: Argument Cartography — Visual Claim Graph

### Overview

Transform the linear debate transcript into an interactive **argument map** (claim graph). Each claim becomes a node; supports, refutations, and contradictions become directed edges. The user can click any claim to see who said it, explore supporting evidence, and navigate the debate spatially instead of reading a wall of text.

### Why

Research on deliberation and argumentation shows that **visual argument maps improve comprehension** and reduce cognitive load significantly compared to linear text transcripts. In a multi-agent debate where three models build on (or tear down) each other’s positions, a graph makes the *structure of reasoning* visible: what supports what, where the fault lines are, and how the models converge or diverge.

### User Story

- **US-7**: As a user, I want to see the debate rendered as an interactive claim graph so I can explore who said what, how claims relate, and where the key disagreements lie.

### Architecture

```mermaid
flowchart TD
    PrepareDebate[Prepare Debate Node] --> ParallelModels

    subgraph ParallelModels[Parallel Models Node — asyncio.gather]
        ModelA[Model A\nPragmatic Senior Engineer]
        ModelB[Model B\nCautious Technical Lead]
        ModelC[Model C\nUnconventional Systems Thinker]
    end

    ParallelModels --> DebateRound[Debate Round Node]
    DebateRound --> Judge[Judge Node]
    Judge --> ArgumentMapper[Argument Mapper Node]
    ArgumentMapper --> SendFinalAnswer[Send Final Answer Node]
```

### New Components

#### 1. Argument Mapper Node (`ArgumentMapper`)

A new `AsyncNode` that runs **after** the Judge and **before** `SendFinalAnswer`.

**Purpose**: Extract structured claims and relationships from the full debate transcript and produce a machine-readable argument graph.

**Steps**:
- *prep_async*: Load `debate_transcript` and `judge_answer` from session.
- *exec_async*: Call `call_llm_stream` with a structured extraction prompt (`argument_mapper_system.txt`). The LLM returns a JSON object conforming to the `ArgumentGraph` Pydantic schema.
- *post_async*: Save the graph to session as `argument_graph`. Put `"🗺️ Argument map generated"` into `flow_queue`. Return `"default"`.

**ArgumentGraph Schema** (Pydantic):

```python
class Claim(BaseModel):
    id: str
    text: str
    speaker: Literal["model_a", "model_b", "model_c", "judge"]
    round: int
    confidence: float  # 0.0–1.0, extracted or estimated

class Edge(BaseModel):
    from_id: str
    to_id: str
    relation: Literal["supports", "refutes", "qualifies", "asks"]
    evidence: str  # brief quote or reasoning

class ArgumentGraph(BaseModel):
    nodes: list[Claim]
    edges: list[Edge]
```

#### 2. Prompt File (`prompts/argument_mapper_system.txt`)

System prompt for the extraction LLM:

```
You are an Argument Cartographer. Your job is to read a debate transcript and produce a structured claim graph.

Rules:
1. Extract every substantive claim made by each speaker.
2. Identify relationships: supports, refutes, qualifies, or asks (Socratic).
3. Include the judge's synthesis claims.
4. Assign confidence scores based on how strongly the speaker defended the claim.
5. Keep claim text under 20 words.
6. Return ONLY valid JSON matching the ArgumentGraph schema.
```

#### 3. Frontend — Argument Graph Panel (`frontend/argument_graph.html` + `frontend/argument_graph.js`)

A new right-hand panel (or modal overlay) that renders the graph using **Cytoscape.js** (lightweight, force-directed layout).

**Features**:
- **Force-directed layout**: Claims naturally cluster by speaker and round.
- **Color coding**: Model A = blue, Model B = amber, Model C = purple, Judge = green.
- **Edge styling**: Supports = solid green, Refutes = dashed red, Qualifies = dotted orange, Asks = dotted gray.
- **Click-to-expand**: Clicking a claim opens a detail card showing the full text, speaker, round, and incoming/outgoing edges.
- **Filter by speaker**: Toggle models on/off to see only certain voices.
- **Zoom & pan**: Standard graph navigation.

**HTML Structure** (`frontend/argument_graph.html`):

```html
<div id="argument-graph-panel" class="graph-panel">
  <div class="graph-header">
    <h3>Argument Cartography</h3>
    <div class="graph-legend">
      <span class="legend-item"><span class="dot blue"></span> Model A</span>
      <span class="legend-item"><span class="dot amber"></span> Model B</span>
      <span class="legend-item"><span class="dot purple"></span> Model C</span>
      <span class="legend-item"><span class="dot green"></span> Judge</span>
    </div>
  </div>
  <div id="cy" class="graph-canvas"></div>
  <div id="claim-detail" class="claim-detail hidden">
    <h4 id="detail-title"></h4>
    <p id="detail-text"></p>
    <div id="detail-meta"></div>
  </div>
</div>
```

**JavaScript** (`frontend/argument_graph.js`):

```javascript
function renderArgumentGraph(graphData) {
  const cy = cytoscape({
    container: document.getElementById('cy'),
    elements: [
      ...graphData.nodes.map(n => ({
        data: { id: n.id, label: n.text, speaker: n.speaker, confidence: n.confidence }
      })),
      ...graphData.edges.map(e => ({
        data: { source: e.from_id, target: e.to_id, relation: e.relation }
      }))
    ],
    style: [
      { selector: 'node', style: { 'label': 'data(label)', 'width': 40, 'height': 40 } },
      { selector: 'node[speaker="model_a"]', style: { 'background-color': '#3b82f6' } },
      { selector: 'node[speaker="model_b"]', style: { 'background-color': '#f59e0b' } },
      { selector: 'node[speaker="model_c"]', style: { 'background-color': '#8b5cf6' } },
      { selector: 'node[speaker="judge"]', style: { 'background-color': '#10b981' } },
      { selector: 'edge[relation="supports"]', style: { 'line-color': '#22c55e', 'target-arrow-color': '#22c55e' } },
      { selector: 'edge[relation="refutes"]', style: { 'line-color': '#ef4444', 'line-style': 'dashed', 'target-arrow-color': '#ef4444' } },
    ],
    layout: { name: 'cose', padding: 10 }
  });

  cy.on('tap', 'node', function(evt){
    const node = evt.target;
    showClaimDetail(node.data());
  });
}
```

#### 4. SSE Protocol Extension

The argument graph is delivered via SSE as a single JSON payload after the judge phase:

```python
# In ArgumentMapper.post_async or main.py
graph_json = json.dumps({"argument_graph": session["argument_graph"]})
stream_queue.put(f"\x00GRAPH:{graph_json}\x00")
```

The SSE handler emits:

```
event: argument_graph
data: {"nodes": [...], "edges": [...]}
```

The frontend listens for `argument_graph` events and calls `renderArgumentGraph()`.

### Session State Update

Session state after a full flow run now includes:

```python
session = {
    # ... existing fields ...
    "argument_graph": {
        "nodes": [...],
        "edges": [...]
    },
}
```

### File Changes Summary

| File | Change |
|---|---|
| `nodes.py` | Add `ArgumentMapper` class |
| `flow.py` | Wire `Judge >> ArgumentMapper >> SendFinalAnswer` |
| `prompts/argument_mapper_system.txt` | New extraction prompt |
| `frontend/argument_graph.html` | New graph panel markup |
| `frontend/argument_graph.js` | Cytoscape.js rendering logic |
| `frontend/styles.css` | Graph panel styles |
| `main.py` | SSE handler: parse `\x00GRAPH:...\x00` sentinel, emit `argument_graph` event |

### Rollback

If the graph generation fails (malformed JSON, schema mismatch), the node catches the exception and passes an empty graph. The frontend detects empty graphs and hides the panel, falling back to the existing linear transcript view. The flow continues to `SendFinalAnswer` regardless.