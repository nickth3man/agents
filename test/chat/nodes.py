"""Async nodes for parallel multi-model debate."""

import asyncio
import os
from datetime import datetime

from pocketflow import AsyncNode

from utils.call_llm import call_llm_stream
from utils.conversation import load_conversation, save_conversation
from utils.format_chat_history import format_chat_history
from utils.observability import get_tracer, logger as log

_PROMPTS = os.path.join(os.path.dirname(__file__), "prompts")


def _prompt(name: str) -> str:
    with open(os.path.join(_PROMPTS, name), encoding="utf-8") as f:
        return f.read()


class PrepareDebate(AsyncNode):
    async def prep_async(self, shared):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        return session, shared["history"], shared["query"]

    async def exec_async(self, prep_res):
        session, history, query = prep_res
        debate_context = f"""### CHAT HISTORY
{format_chat_history(history)}

### CURRENT USER QUESTION
user: {query}

Current Date: {datetime.now().date()}
"""
        return debate_context

    async def post_async(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        
        session["debate_context"] = exec_res
        session["model_responses"] = {}
        session["debate_transcript"] = []
        session["judge_answer"] = None
        
        save_conversation(conversation_id, session)
        
        flow_log = shared["flow_queue"]
        flow_log.put("🎯 Preparing debate context...")
        
        return "default"


# ── Model prompts (used by ParallelModels) ─────────────────────────────────

MODEL_A_PROMPT = _prompt("model_a_user.txt")
MODEL_B_PROMPT = _prompt("model_b_user.txt")
MODEL_C_PROMPT = _prompt("model_c_user.txt")

MODEL_A_SYSTEM = _prompt("model_a_system.txt")
MODEL_B_SYSTEM = _prompt("model_b_system.txt")
MODEL_C_SYSTEM = _prompt("model_c_system.txt")


class ParallelModels(AsyncNode):
    """Runs Model A, B, C in parallel using asyncio.gather()."""
    
    async def prep_async(self, shared):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        stream_queue = shared.get("stream_queue")
        return (session["debate_context"], stream_queue)

    async def exec_async(self, prep_res):
        context, stream_queue = prep_res
        import time
        tracer = get_tracer()
        
        with tracer.start_as_current_span("parallel_models"):
            log.info("parallel_models_start")
            t0 = time.perf_counter()
            
            # Run all three models concurrently
            results = await asyncio.gather(
                self._run_model("model_a", context, MODEL_A_PROMPT, MODEL_A_SYSTEM, stream_queue),
                self._run_model("model_b", context, MODEL_B_PROMPT, MODEL_B_SYSTEM, stream_queue),
                self._run_model("model_c", context, MODEL_C_PROMPT, MODEL_C_SYSTEM, stream_queue),
                return_exceptions=True,
            )
            
            elapsed = time.perf_counter() - t0
            log.info("parallel_models_done", total_elapsed_ms=round(elapsed * 1000, 2))
            
            # Check for failures
            responses = {}
            for name, result in zip(["model_a", "model_b", "model_c"], results):
                if isinstance(result, Exception):
                    log.error("parallel_models_failed", model=name, error=str(result))
                    responses[name] = f"Error: {result}"
                else:
                    responses[name] = result
            
            return responses

    async def _run_model(self, phase_name, context, prompt_template, system_prompt, stream_queue):
        """Run a single model and return its response."""
        import time
        tracer = get_tracer()
        t0 = time.perf_counter()
        log.info(f"model_start", phase=phase_name)
        with tracer.start_as_current_span(f"model_{phase_name}"):
            # Send phase marker to UI
            if stream_queue is not None:
                from main import PHASE_SENTINEL_PREFIX, PHASE_SENTINEL_SUFFIX
                stream_queue.put(f"{PHASE_SENTINEL_PREFIX}{phase_name}{PHASE_SENTINEL_SUFFIX}")
            
            prompt = prompt_template.format(context=context)
            response = await call_llm_stream(
                prompt, system_prompt=system_prompt, token_queue=stream_queue
            )
            elapsed = time.perf_counter() - t0
            log.info(f"model_end", phase=phase_name, elapsed_ms=round(elapsed * 1000, 2))
            
            return response

    async def post_async(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        
        # Store all responses, filtering out errors
        for model_name, response in exec_res.items():
            session["model_responses"][model_name] = response
            session["debate_transcript"].append({
                "speaker": model_name,
                "content": response
            })
        
        save_conversation(conversation_id, session)
        
        flow_log = shared["flow_queue"]
        flow_log.put("✅ Model A (Direct & Practical) has responded")
        flow_log.put("✅ Model B (Cautious & Analytical) has responded")
        flow_log.put("✅ Model C (Creative & Alternative) has responded")
        
        return "default"


class DebateRound(AsyncNode):
    SYSTEM_PROMPT = _prompt("debate_round_system.txt")

    async def prep_async(self, shared):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        stream_queue = shared.get("stream_queue")
        return (
            session["debate_context"],
            session["model_responses"],
            stream_queue,
        )

    async def exec_async(self, prep_res):
        debate_context, model_responses, stream_queue = prep_res
        
        # Filter out error responses so they don't pollute the debate
        clean_responses = {}
        for name, resp in model_responses.items():
            if isinstance(resp, str) and resp.startswith("Error:"):
                clean_responses[name] = "[This model failed to respond — skip in critique]"
            else:
                clean_responses[name] = resp
        
        prompt = f"""{debate_context}

### INITIAL RESPONSES FROM THREE MODELS

**Model A (Direct & Practical):**
{clean_responses.get('model_a', 'No response')}

**Model B (Cautious & Analytical):**
{clean_responses.get('model_b', 'No response')}

**Model C (Creative & Alternative):**
{clean_responses.get('model_c', 'No response')}

### DEBATE ROUND TASK
Produce a concise debate critique. For each model, identify:
1. What are the strongest points in their answer?
2. What are the weaknesses or gaps?
3. What would improve their response?

Format as a structured critique. Be specific and constructive."""
        
        if stream_queue is not None:
            return await call_llm_stream(
                prompt, system_prompt=self.SYSTEM_PROMPT, token_queue=stream_queue
            )
        return await call_llm_stream(prompt, system_prompt=self.SYSTEM_PROMPT)

    async def post_async(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        
        session["debate_transcript"].append({
            "speaker": "debate_round",
            "content": exec_res
        })
        session["debate_critique"] = exec_res
        
        save_conversation(conversation_id, session)
        
        flow_log = shared["flow_queue"]
        flow_log.put("🔄 Debate round completed - models critiqued each other")
        
        return "default"


class Judge(AsyncNode):
    SYSTEM_PROMPT = _prompt("judge_system.txt")

    async def prep_async(self, shared):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        stream_queue = shared.get("stream_queue")
        return (
            session["debate_context"],
            session["model_responses"],
            session.get("debate_critique", ""),
            stream_queue,
        )

    async def exec_async(self, prep_res):
        debate_context, model_responses, debate_critique, stream_queue = prep_res
        
        # Filter out error responses for judge synthesis
        clean_responses = {}
        for name, resp in model_responses.items():
            if isinstance(resp, str) and resp.startswith("Error:"):
                clean_responses[name] = "[This model failed to respond]"
            else:
                clean_responses[name] = resp
        
        prompt = f"""{debate_context}

### MODEL RESPONSES

**Model A (Direct & Practical):**
{clean_responses.get('model_a', 'No response')}

**Model B (Cautious & Analytical):**
{clean_responses.get('model_b', 'No response')}

**Model C (Creative & Alternative):**
{clean_responses.get('model_c', 'No response')}

### DEBATE CRITIQUE
{debate_critique}

### YOUR TASK AS JUDGE
Synthesize the above into ONE final answer for the user. You must:
1. Resolve any conflicts or disagreements between models
2. Combine the strongest points from each response
3. Remove weak or unsupported claims
4. Ensure the final answer is clear, accurate, and directly addresses the user's question
5. Maintain an enthusiastic and helpful tone

Return ONLY the final answer. Do not explain your judging process."""
        
        if stream_queue is not None:
            return await call_llm_stream(
                prompt, system_prompt=self.SYSTEM_PROMPT, token_queue=stream_queue
            )
        return await call_llm_stream(prompt, system_prompt=self.SYSTEM_PROMPT)

    async def post_async(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        
        session["judge_answer"] = exec_res
        save_conversation(conversation_id, session)
        
        flow_log = shared["flow_queue"]
        flow_log.put("⚖️ Judge has synthesized the final answer")
        
        return "default"


class SendFinalAnswer(AsyncNode):
    async def prep_async(self, shared):
        flow_log = shared["flow_queue"]
        flow_log.put(None)
        
        # Signal end of stream if streaming was used
        stream_queue = shared.get("stream_queue")
        if stream_queue is not None:
            stream_queue.put(None)
        
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        return session["judge_answer"], shared["queue"]

    async def exec_async(self, prep_res):
        answer, queue = prep_res
        queue.put(answer)
        queue.put(None)
        return answer

    async def post_async(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        
        session["action_result"] = exec_res
        save_conversation(conversation_id, session)
        
        return "done"
