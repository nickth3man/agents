"""Async nodes for parallel multi-model debate."""

import asyncio
from datetime import datetime
from queue import Queue

from pocketflow import AsyncNode

from utils.call_llm_async import call_llm_stream_async
from utils.conversation import load_conversation, save_conversation
from utils.format_chat_history import format_chat_history
from utils.observability import get_tracer, logger as log


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


class ModelAResponse(AsyncNode):
    SYSTEM_PROMPT = """You are Model A, a direct and practical AI assistant. 
Your style is concise, action-oriented, and focused on immediate usefulness.
Provide clear, straightforward answers that the user can act on right away."""

    async def prep_async(self, shared):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        stream_queue = shared.get("stream_queue")
        return (session["debate_context"], stream_queue)

    async def exec_async(self, prep_res):
        context, stream_queue = prep_res
        prompt = f"""{context}

### YOUR TASK
Provide your best answer to the user's question. Be direct, practical, and actionable.
Keep your response concise but complete."""
        
        if stream_queue is not None:
            return await call_llm_stream_async(
                prompt, system_prompt=self.SYSTEM_PROMPT, token_queue=stream_queue
            )
        return await call_llm_stream_async(prompt, system_prompt=self.SYSTEM_PROMPT)

    async def post_async(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        
        session["model_responses"]["model_a"] = exec_res
        session["debate_transcript"].append({
            "speaker": "model_a",
            "content": exec_res
        })
        
        save_conversation(conversation_id, session)
        
        flow_log = shared["flow_queue"]
        flow_log.put("✅ Model A (Direct & Practical) has responded")
        
        return "default"


class ModelBResponse(AsyncNode):
    SYSTEM_PROMPT = """You are Model B, a cautious and analytical AI assistant.
Your style is thorough, risk-aware, and focused on identifying edge cases, assumptions, and potential problems.
Always consider what could go wrong and what the user might be missing."""

    async def prep_async(self, shared):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        stream_queue = shared.get("stream_queue")
        return (session["debate_context"], stream_queue)

    async def exec_async(self, prep_res):
        context, stream_queue = prep_res
        prompt = f"""{context}

### YOUR TASK
Provide your best answer to the user's question, but be sure to:
1. Identify key assumptions you're making
2. Point out edge cases or exceptions the user should consider
3. Highlight any risks or caveats
4. Suggest what information might be missing for a complete answer

Keep your response concise but thorough."""
        
        if stream_queue is not None:
            return await call_llm_stream_async(
                prompt, system_prompt=self.SYSTEM_PROMPT, token_queue=stream_queue
            )
        return await call_llm_stream_async(prompt, system_prompt=self.SYSTEM_PROMPT)

    async def post_async(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        
        session["model_responses"]["model_b"] = exec_res
        session["debate_transcript"].append({
            "speaker": "model_b",
            "content": exec_res
        })
        
        save_conversation(conversation_id, session)
        
        flow_log = shared["flow_queue"]
        flow_log.put("✅ Model B (Cautious & Analytical) has responded")
        
        return "default"


class ModelCResponse(AsyncNode):
    SYSTEM_PROMPT = """You are Model C, a creative and alternative-thinking AI assistant.
Your style is innovative, looking at problems from different angles and considering non-obvious solutions.
You challenge conventional thinking and offer fresh perspectives."""

    async def prep_async(self, shared):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        stream_queue = shared.get("stream_queue")
        return (session["debate_context"], stream_queue)

    async def exec_async(self, prep_res):
        context, stream_queue = prep_res
        prompt = f"""{context}

### YOUR TASK
Provide your best answer to the user's question from a unique or alternative perspective.
Consider:
1. Are there non-obvious approaches the user hasn't considered?
2. Can you reframe the problem in a useful way?
3. What creative solutions exist beyond the conventional answer?
4. How might different contexts or stakeholders view this differently?

Keep your response concise but insightful."""
        
        if stream_queue is not None:
            return await call_llm_stream_async(
                prompt, system_prompt=self.SYSTEM_PROMPT, token_queue=stream_queue
            )
        return await call_llm_stream_async(prompt, system_prompt=self.SYSTEM_PROMPT)

    async def post_async(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        
        session["model_responses"]["model_c"] = exec_res
        session["debate_transcript"].append({
            "speaker": "model_c",
            "content": exec_res
        })
        
        save_conversation(conversation_id, session)
        
        flow_log = shared["flow_queue"]
        flow_log.put("✅ Model C (Creative & Alternative) has responded")
        
        return "default"


class ParallelModels(AsyncNode):
    """Runs Model A, B, C in parallel using asyncio.gather()."""
    
    async def prep_async(self, shared):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        stream_queue = shared.get("stream_queue")
        return (session["debate_context"], stream_queue)

    async def exec_async(self, prep_res):
        context, stream_queue = prep_res
        
        # Create model instances
        model_a = ModelAResponse()
        model_b = ModelBResponse()
        model_c = ModelCResponse()
        
        # Build a minimal shared context for each model
        # They need access to conversation_id and stream_queue
        # But we can't pass the full shared (would include queues)
        # So we manually call prep → exec → post for each
        
        # Actually, PocketFlow nodes expect to be run in a flow context
        # For parallel execution, we'll manually orchestrate them
        tracer = get_tracer()
        
        with tracer.start_as_current_span("parallel_models"):
            log.info("parallel_models_start")
            
            # Run all three models concurrently
            # Each model will stream tokens to the same stream_queue
            results = await asyncio.gather(
                self._run_model(model_a, context, stream_queue, "model_a"),
                self._run_model(model_b, context, stream_queue, "model_b"),
                self._run_model(model_c, context, stream_queue, "model_c"),
                return_exceptions=True,
            )
            
            # Check for failures
            responses = {}
            for i, (name, result) in enumerate(zip(["model_a", "model_b", "model_c"], results)):
                if isinstance(result, Exception):
                    log.error(f"parallel_models_failed", model=name, error=str(result))
                    responses[name] = f"Error: {result}"
                else:
                    responses[name] = result
            
            log.info("parallel_models_done")
            return responses

    async def _run_model(self, node, context, stream_queue, phase_name):
        """Run a single model and return its response."""
        tracer = get_tracer()
        with tracer.start_as_current_span(f"model_{phase_name}"):
            # Send phase marker to UI
            if stream_queue is not None:
                from main import PHASE_SENTINEL_PREFIX, PHASE_SENTINEL_SUFFIX
                stream_queue.put(f"{PHASE_SENTINEL_PREFIX}{phase_name}{PHASE_SENTINEL_SUFFIX}")
            
            # Call the LLM directly
            prompt = self._build_prompt(context, phase_name)
            system_prompt = node.SYSTEM_PROMPT
            
            response = await call_llm_stream_async(
                prompt, system_prompt=system_prompt, token_queue=stream_queue
            )
            
            return response
    
    def _build_prompt(self, context, model_name):
        """Build appropriate prompt for each model."""
        if model_name == "model_a":
            return f"""{context}

### YOUR TASK
Provide your best answer to the user's question. Be direct, practical, and actionable.
Keep your response concise but complete."""
        elif model_name == "model_b":
            return f"""{context}

### YOUR TASK
Provide your best answer to the user's question, but be sure to:
1. Identify key assumptions you're making
2. Point out edge cases or exceptions the user should consider
3. Highlight any risks or caveats
4. Suggest what information might be missing for a complete answer

Keep your response concise but thorough."""
        else:  # model_c
            return f"""{context}

### YOUR TASK
Provide your best answer to the user's question from a unique or alternative perspective.
Consider:
1. Are there non-obvious approaches the user hasn't considered?
2. Can you reframe the problem in a useful way?
3. What creative solutions exist beyond the conventional answer?
4. How might different contexts or stakeholders view this differently?

Keep your response concise but insightful."""

    async def post_async(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        
        # Store all responses
        for model_name, response in exec_res.items():
            session["model_responses"][model_name] = response
            session["debate_transcript"].append({
                "speaker": model_name,
                "content": response
            })
        
        save_conversation(conversation_id, session)
        
        flow_log = shared["flow_queue"]
        flow_log.put("✅ All three models have responded (parallel)")
        
        return "default"


class DebateRound(AsyncNode):
    SYSTEM_PROMPT = """You are facilitating a debate between three AI models.
Your job is to produce a concise critique round where each model reviews the others' answers.
Be objective and focus on substantive points."""

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
        
        prompt = f"""{debate_context}

### INITIAL RESPONSES FROM THREE MODELS

**Model A (Direct & Practical):**
{model_responses.get('model_a', 'No response')}

**Model B (Cautious & Analytical):**
{model_responses.get('model_b', 'No response')}

**Model C (Creative & Alternative):**
{model_responses.get('model_c', 'No response')}

### DEBATE ROUND TASK
Produce a concise debate critique. For each model, identify:
1. What are the strongest points in their answer?
2. What are the weaknesses or gaps?
3. What would improve their response?

Format as a structured critique. Be specific and constructive."""
        
        if stream_queue is not None:
            return await call_llm_stream_async(
                prompt, system_prompt=self.SYSTEM_PROMPT, token_queue=stream_queue
            )
        return await call_llm_stream_async(prompt, system_prompt=self.SYSTEM_PROMPT)

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
    SYSTEM_PROMPT = """You are the Judge, a senior expert reviewer.
Your job is to synthesize multiple model responses and their debate into one final, high-quality answer.
You must resolve conflicts, select the best ideas, and produce a coherent response.
Never simply copy one model's answer - always synthesize."""

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
        
        prompt = f"""{debate_context}

### MODEL RESPONSES

**Model A (Direct & Practical):**
{model_responses.get('model_a', 'No response')}

**Model B (Cautious & Analytical):**
{model_responses.get('model_b', 'No response')}

**Model C (Creative & Alternative):**
{model_responses.get('model_c', 'No response')}

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
            return await call_llm_stream_async(
                prompt, system_prompt=self.SYSTEM_PROMPT, token_queue=stream_queue
            )
        return await call_llm_stream_async(prompt, system_prompt=self.SYSTEM_PROMPT)

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
