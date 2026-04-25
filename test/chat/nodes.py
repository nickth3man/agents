from datetime import datetime
from queue import Queue

from pocketflow import Node

from utils.call_llm import call_llm, call_llm_stream
from utils.conversation import load_conversation, save_conversation
from utils.format_chat_history import format_chat_history


class PrepareDebate(Node):
    def prep(self, shared):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        return session, shared["history"], shared["query"]

    def exec(self, prep_res):
        session, history, query = prep_res
        debate_context = f"""### CHAT HISTORY
{format_chat_history(history)}

### CURRENT USER QUESTION
user: {query}

Current Date: {datetime.now().date()}
"""
        return debate_context

    def post(self, shared, prep_res, exec_res):
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


class ModelAResponse(Node):
    SYSTEM_PROMPT = """You are Model A, a direct and practical AI assistant. 
Your style is concise, action-oriented, and focused on immediate usefulness.
Provide clear, straightforward answers that the user can act on right away."""

    def prep(self, shared):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        stream_queue = shared.get("stream_queue")
        return (session["debate_context"], stream_queue)

    def exec(self, prep_res):
        context, stream_queue = prep_res
        prompt = f"""{context}

### YOUR TASK
Provide your best answer to the user's question. Be direct, practical, and actionable.
Keep your response concise but complete."""
        
        if stream_queue is not None:
            return call_llm_stream(prompt, system_prompt=self.SYSTEM_PROMPT, stream_queue=stream_queue)
        return call_llm(prompt, system_prompt=self.SYSTEM_PROMPT)

    def post(self, shared, prep_res, exec_res):
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


class ModelBResponse(Node):
    SYSTEM_PROMPT = """You are Model B, a cautious and analytical AI assistant.
Your style is thorough, risk-aware, and focused on identifying edge cases, assumptions, and potential problems.
Always consider what could go wrong and what the user might be missing."""

    def prep(self, shared):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        stream_queue = shared.get("stream_queue")
        return (session["debate_context"], stream_queue)

    def exec(self, prep_res):
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
            return call_llm_stream(prompt, system_prompt=self.SYSTEM_PROMPT, stream_queue=stream_queue)
        return call_llm(prompt, system_prompt=self.SYSTEM_PROMPT)

    def post(self, shared, prep_res, exec_res):
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


class ModelCResponse(Node):
    SYSTEM_PROMPT = """You are Model C, a creative and alternative-thinking AI assistant.
Your style is innovative, looking at problems from different angles and considering non-obvious solutions.
You challenge conventional thinking and offer fresh perspectives."""

    def prep(self, shared):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        stream_queue = shared.get("stream_queue")
        return (session["debate_context"], stream_queue)

    def exec(self, prep_res):
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
            return call_llm_stream(prompt, system_prompt=self.SYSTEM_PROMPT, stream_queue=stream_queue)
        return call_llm(prompt, system_prompt=self.SYSTEM_PROMPT)

    def post(self, shared, prep_res, exec_res):
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


class DebateRound(Node):
    SYSTEM_PROMPT = """You are facilitating a debate between three AI models.
Your job is to produce a concise critique round where each model reviews the others' answers.
Be objective and focus on substantive points."""

    def prep(self, shared):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        stream_queue = shared.get("stream_queue")
        return (
            session["debate_context"],
            session["model_responses"],
            stream_queue,
        )

    def exec(self, prep_res):
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
            return call_llm_stream(prompt, system_prompt=self.SYSTEM_PROMPT, stream_queue=stream_queue)
        return call_llm(prompt, system_prompt=self.SYSTEM_PROMPT)

    def post(self, shared, prep_res, exec_res):
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


class Judge(Node):
    SYSTEM_PROMPT = """You are the Judge, a senior expert reviewer.
Your job is to synthesize multiple model responses and their debate into one final, high-quality answer.
You must resolve conflicts, select the best ideas, and produce a coherent response.
Never simply copy one model's answer - always synthesize."""

    def prep(self, shared):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        stream_queue = shared.get("stream_queue")
        return (
            session["debate_context"],
            session["model_responses"],
            session.get("debate_critique", ""),
            stream_queue,
        )

    def exec(self, prep_res):
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
            return call_llm_stream(prompt, system_prompt=self.SYSTEM_PROMPT, stream_queue=stream_queue)
        return call_llm(prompt, system_prompt=self.SYSTEM_PROMPT)

    def post(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        
        session["judge_answer"] = exec_res
        save_conversation(conversation_id, session)
        
        flow_log = shared["flow_queue"]
        flow_log.put("⚖️ Judge has synthesized the final answer")
        
        return "default"


class SendFinalAnswer(Node):
    def prep(self, shared):
        flow_log = shared["flow_queue"]
        flow_log.put(None)
        
        # Signal end of stream if streaming was used
        stream_queue = shared.get("stream_queue")
        if stream_queue is not None:
            stream_queue.put(None)
        
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        return session["judge_answer"], shared["queue"]

    def exec(self, prep_res):
        answer, queue = prep_res
        queue.put(answer)
        queue.put(None)
        return answer

    def post(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        
        session["action_result"] = exec_res
        save_conversation(conversation_id, session)
        
        return "done"
