"""Async nodes for sequential multi-round conversational debate."""

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


DEBATE_ROUNDS = 2
SPEAKER_ORDER = ["model_a", "model_b", "model_c"]
SPEAKER_LABELS = {
    "model_a": "Model A (Direct & Practical)",
    "model_b": "Model B (Cautious & Analytical)",
    "model_c": "Model C (Creative & Alternative)",
}
SPEAKER_SHORT = {
    "model_a": "A",
    "model_b": "B",
    "model_c": "C",
}


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
        session["conversation_log"] = []
        session["judge_answer"] = None

        save_conversation(conversation_id, session)

        flow_log = shared["flow_queue"]
        flow_log.put("preparing")

        return "default"


MODEL_SYSTEM_PROMPTS = {
    "model_a": _prompt("model_a_system.txt"),
    "model_b": _prompt("model_b_system.txt"),
    "model_c": _prompt("model_c_system.txt"),
}
MODEL_OPENING_PROMPTS = {
    "model_a": _prompt("model_a_user.txt"),
    "model_b": _prompt("model_b_user.txt"),
    "model_c": _prompt("model_c_user.txt"),
}
MODEL_REPLY_SYSTEM = {
    "model_a": _prompt("model_a_reply_system.txt"),
    "model_b": _prompt("model_b_reply_system.txt"),
    "model_c": _prompt("model_c_reply_system.txt"),
}


def _build_opening_prompt(speaker, context, conversation_log):
    if speaker == "model_a":
        return MODEL_OPENING_PROMPTS["model_a"].format(context=context)
    elif speaker == "model_b":
        a_resp = next(
            (e["content"] for e in conversation_log if e["speaker"] == "model_a"),
            "No response yet.",
        )
        return MODEL_OPENING_PROMPTS["model_b"].format(
            context=context, model_a_response=a_resp
        )
    elif speaker == "model_c":
        a_resp = next(
            (e["content"] for e in conversation_log if e["speaker"] == "model_a"),
            "No response yet.",
        )
        b_resp = next(
            (e["content"] for e in conversation_log if e["speaker"] == "model_b"),
            "No response yet.",
        )
        return MODEL_OPENING_PROMPTS["model_c"].format(
            context=context,
            model_a_response=a_resp,
            model_b_response=b_resp,
        )



class ConversationRound(AsyncNode):
    """Runs one round of the sequential debate: A → B → C."""

    async def prep_async(self, shared):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        stream_queue = shared.get("stream_queue")
        return session, shared, stream_queue

    async def exec_async(self, prep_res):
        session, shared, stream_queue = prep_res
        context = session["debate_context"]
        conversation_log = session.get("conversation_log", [])
        current_round = shared.get("current_round", 0)

        tracer = get_tracer()
        import time

        with tracer.start_as_current_span(f"conversation_round_{current_round}"):
            for speaker in SPEAKER_ORDER:
                t0 = time.perf_counter()
                phase_name = f"round{current_round}_{speaker}" if current_round > 0 else speaker
                log.info("speaker_start", phase=phase_name, speaker=speaker)

                if stream_queue is not None:
                    from main import PHASE_SENTINEL_PREFIX, PHASE_SENTINEL_SUFFIX
                    stream_queue.put(f"{PHASE_SENTINEL_PREFIX}{phase_name}{PHASE_SENTINEL_SUFFIX}")

                if current_round == 0:
                    prompt = _build_opening_prompt(speaker, context, conversation_log)
                    system_prompt = MODEL_SYSTEM_PROMPTS[speaker]
                else:
                    prompt = self._build_reply(speaker, context, conversation_log, current_round)
                    system_prompt = MODEL_REPLY_SYSTEM[speaker]

                response = await call_llm_stream(
                    prompt, system_prompt=system_prompt, token_queue=stream_queue
                )
                elapsed = time.perf_counter() - t0
                log.info("speaker_done", phase=phase_name, elapsed_ms=round(elapsed * 1000, 2))

                conversation_log.append({
                    "speaker": speaker,
                    "content": response,
                    "round": current_round,
                })

                flow_log = shared["flow_queue"]
                flow_log.put(f"speaker_done:{speaker}:round{current_round}")

        return conversation_log

    def _build_reply(self, speaker, context, conversation_log, current_round):
        transcript_lines = []
        for entry in conversation_log:
            label = SPEAKER_LABELS.get(entry["speaker"], entry["speaker"])
            short = SPEAKER_SHORT[entry["speaker"]]
            transcript_lines.append(f"**{short}:** {entry['content']}")
        transcript = "\n\n---\n\n".join(transcript_lines)

        return f"""{context}

### CONVERSATION SO FAR (Round {current_round} of {DEBATE_ROUNDS})
{transcript}

### YOUR TURN
You are {SPEAKER_SHORT[speaker]} — {SPEAKER_LABELS[speaker].split('(')[1].rstrip(')')}.
Continue the conversation. Respond directly to what was just said.
Agree, push back, refine your position, or concede a point.
Keep it conversational and punchy. Under 200 words."""

    async def post_async(self, shared, prep_res, exec_res):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        session["conversation_log"] = exec_res
        save_conversation(conversation_id, session)

        current_round = shared.get("current_round", 0)
        shared["current_round"] = current_round + 1
        if current_round + 1 < DEBATE_ROUNDS:
            return "next_round"
        return "to_judge"


class JudgeFinal(AsyncNode):
    SYSTEM_PROMPT = _prompt("judge_final_system.txt")

    async def prep_async(self, shared):
        conversation_id = shared["conversation_id"]
        session = load_conversation(conversation_id)
        stream_queue = shared.get("stream_queue")
        return (
            session["debate_context"],
            session.get("conversation_log", []),
            stream_queue,
        )

    async def exec_async(self, prep_res):
        debate_context, conversation_log, stream_queue = prep_res

        transcript_lines = []
        for entry in conversation_log:
            label = SPEAKER_LABELS.get(entry["speaker"], entry["speaker"])
            short = SPEAKER_SHORT[entry["speaker"]]
            rnd = entry.get("round", 0)
            transcript_lines.append(f"[Round {rnd + 1}] **{short}:** {entry['content']}")
        transcript = "\n\n---\n\n".join(transcript_lines)

        prompt = f"""{debate_context}

### FULL DEBATE TRANSCRIPT
{transcript}

### YOUR TASK AS FINAL JUDGE
Synthesize the above conversation into ONE final answer for the user. You must:
1. Resolve any conflicts or disagreements between models
2. Combine the strongest points from each response
3. Remove weak or unsupported claims
4. Ensure the final answer is clear, accurate, and directly addresses the user's question
5. Maintain an enthusiastic and helpful tone

Return ONLY the final answer. Do not explain your judging process."""

        if stream_queue is not None:
            from main import PHASE_SENTINEL_PREFIX, PHASE_SENTINEL_SUFFIX
            stream_queue.put(f"{PHASE_SENTINEL_PREFIX}judge{PHASE_SENTINEL_SUFFIX}")
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
        flow_log.put("judge_done")

        return "default"


class SendFinalAnswer(AsyncNode):
    async def prep_async(self, shared):
        flow_log = shared["flow_queue"]
        flow_log.put(None)

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
