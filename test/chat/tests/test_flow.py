import pytest
from queue import Queue
from unittest.mock import AsyncMock

from flow import create_flow


class TestFullFlow:
    @pytest.mark.asyncio
    async def test_end_to_end_flow(self, mock_call_llm_stream, shared_state):
        call_count = 0

        async def llm_side_effect(message, system_prompt=None, token_queue=None):
            nonlocal call_count
            call_count += 1
            if system_prompt and "Judge" in system_prompt:
                return "The answer is 42"
            if "debate" in message.lower() or "critique" in message.lower():
                return "Debate round completed"
            return f"Model response {call_count}"

        mock_call_llm_stream.side_effect = llm_side_effect

        flow = create_flow()
        await flow.run_async(shared_state)

        final_answer = shared_state["queue"].get()
        assert final_answer == "The answer is 42"
        assert shared_state["queue"].get() is None

    @pytest.mark.asyncio
    async def test_flow_with_error_response(self, mock_call_llm_stream, shared_state):
        async def llm_side_effect(message, system_prompt=None, token_queue=None):
            if system_prompt and "Judge" in system_prompt:
                return "Synthesized answer despite errors"
            if "debate" in message.lower() or "critique" in message.lower():
                return "Debate with missing models"
            if "direct and practical" in system_prompt.lower():
                raise RuntimeError("Model A failed")
            return "Valid model response"

        mock_call_llm_stream.side_effect = llm_side_effect

        flow = create_flow()
        await flow.run_async(shared_state)

        final_answer = shared_state["queue"].get()
        assert final_answer == "Synthesized answer despite errors"
        assert shared_state["queue"].get() is None
