import pytest
from queue import Queue
from unittest.mock import AsyncMock, patch

from nodes import (
    PrepareDebate,
    ParallelModels,
    DebateRound,
    Judge,
    SendFinalAnswer,
)


class TestPrepareDebate:
    @pytest.mark.asyncio
    async def test_prepare_debate_formats_context(self, mock_format_chat_history):
        mock_format_chat_history("user: Hello\nassistant: Hi there")

        node = PrepareDebate()
        shared = {
            "conversation_id": "test-conv",
            "history": [{"role": "user", "content": "Hello"}],
            "query": "What is the meaning of life?",
        }

        prep_res = await node.prep_async(shared)
        exec_res = await node.exec_async(prep_res)

        assert "### CHAT HISTORY" in exec_res
        assert "user: Hello\nassistant: Hi there" in exec_res
        assert "What is the meaning of life?" in exec_res
        assert "Current Date:" in exec_res

    @pytest.mark.asyncio
    async def test_prepare_debate_post_async(self, mock_conversation_storage):
        node = PrepareDebate()
        shared = {
            "conversation_id": "test-conv",
            "history": [],
            "query": "Test",
            "flow_queue": Queue(),
        }

        prep_res = await node.prep_async(shared)
        exec_res = await node.exec_async(prep_res)
        result = await node.post_async(shared, prep_res, exec_res)

        session = mock_conversation_storage["test-conv"]
        assert session["debate_context"] == exec_res
        assert session["model_responses"] == {}
        assert session["debate_transcript"] == []
        assert session["judge_answer"] is None
        assert result == "default"
        assert shared["flow_queue"].get() == "🎯 Preparing debate context..."


class TestParallelModels:
    @pytest.mark.asyncio
    async def test_exec_async_returns_all_models(self, mock_call_llm_stream):

        async def side_effect(message, system_prompt=None, token_queue=None):
            if "pragmatic senior engineer" in system_prompt.lower():
                return "Response from Model A"
            if "cautious technical lead" in system_prompt.lower():
                return "Response from Model B"
            if "unconventional systems thinker" in system_prompt.lower():
                return "Response from Model C"
            return "Unknown"

        mock_call_llm_stream.side_effect = side_effect

        node = ParallelModels()
        result = await node.exec_async(("test context", None))

        assert isinstance(result, dict)
        assert "model_a" in result
        assert "model_b" in result
        assert "model_c" in result
        assert result["model_a"] == "Response from Model A"
        assert result["model_b"] == "Response from Model B"
        assert result["model_c"] == "Response from Model C"

    @pytest.mark.asyncio
    async def test_exec_async_catches_exception(self, mock_call_llm_stream):

        async def side_effect(message, system_prompt=None, token_queue=None):
            if "pragmatic senior engineer" in system_prompt.lower():
                raise RuntimeError("Model A crashed")
            if "cautious technical lead" in system_prompt.lower():
                return "Response from Model B"
            if "unconventional systems thinker" in system_prompt.lower():
                return "Response from Model C"
            return "Unknown"

        mock_call_llm_stream.side_effect = side_effect

        node = ParallelModels()
        result = await node.exec_async(("test context", None))

        assert result["model_a"].startswith("Error:")
        assert "RuntimeError" in result["model_a"] or "Model A crashed" in result["model_a"]
        assert result["model_b"] == "Response from Model B"
        assert result["model_c"] == "Response from Model C"


class TestDebateRound:
    @pytest.mark.asyncio
    async def test_exec_async_filters_error_responses(self, mock_call_llm_stream):
        mock_call_llm_stream.side_effect = None
        mock_call_llm_stream.return_value = "Debate critique result"

        node = DebateRound()
        model_responses = {
            "model_a": "Error: something went wrong",
            "model_b": "Good response from B",
            "model_c": "Error: another failure",
        }

        result = await node.exec_async(("debate context", model_responses, None))

        assert result == "Debate critique result"

        call_args = mock_call_llm_stream.call_args
        prompt = call_args[0][0]
        assert "[This model failed to respond — skip in critique]" in prompt
        assert "Good response from B" in prompt
        assert "Error: something went wrong" not in prompt


class TestJudge:
    @pytest.mark.asyncio
    async def test_exec_async_filters_error_responses(self, mock_call_llm_stream):
        mock_call_llm_stream.side_effect = None
        mock_call_llm_stream.return_value = "Final synthesized answer"

        node = Judge()
        model_responses = {
            "model_a": "Error: model failed",
            "model_b": "Valid response B",
            "model_c": "Valid response C",
        }
        debate_critique = "Some critique"

        result = await node.exec_async(("debate context", model_responses, debate_critique, None))

        assert result == "Final synthesized answer"

        call_args = mock_call_llm_stream.call_args
        prompt = call_args[0][0]
        assert "[This model failed to respond]" in prompt
        assert "Valid response B" in prompt
        assert "Valid response C" in prompt
        assert "Error: model failed" not in prompt


class TestSendFinalAnswer:
    @pytest.mark.asyncio
    async def test_puts_answer_and_none_into_queue(self):
        node = SendFinalAnswer()
        queue = Queue()
        answer = "The final verdict is 42"

        result = await node.exec_async((answer, queue))

        assert result == answer
        assert queue.get() == answer
        assert queue.get() is None
        assert queue.empty()

    @pytest.mark.asyncio
    async def test_prep_async_signals_end(self, mock_conversation_storage):
        node = SendFinalAnswer()
        shared = {
            "conversation_id": "test-conv",
            "flow_queue": Queue(),
            "stream_queue": Queue(),
            "queue": Queue(),
        }
        mock_conversation_storage["test-conv"] = {"judge_answer": "Final answer"}

        prep_res = await node.prep_async(shared)

        assert prep_res == ("Final answer", shared["queue"])
        assert shared["flow_queue"].get() is None
        assert shared["stream_queue"].get() is None
