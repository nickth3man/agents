import pytest
from queue import Queue
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture(autouse=True)
def mock_observability(monkeypatch):
    """Mock observability to avoid side effects (log files, tracing)."""
    dummy_tracer = MagicMock()
    dummy_span = MagicMock()
    dummy_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=dummy_span)
    dummy_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr("nodes.get_tracer", lambda: dummy_tracer)
    monkeypatch.setattr("nodes.log", MagicMock())
    monkeypatch.setattr("utils.observability.get_tracer", lambda: dummy_tracer)
    monkeypatch.setattr("utils.observability.logger", MagicMock())
    monkeypatch.setattr("utils.observability.bind_request", lambda x: None)


@pytest.fixture(autouse=True)
def mock_conversation_storage(monkeypatch):
    """Mock conversation storage for all tests."""
    sessions = {}

    def load_conversation(cid):
        return sessions.setdefault(cid, {})

    def save_conversation(cid, session):
        sessions[cid] = session

    def delete_conversation(cid):
        sessions.pop(cid, None)

    monkeypatch.setattr("nodes.load_conversation", load_conversation)
    monkeypatch.setattr("nodes.save_conversation", save_conversation)
    monkeypatch.setattr("main.delete_conversation", delete_conversation)
    return sessions


@pytest.fixture
def mock_format_chat_history(monkeypatch):
    """Fixture to mock format_chat_history."""
    def _install(return_value="Formatted history"):
        monkeypatch.setattr("nodes.format_chat_history", lambda h: return_value)
        return return_value

    return _install


@pytest.fixture
def mock_call_llm_stream(monkeypatch):
    """Fixture that mocks call_llm_stream to return deterministic responses."""
    mock = AsyncMock()

    async def default_side_effect(message, system_prompt=None, token_queue=None):
        if system_prompt and "Judge" in system_prompt:
            return "Synthesized final answer from judge"
        if "debate" in message.lower() or "critique" in message.lower():
            return "Debate round critique"
        return "Default model response"

    mock.side_effect = default_side_effect
    monkeypatch.setattr("nodes.call_llm_stream", mock)
    return mock


@pytest.fixture
def shared_state():
    """Fixture providing a standard shared dict for flow tests."""
    return {
        "conversation_id": "test-conv-123",
        "query": "What is the meaning of life?",
        "history": [{"role": "user", "content": "Hello"}],
        "queue": Queue(),
        "flow_queue": Queue(),
        "stream_queue": None,
    }
