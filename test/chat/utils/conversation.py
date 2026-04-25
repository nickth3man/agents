import logging

logger = logging.getLogger(__name__)

conversation_cache = {}


def load_conversation(conversation_id: str):
    session = conversation_cache.get(conversation_id, {})
    logger.debug("load_conversation id=%s keys=%s", conversation_id, list(session.keys()))
    return session


def save_conversation(conversation_id: str, session: dict):
    logger.debug("save_conversation id=%s keys=%s", conversation_id, list(session.keys()))
    conversation_cache[conversation_id] = session
