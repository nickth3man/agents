import logging
import threading
import time

logger = logging.getLogger(__name__)

_CACHE_TTL = 1800
_cache_lock = threading.Lock()
_conversation_cache: dict[str, tuple[float, dict]] = {}


def _evict_expired():
    now = time.time()
    expired = [k for k, (ts, _) in _conversation_cache.items() if now - ts > _CACHE_TTL]
    for k in expired:
        del _conversation_cache[k]
    if expired:
        logger.debug("evicted %d expired sessions", len(expired))


def load_conversation(conversation_id: str):
    with _cache_lock:
        _evict_expired()
        entry = _conversation_cache.get(conversation_id)
        if entry is None:
            return {}
        ts, session = entry
        logger.debug("load_conversation id=%s keys=%s age=%.0fs", conversation_id, list(session.keys()), time.time() - ts)
        return session


def save_conversation(conversation_id: str, session: dict):
    with _cache_lock:
        _evict_expired()
        _conversation_cache[conversation_id] = (time.time(), session)
        logger.debug("save_conversation id=%s keys=%s", conversation_id, list(session.keys()))


def delete_conversation(conversation_id: str):
    with _cache_lock:
        _conversation_cache.pop(conversation_id, None)
        logger.debug("delete_conversation id=%s", conversation_id)
