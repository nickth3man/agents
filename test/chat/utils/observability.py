"""Observability stack: structured logging + distributed tracing."""

import functools
import logging.handlers

import structlog
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# ── Structured logging ──────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.dict_tracebacks,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.WriteLoggerFactory(
        file=open("debate.log", "a", encoding="utf-8")
    ),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("debate")


def bind_request(conversation_id: str):
    """Bind conversation_id and trace_id to every log entry in this scope."""
    structlog.contextvars.clear_contextvars()
    trace_id = ""
    span = trace.get_current_span()
    if span:
        ctx = span.get_span_context()
        if ctx.is_valid:
            trace_id = trace.format_trace_id(ctx.trace_id)
    structlog.contextvars.bind_contextvars(
        conversation_id=conversation_id,
        trace_id=trace_id,
    )


# ── OpenTelemetry tracing ───────────────────────────────────────────────────

_trace_provider = TracerProvider()
trace.set_tracer_provider(_trace_provider)
_tracer = trace.get_tracer("debate.app")

# Export spans to console (replace with OTLP for production)
_trace_provider.add_span_processor(
    BatchSpanProcessor(ConsoleSpanExporter())
)


def get_tracer():
    return _tracer


# ── Thread-safe trace propagation ───────────────────────────────────────────

def serialize_trace_context() -> dict:
    """Serialize current trace context into a dict for thread hand-off."""
    carrier = {}
    TraceContextTextMapPropagator().inject(carrier)
    return carrier


def extract_trace_context(carrier: dict):
    """Extract trace context from a dict (used in worker threads)."""
    return TraceContextTextMapPropagator().extract(carrier=carrier)


def propagate_trace_context(func):
    """Decorator that re-attaches OTel context in a worker thread.

    Usage:
        @propagate_trace_context
        def my_worker(carrier, conversation_id, ...):
            ...

        executor.submit(my_worker, serialize_trace_context(), conversation_id, ...)
    """
    @functools.wraps(func)
    def wrapper(carrier, *args, **kwargs):
        ctx = extract_trace_context(carrier)
        token = context.attach(ctx)
        try:
            return func(*args, **kwargs)
        finally:
            context.detach(token)

    return wrapper
