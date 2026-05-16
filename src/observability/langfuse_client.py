"""
Langfuse client — graceful no-op when not configured.

When LANGFUSE_ENABLED=true and keys are set, every agent decision and email
draft gets a Langfuse span. When disabled, all trace operations become
no-ops so the rest of the code stays clean.
"""
import os
from contextlib import contextmanager
from typing import Optional
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

_LF_CLIENT = None


class _NoOpTrace:
    """A tiny stand-in object that swallows all trace API calls."""
    def __init__(self, name: str = ""):
        self.name = name
    def update(self, **kwargs): pass
    def span(self, **kwargs): return _NoOpTrace(kwargs.get("name", ""))
    def generation(self, **kwargs): return _NoOpTrace(kwargs.get("name", ""))
    def event(self, **kwargs): pass
    def end(self, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


def get_langfuse():
    """Lazy-load the Langfuse client, return None if disabled or unconfigured."""
    global _LF_CLIENT
    if _LF_CLIENT is not None:
        return _LF_CLIENT

    enabled = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
    if not enabled:
        return None

    pub = os.getenv("LANGFUSE_PUBLIC_KEY")
    sec = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    if not (pub and sec):
        logger.warning("Langfuse enabled but keys not set; running without tracing")
        return None

    try:
        from langfuse import Langfuse
        _LF_CLIENT = Langfuse(public_key=pub, secret_key=sec, host=host)
        logger.info(f"Langfuse connected → {host}")
        return _LF_CLIENT
    except Exception as e:
        logger.warning(f"Langfuse init failed: {e}")
        return None


@contextmanager
def with_trace(name: str, **trace_kwargs):
    """
    Context manager that yields a Langfuse trace OR a no-op stand-in.
    Use like:
        with with_trace("ar_collections_run", user_id="...") as t:
            t.span(name="action_planner", input={...}, output={...})
    """
    client = get_langfuse()
    if client is None:
        yield _NoOpTrace(name)
        return
    try:
        trace = client.trace(name=name, **trace_kwargs)
        yield trace
    except Exception as e:
        logger.warning(f"Langfuse trace error: {e}")
        yield _NoOpTrace(name)
