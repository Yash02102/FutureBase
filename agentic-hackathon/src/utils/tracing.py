import contextlib
import logging
import time


@contextlib.contextmanager
def trace_span(name: str):
    logger = logging.getLogger("trace")
    start = time.time()
    try:
        logger.info("span_start %s", name)
        yield
    finally:
        duration = time.time() - start
        logger.info("span_end %s duration=%.3fs", name, duration)