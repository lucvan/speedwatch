"""
Lightweight per-stage timing profiler. Accumulates per-stage durations and logs
the averages periodically. Enabled by config.PROFILE — when off, record() returns
before doing any work, so the only cost is a couple of perf_counter() calls.

Usage:
    t = profiling.start()
    ... work ...
    profiling.record("mog2", t)      # records elapsed since t
    profiling.maybe_flush()          # call from a loop; logs every PROFILE_FLUSH_S
"""
from __future__ import annotations
import logging
import threading
import time
from collections import defaultdict

from . import config

log = logging.getLogger("app.profile")

_lock = threading.Lock()
_sum_ms: dict[str, float] = defaultdict(float)
_count: dict[str, int] = defaultdict(int)
_last_flush = time.time()

# Stable display order (others appended alphabetically)
_ORDER = ["pipe_read", "frame_copy", "mog2",
          "yolo_pre", "yolo_infer", "yolo_post", "track",
          "clip_encode", "stills", "alpr"]


def start() -> float:
    return time.perf_counter()


def record(stage: str, t_start: float) -> None:
    if not config.PROFILE:
        return
    ms = (time.perf_counter() - t_start) * 1000.0
    with _lock:
        _sum_ms[stage] += ms
        _count[stage] += 1


def maybe_flush() -> None:
    if not config.PROFILE:
        return
    global _last_flush
    now = time.time()
    if now - _last_flush < config.PROFILE_FLUSH_S:
        return
    with _lock:
        window = now - _last_flush
        _last_flush = now
        if not _count:
            return
        sums = dict(_sum_ms)
        counts = dict(_count)
        _sum_ms.clear()
        _count.clear()

    keys = [k for k in _ORDER if k in counts] + sorted(k for k in counts if k not in _ORDER)
    parts = []
    for k in keys:
        n = counts[k]
        avg = sums[k] / n if n else 0.0
        per_s = sums[k] / window  # ms of CPU spent in this stage per wall-second
        parts.append(f"{k} {avg:.1f}ms x{n} ({per_s:.0f}ms/s)")
    log.info("PROFILE %.0fs | %s", window, "  ".join(parts))
