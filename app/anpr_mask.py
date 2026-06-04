"""
ANPR ignore zones — rectangular regions of the frame where detected number plates are
discarded.

Use case: a parked/background car whose plate falls inside the moving car's bounding-box
crop, so ANPR keeps reading the *background* plate for every pass. Because that plate sits
at a fixed location, masking the region kills it (and every misread of it) regardless of
what it OCRs to — and a car that actually drives through still reads fine elsewhere.

Rectangles are stored per camera as normalised [x1, y1, x2, y2] (0..1) in
``<DATA_DIR>/anpr_mask/<camera>.json`` and edited on the Calibrate page. The file is
mtime-cached so the ingest thread picks up edits without a restart.
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path

from . import config

log = logging.getLogger(__name__)

# camera -> (mtime, rects)
_cache: dict[str, tuple[float, list[list[float]]]] = {}


def _dir() -> Path:
    return Path(config.DATA_DIR) / "anpr_mask"


def _path(camera: str) -> Path:
    return _dir() / f"{camera}.json"


def load(camera: str) -> list[list[float]]:
    """Normalised [x1,y1,x2,y2] ignore rectangles for the camera ([] if none). mtime-cached."""
    p = _path(camera)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        _cache.pop(camera, None)
        return []
    cached = _cache.get(camera)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        data = json.loads(p.read_text())
        rects = [[float(v) for v in r] for r in data.get("rects", []) if len(r) == 4]
    except Exception as e:
        log.warning("Could not read ANPR mask %s: %s", p, e)
        rects = []
    _cache[camera] = (mtime, rects)
    return rects


def save(camera: str, rects: list) -> Path:
    """Persist ignore rectangles (clamped to 0..1, tiny ones dropped). Invalidates the cache."""
    clean: list[list[float]] = []
    for r in rects or []:
        if len(r) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in r)
        x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
        y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
        if (x2 - x1) < 0.005 or (y2 - y1) < 0.005:
            continue
        clean.append([round(x1, 5), round(y1, 5), round(x2, 5), round(y2, 5)])
    d = _dir()
    d.mkdir(parents=True, exist_ok=True)
    p = _path(camera)
    p.write_text(json.dumps({"camera": camera, "rects": clean, "updated_at": time.time()}, indent=2))
    _cache.pop(camera, None)
    return p


def is_ignored(camera: str, x_norm: float, y_norm: float) -> bool:
    """True if the normalised point falls inside any ignore rectangle for the camera."""
    for x1, y1, x2, y2 in load(camera):
        if x1 <= x_norm <= x2 and y1 <= y_norm <= y2:
            return True
    return False
