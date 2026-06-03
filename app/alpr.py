"""
Number-plate recognition via fast-alpr (plate detector + OCR, ONNX on CPU).

Runs on CPU so it does not contend with YOLO11 on DmlExecutionProvider. The plate
models are tiny (~10-40 ms per crop). We always pass a *crop* of the detected car's
bounding box rather than the full frame — at full 1080p the plate is too small for
the detector after its internal downscale; cropping to the car box makes it findable.

Thread-safe lazy singleton, mirroring detect.py.
"""
from __future__ import annotations
import logging
import re
import threading

import numpy as np

from . import config

log = logging.getLogger(__name__)

_lock = threading.Lock()
_alpr = None
_unavailable = False  # set if import/load fails, so we stop retrying

_PLATE_RE = re.compile(r"[^A-Z0-9]")


def _get_alpr():
    global _alpr, _unavailable
    if _alpr is not None or _unavailable:
        return _alpr
    with _lock:
        if _alpr is None and not _unavailable:
            try:
                from fast_alpr import ALPR
                log.info("Loading ALPR (detector=%s ocr=%s, CPU)",
                         config.ALPR_DETECTOR_MODEL, config.ALPR_OCR_MODEL)
                _alpr = ALPR(
                    detector_model=config.ALPR_DETECTOR_MODEL,
                    ocr_model=config.ALPR_OCR_MODEL,
                    detector_providers=["CPUExecutionProvider"],
                    ocr_providers=["CPUExecutionProvider"],
                )
                log.info("ALPR ready")
            except Exception as e:
                _unavailable = True
                log.error("ALPR unavailable — plate recognition disabled: %s", e)
    return _alpr


def warmup():
    """Load the ALPR models up front (called from the ingest thread at startup)."""
    if config.ENABLE_ALPR:
        _get_alpr()


def _mean_conf(conf) -> float:
    if conf is None:
        return 0.0
    if hasattr(conf, "__len__"):
        vals = [float(c) for c in conf]
        return float(np.mean(vals)) if vals else 0.0
    return float(conf)


def _normalise(text: str) -> str:
    return _PLATE_RE.sub("", (text or "").upper())


def read_plate(frame_bgr: np.ndarray,
               bbox_xyxy: tuple[float, float, float, float]) -> tuple[str, float] | None:
    """
    Detect + OCR the number plate inside the car bounding box.
    Returns (plate_text, mean_confidence) for the best plate above ALPR_MIN_CONF, or None.
    """
    if not config.ENABLE_ALPR or frame_bgr is None or bbox_xyxy is None:
        return None
    alpr = _get_alpr()
    if alpr is None:
        return None

    h, w = frame_bgr.shape[:2]
    pad = config.ALPR_CROP_PAD
    x1, y1, x2, y2 = bbox_xyxy
    cx1 = max(0, int(x1) - pad); cy1 = max(0, int(y1) - pad)
    cx2 = min(w, int(x2) + pad); cy2 = min(h, int(y2) + pad)
    if cx2 - cx1 < 16 or cy2 - cy1 < 16:
        return None
    crop = frame_bgr[cy1:cy2, cx1:cx2]

    try:
        results = alpr.predict(crop)
    except Exception as e:
        log.warning("ALPR predict failed: %s", e)
        return None

    best: tuple[str, float] | None = None
    for r in results:
        if not r.ocr:
            continue
        text = _normalise(r.ocr.text)
        conf = _mean_conf(r.ocr.confidence)
        if len(text) < 4 or conf < config.ALPR_MIN_CONF:
            continue
        if best is None or conf > best[1]:
            best = (text, round(conf, 3))
    return best
