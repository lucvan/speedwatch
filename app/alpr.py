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
from . import anpr_mask

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


def _plate_bbox_frame(r, off_x: int, off_y: int) -> tuple[float, float, float, float] | None:
    """A detected plate's bbox in *frame* pixels (crop bbox + crop offset).
    None if the detector result carries no bounding box."""
    bb = getattr(getattr(r, "detection", None), "bounding_box", None)
    if bb is None:
        return None
    try:
        return (bb.x1 + off_x, bb.y1 + off_y, bb.x2 + off_x, bb.y2 + off_y)
    except Exception:
        return None


def _plate_centre_frame(r, off_x: int, off_y: int) -> tuple[float, float] | None:
    """Centre of a detected plate's bbox in *frame* pixels. None if no bounding box."""
    fb = _plate_bbox_frame(r, off_x, off_y)
    if fb is None:
        return None
    x1, y1, x2, y2 = fb
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def read_plates_located(frame_bgr: np.ndarray,
                        bbox_xyxy: tuple[float, float, float, float],
                        camera: str | None = None) -> list[dict]:
    """
    Detect + OCR all number plates inside the car bounding box. Returns one dict per read
    above ALPR_MIN_CONF: {text, conf, box, centre} where `box` is the plate's bounding box and
    `centre` its centre, both normalised to the *full frame* (0..1) — or None if the detector
    gave no bounding box. The frame-normalised location is what lets the caller offer to add an
    ANPR ignore box for a fixed parked/background plate.

    Plates whose detection centre falls inside a configured ANPR ignore zone (see `anpr_mask`)
    are discarded here, so masked plates never reach the caller.
    """
    if not config.ENABLE_ALPR or frame_bgr is None or bbox_xyxy is None:
        return []
    alpr = _get_alpr()
    if alpr is None:
        return []

    cam = camera or config.CAMERA_NAME
    h, w = frame_bgr.shape[:2]
    pad = config.ALPR_CROP_PAD
    x1, y1, x2, y2 = bbox_xyxy
    cx1 = max(0, int(x1) - pad); cy1 = max(0, int(y1) - pad)
    cx2 = min(w, int(x2) + pad); cy2 = min(h, int(y2) + pad)
    if cx2 - cx1 < 16 or cy2 - cy1 < 16:
        return []
    crop = frame_bgr[cy1:cy2, cx1:cx2]

    try:
        results = alpr.predict(crop)
    except Exception as e:
        log.warning("ALPR predict failed: %s", e)
        return []

    reads: list[dict] = []
    for r in results:
        if not r.ocr:
            continue
        text = _normalise(r.ocr.text)
        conf = _mean_conf(r.ocr.confidence)
        if len(text) < 4 or conf < config.ALPR_MIN_CONF:
            continue
        fb = _plate_bbox_frame(r, cx1, cy1)
        centre = None
        box = None
        if fb is not None and w and h:
            fx1, fy1, fx2, fy2 = fb
            cxn = ((fx1 + fx2) / 2.0) / w
            cyn = ((fy1 + fy2) / 2.0) / h
            # Spatial ignore zone: drop plates detected at a masked frame location.
            if anpr_mask.is_ignored(cam, cxn, cyn):
                log.info("ignored plate %s at masked region (%.3f,%.3f)", text, cxn, cyn)
                continue
            centre = [round(cxn, 5), round(cyn, 5)]
            box = [round(max(0.0, min(1.0, fx1 / w)), 5), round(max(0.0, min(1.0, fy1 / h)), 5),
                   round(max(0.0, min(1.0, fx2 / w)), 5), round(max(0.0, min(1.0, fy2 / h)), 5)]
        reads.append({"text": text, "conf": round(conf, 3), "box": box, "centre": centre})
    return reads


def read_plates(frame_bgr: np.ndarray,
                bbox_xyxy: tuple[float, float, float, float],
                camera: str | None = None) -> list[tuple[str, float]]:
    """Every plate read above ALPR_MIN_CONF as (plate_text, mean_confidence)."""
    return [(r["text"], r["conf"]) for r in read_plates_located(frame_bgr, bbox_xyxy, camera)]


def read_plate(frame_bgr: np.ndarray,
               bbox_xyxy: tuple[float, float, float, float],
               camera: str | None = None) -> tuple[str, float] | None:
    """
    Detect + OCR the number plate inside the car bounding box.
    Returns (plate_text, mean_confidence) for the best plate above ALPR_MIN_CONF, or None.
    """
    reads = read_plates(frame_bgr, bbox_xyxy, camera)
    if not reads:
        return None
    return max(reads, key=lambda r: r[1])
