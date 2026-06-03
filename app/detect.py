"""
In-process YOLO11 inference using onnxruntime directly with DmlExecutionProvider.
Preprocessing and postprocessing are self-contained (numpy + cv2 only).
Thread-safe singleton.
"""
from __future__ import annotations
import logging
import threading
from typing import NamedTuple

import cv2
import numpy as np

from . import config

log = logging.getLogger(__name__)

_lock = threading.Lock()
_session = None
_active_provider: str = "unknown"
_INPUT_SIZE = 640
_CONF_THRESHOLD = 0.25
_IOU_THRESHOLD  = 0.45


class Detection(NamedTuple):
    bbox_xyxy: tuple[float, float, float, float]  # x1, y1, x2, y2  (pixel coords)
    class_id: int
    conf: float


def _resolve_providers() -> list[str]:
    if config.ONNX_PROVIDER != "auto":
        return [config.ONNX_PROVIDER]
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "DmlExecutionProvider" in available:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


def _get_session():
    global _session, _active_provider
    if _session is not None:
        return _session
    with _lock:
        if _session is None:
            import onnxruntime as ort
            providers = _resolve_providers()
            mp = config.model_path()
            log.info("Creating ORT session: %s  providers=%s", mp, providers)
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            _session = ort.InferenceSession(str(mp), sess_options=opts, providers=providers)
            _active_provider = _session.get_providers()[0]
            log.info("ORT session ready — active provider: %s", _active_provider)
    return _session


def warmup():
    """Load model and run warmup inference to compile DML shaders."""
    sess = _get_session()
    inp_name = sess.get_inputs()[0].name
    dummy = np.zeros((1, 3, _INPUT_SIZE, _INPUT_SIZE), dtype=np.float32)
    sess.run(None, {inp_name: dummy})
    log.info("Warmup done — provider: %s", _active_provider)


def _letterbox(
    img: np.ndarray,
    new_size: int = _INPUT_SIZE,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize img to new_size×new_size with letterboxing.
    Returns (resized_img, scale, (pad_left, pad_top)).
    """
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_left = (new_size - nw) // 2
    pad_top  = (new_size - nh) // 2
    pad_right  = new_size - nw - pad_left
    pad_bottom = new_size - nh - pad_top
    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, scale, (pad_left, pad_top)


def _preprocess(frame_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
    img, scale, pad = _letterbox(frame_bgr)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.ascontiguousarray(img.transpose(2, 0, 1)[None])  # HWC→BCHW
    return img, scale, pad


def _postprocess(
    output: np.ndarray,
    scale: float,
    pad: tuple[int, int],
    orig_w: int,
    orig_h: int,
    target_class_ids: list[int],
) -> list[Detection]:
    """
    Decode YOLO11 output tensor.
    output shape: (1, 84, 8400) — cx,cy,w,h + 80 class scores
    """
    # Squeeze batch dim: (84, 8400) → transpose to (8400, 84)
    pred = output[0].T  # (8400, 84)
    bboxes = pred[:, :4]        # cx, cy, w, h  (in 640×640 space)
    class_scores = pred[:, 4:]  # (8400, 80)

    # Max class confidence + class id per box
    class_confs = class_scores.max(axis=1)
    class_ids   = class_scores.argmax(axis=1)

    # Filter by class and confidence
    mask = (class_confs >= _CONF_THRESHOLD)
    if target_class_ids:
        class_mask = np.isin(class_ids, target_class_ids)
        mask = mask & class_mask

    bboxes      = bboxes[mask]
    class_ids   = class_ids[mask]
    class_confs = class_confs[mask]

    if len(bboxes) == 0:
        return []

    # cx,cy,w,h → x1,y1,x2,y2 in 640×640 space
    x1 = bboxes[:, 0] - bboxes[:, 2] / 2
    y1 = bboxes[:, 1] - bboxes[:, 3] / 2
    x2 = bboxes[:, 0] + bboxes[:, 2] / 2
    y2 = bboxes[:, 1] + bboxes[:, 3] / 2

    # Map back to original frame: subtract padding, divide by scale
    pad_left, pad_top = pad
    x1 = np.clip((x1 - pad_left) / scale, 0, orig_w)
    y1 = np.clip((y1 - pad_top)  / scale, 0, orig_h)
    x2 = np.clip((x2 - pad_left) / scale, 0, orig_w)
    y2 = np.clip((y2 - pad_top)  / scale, 0, orig_h)

    # NMS via OpenCV
    boxes_cv = np.column_stack([x1, y1, x2 - x1, y2 - y1]).tolist()  # x,y,w,h format
    scores   = class_confs.tolist()
    idxs     = cv2.dnn.NMSBoxes(boxes_cv, scores, _CONF_THRESHOLD, _IOU_THRESHOLD)

    dets: list[Detection] = []
    for i in (idxs.flatten() if len(idxs) else []):
        dets.append(Detection(
            bbox_xyxy=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
            class_id=int(class_ids[i]),
            conf=float(class_confs[i]),
        ))
    return dets


def run(frame_bgr: np.ndarray) -> list[Detection]:
    """Run YOLO11 on one BGR frame via DML ORT session. Returns car/truck/bus detections."""
    from . import profiling
    sess = _get_session()
    inp_name = sess.get_inputs()[0].name
    h, w = frame_bgr.shape[:2]
    _t = profiling.start()
    inp_tensor, scale, pad = _preprocess(frame_bgr)
    profiling.record("yolo_pre", _t)
    _t = profiling.start()
    outputs = sess.run(None, {inp_name: inp_tensor})
    profiling.record("yolo_infer", _t)
    _t = profiling.start()
    dets = _postprocess(outputs[0], scale, pad, w, h, config.object_class_ids())
    profiling.record("yolo_post", _t)
    return dets


def active_provider() -> str:
    return _active_provider
