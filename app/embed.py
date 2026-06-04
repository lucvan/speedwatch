"""
Vehicle Re-ID image embeddings — visual grouping of cars (esp. plateless ones).

Model: OpenVINO Open Model Zoo `vehicle-reid-0001` (an OSNet from the MIT-licensed
deep-person-reid, trained on VeRi-776), exported to ONNX by bootstrap.ps1 and run on
onnxruntime. Input 1×3×208×208 RGB, output a 512-d descriptor compared by cosine distance.

Everything here is best-effort: if the ONNX model isn't present the functions no-op
(`available()` is False) and the rest of the pipeline runs unchanged — embeddings are an
optional enrichment, like the vision descriptions.

Embeddings are stored L2-normalised, so cosine similarity is just a dot product.
"""
from __future__ import annotations
import logging
import threading

import numpy as np

from . import config

log = logging.getLogger(__name__)

_INPUT_W = _INPUT_H = 208
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

_sess = None
_input_name: str | None = None
_load_failed = False
_lock = threading.Lock()


def available() -> bool:
    """True if the Re-ID model file exists and hasn't failed to load."""
    return (not _load_failed) and config.reid_model_path().exists()


def _ensure_loaded() -> bool:
    global _sess, _input_name, _load_failed
    if _sess is not None:
        return True
    if _load_failed or not config.reid_model_path().exists():
        return False
    with _lock:
        if _sess is not None:
            return True
        try:
            import onnxruntime as ort
            providers = (["DmlExecutionProvider", "CPUExecutionProvider"]
                         if config.REID_PROVIDER.lower() == "dml"
                         else ["CPUExecutionProvider"])
            _sess = ort.InferenceSession(str(config.reid_model_path()), providers=providers)
            _input_name = _sess.get_inputs()[0].name
            log.info("Re-ID embedder loaded (%s, providers=%s)",
                     config.reid_model_path().name, _sess.get_providers())
            return True
        except Exception as e:
            _load_failed = True
            log.warning("Re-ID embedder failed to load (%s) — visual grouping disabled", e)
            return False


def warmup() -> None:
    if config.reid_model_path().exists():
        _ensure_loaded()
    else:
        log.info("Re-ID model not present (%s) — visual grouping disabled until converted",
                 config.reid_model_path())


def embed_bgr(crop_bgr: "np.ndarray | None") -> "np.ndarray | None":
    """Embed a BGR car crop → L2-normalised 512-d float32 vector, or None if unavailable."""
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return None
    if not _ensure_loaded():
        return None
    try:
        import cv2
        img = cv2.resize(crop_bgr, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_AREA)
        rgb = img[:, :, ::-1]                                   # BGR → RGB
        x = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0   # CHW, [0,1]
        x = (x - _MEAN) / _STD                                  # ImageNet normalise
        out = _sess.run(None, {_input_name: x[None]})[0]        # (1, 512)
        v = np.asarray(out[0], dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v
    except Exception as e:
        log.warning("Re-ID embed failed: %s", e)
        return None


# ── Vector (de)serialisation + similarity ────────────────────────────────────

def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Assumes inputs are L2-normalised (as stored)."""
    return float(np.dot(a, b))


def cluster(items: list[tuple[int, np.ndarray]], threshold: float | None = None
            ) -> list[list[int]]:
    """
    Greedy single-pass clustering of (id, vector) by cosine similarity. Each item joins the
    most-similar existing cluster whose centroid similarity ≥ threshold, else starts a new
    cluster. Order-dependent but cheap and good enough for a review queue. Returns lists of
    ids, largest cluster first.
    """
    th = config.REID_SIM_THRESHOLD if threshold is None else threshold
    clusters: list[dict] = []   # {centroid, sum, members:[ids]}
    for pid, vec in items:
        best, best_sim = None, th
        for c in clusters:
            s = cosine(vec, c["centroid"])
            if s >= best_sim:
                best, best_sim = c, s
        if best is None:
            clusters.append({"centroid": vec.copy(), "sum": vec.copy(), "members": [pid]})
        else:
            best["members"].append(pid)
            best["sum"] = best["sum"] + vec
            c_sum = best["sum"]
            n = float(np.linalg.norm(c_sum))
            best["centroid"] = c_sum / n if n > 0 else c_sum
    clusters.sort(key=lambda c: len(c["members"]), reverse=True)
    return [c["members"] for c in clusters]


def mean_vector(vecs: list[np.ndarray]) -> "np.ndarray | None":
    """L2-normalised mean of a set of (normalised) vectors."""
    if not vecs:
        return None
    m = np.mean(np.stack(vecs), axis=0)
    n = float(np.linalg.norm(m))
    return m / n if n > 0 else m
