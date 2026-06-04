"""
Camera lens intrinsics: distortion correction for the wide-FOV stream.

Why this exists
---------------
The zone homography (calibration.py) is a *planar projective* transform — it is only
exact for an ideal pinhole camera. A wide-FOV lens adds radial (barrel) distortion, a
nonlinear pixel-space warp a homography cannot represent. The result is a pixel→metre
mapping that's accurate near the clicked calibration corners but drifts toward the frame
edges, exactly where a side-on car enters/leaves frame. That biases speed.

The fix is standard: estimate the camera's intrinsic matrix K + distortion coefficients
once (chessboard calibration), then undistort image points *before* applying the
homography. We undistort *points*, not whole frames — only the geometry math needs it, so
detection / ALPR / clips keep running on the natural (distorted) image with zero overhead.

Storage
-------
Intrinsics are camera-level and captured once. They live in
``<DATA_DIR>/intrinsics/<camera>.json``. When a zone calibration is (re)built, the
intrinsics are *baked into that calibration row* (scaled to the calibration resolution) so
the homography and the K/dist it was built with always travel together and stay consistent.

CLI (run from the install root, venv active):

    # 1. Capture chessboard views through the actual camera stream
    python -m app.intrinsics capture --count 30 --cols 9 --rows 6

    # 2. Compute intrinsics from the captured views
    python -m app.intrinsics calibrate --cols 9 --rows 6 --square 0.025

``--cols``/``--rows`` are the count of *inner* corners (a standard 10x7-square board has
9x6 inner corners). ``--square`` is the printed square size in metres (only affects the
discarded extrinsics, so it is not critical — but keep it roughly right).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from . import config

log = logging.getLogger(__name__)


def _intrinsics_dir() -> Path:
    return Path(config.DATA_DIR) / "intrinsics"


def path_for(camera: str) -> Path:
    return _intrinsics_dir() / f"{camera}.json"


# ── Load / apply ────────────────────────────────────────────────────────────────

def load(camera: str) -> dict | None:
    """
    Return the saved intrinsics for a camera, or None if it has never been lens-calibrated.
    Shape: {camera_matrix: 3x3, dist_coeffs: [...], image_w, image_h, rms, n_images, ...}.
    """
    p = path_for(camera)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        log.warning("Could not read intrinsics %s: %s", p, e)
        return None


def scale_camera_matrix(K: np.ndarray, from_wh: tuple[int, int], to_wh: tuple[int, int]) -> np.ndarray:
    """
    Rescale a camera matrix from the resolution it was estimated at to another resolution.
    fx, fy, cx, cy all scale linearly with the axis ratio; distortion coefficients are
    dimensionless (they act on K-normalised coords) and need no scaling.
    """
    fw, fh = from_wh
    tw, th = to_wh
    if fw == tw and fh == th:
        return np.array(K, dtype=np.float64)
    sx, sy = tw / fw, th / fh
    Ks = np.array(K, dtype=np.float64).copy()
    Ks[0, 0] *= sx   # fx
    Ks[0, 2] *= sx   # cx
    Ks[1, 1] *= sy   # fy
    Ks[1, 2] *= sy   # cy
    return Ks


def build_manual(camera: str, k1: float, k2: float, f_frac: float,
                 image_w: int, image_h: int) -> dict:
    """
    Assemble an intrinsics dict from manually-tuned (plumb-line) parameters instead of a
    chessboard fit. Principal point is assumed at the image centre and focal length is
    parameterised as a fraction of width (f = f_frac * image_w) — resolution-independent,
    and the absolute metric scale is set later by the zone homography, not by f. Only the
    two main radial terms (k1, k2) are tuned; tangential terms are left zero.
    """
    f = float(f_frac) * image_w
    K = [[f, 0.0, image_w / 2.0],
         [0.0, f, image_h / 2.0],
         [0.0, 0.0, 1.0]]
    return {
        "camera": camera,
        "camera_matrix": K,
        "dist_coeffs": [float(k1), float(k2), 0.0, 0.0, 0.0],
        "image_w": int(image_w),
        "image_h": int(image_h),
        "rms": None,
        "n_images": 0,
        "method": "manual",
        "f_frac": float(f_frac),
        "calibrated_at": time.time(),
    }


def save(camera: str, intr: dict) -> Path:
    """Persist an intrinsics dict for a camera. Returns the written path."""
    p = path_for(camera)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(intr, indent=2))
    return p


def undistort_points(pts_xy, K: np.ndarray, dist) -> np.ndarray:
    """
    Map distorted pixel coords → ideal (pinhole) pixel coords in the SAME pixel frame.
    ``pts_xy``: (N,2) array-like. K and dist must be expressed at the same resolution as pts.
    Returns an (N,2) float64 array.
    """
    pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    # P=K projects the undistorted normalised coords back into the original pixel frame.
    out = cv2.undistortPoints(pts, K, dist, P=K)
    return out.reshape(-1, 2).astype(np.float64)


# ── Chessboard calibration (CLI) ─────────────────────────────────────────────────

_CB_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    + cv2.CALIB_CB_NORMALIZE_IMAGE
    + cv2.CALIB_CB_FAST_CHECK
)
_SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def _find_corners(gray: np.ndarray, cols: int, rows: int):
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), _CB_FLAGS)
    if not found:
        return None
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _SUBPIX_CRIT)
    return corners


def capture(camera: str, count: int, cols: int, rows: int, interval_s: float = 1.5) -> int:
    """
    Pull frames from the live camera stream and save those that contain a detectable
    chessboard, until `count` good views are collected. Returns the number saved.

    Move the printed board around the FOV between captures — crucially out to the corners
    and edges, where barrel distortion is strongest, and tilt it at varied angles. Good
    coverage of the frame edges is what makes the distortion estimate trustworthy.
    """
    import subprocess

    out_dir = _intrinsics_dir() / f"{camera}_chessboard"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Native-resolution stream (same source as the /snapshot endpoint) so the intrinsics
    # match the resolution the zone calibration is clicked at.
    cmd = [
        config.FFMPEG_BIN, "-loglevel", "warning",
        "-timeout", "5000000", "-rtsp_transport", "tcp",
        "-fflags", "+discardcorrupt",
        "-i", config.CAMERA_RTSP,
        "-map", "0:v:0", "-vf", f"fps={1.0 / interval_s:.4f}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    # Probe resolution first.
    w, h = _probe_resolution()
    if not w:
        raise RuntimeError("Could not probe camera resolution")
    log.info("Capturing chessboard views at %dx%d — move the board across the whole frame", w, h)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=w * h * 3 * 2)
    saved = 0
    try:
        frame_bytes = w * h * 3
        while saved < count:
            raw = b""
            while len(raw) < frame_bytes:
                chunk = proc.stdout.read(frame_bytes - len(raw))
                if not chunk:
                    break
                raw += chunk
            if len(raw) < frame_bytes:
                log.warning("Stream ended after %d saved views", saved)
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if _find_corners(gray, cols, rows) is not None:
                saved += 1
                cv2.imwrite(str(out_dir / f"view_{saved:03d}.png"), frame)
                log.info("  ✓ captured view %d/%d", saved, count)
            else:
                log.info("  …no board in frame (have %d/%d)", saved, count)
    finally:
        if proc.poll() is None:
            proc.terminate()
    log.info("Saved %d chessboard views to %s", saved, out_dir)
    return saved


def calibrate(camera: str, cols: int, rows: int, square_m: float) -> dict:
    """
    Estimate K + distortion from the captured chessboard views and save the intrinsics
    JSON for `camera`. Returns the intrinsics dict.
    """
    img_dir = _intrinsics_dir() / f"{camera}_chessboard"
    images = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    if len(images) < 5:
        raise RuntimeError(
            f"Need ≥5 chessboard views, found {len(images)} in {img_dir}. "
            f"Run `python -m app.intrinsics capture` first."
        )

    # 3D object points for one board (Z=0 plane); same for every view.
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square_m)

    objpoints, imgpoints = [], []
    shape = None
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray.shape[::-1]  # (w, h)
        corners = _find_corners(gray, cols, rows)
        if corners is None:
            log.warning("  no board in %s — skipped", img_path.name)
            continue
        objpoints.append(objp)
        imgpoints.append(corners)

    if len(objpoints) < 5:
        raise RuntimeError(f"Only {len(objpoints)} usable views — recapture with the board "
                           f"clearly visible and spanning the frame edges.")

    rms, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
    w, h = shape
    intr = {
        "camera": camera,
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.reshape(-1).tolist(),
        "image_w": int(w),
        "image_h": int(h),
        "rms": float(rms),
        "n_images": len(objpoints),
        "method": "chessboard",
        "board_cols": cols,
        "board_rows": rows,
        "square_m": float(square_m),
        "calibrated_at": time.time(),
    }
    p = save(camera, intr)

    log.info("Intrinsics saved to %s", p)
    log.info("  reprojection RMS = %.3f px over %d views (aim for < 1.0)", rms, len(objpoints))
    if rms > 1.0:
        log.warning("  RMS > 1px — recapture with sharper / more varied board views for a better fit")
    return intr


def _probe_resolution() -> tuple[int, int]:
    import subprocess as sp
    try:
        r = sp.run(
            [config.FFPROBE_BIN, "-v", "error", "-rtsp_transport", "tcp",
             "-select_streams", "v:0", "-show_entries", "stream=width,height",
             "-of", "csv=p=0", config.CAMERA_RTSP],
            capture_output=True, text=True, timeout=15,
        )
        parts = r.stdout.strip().split(",")
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
    except Exception as e:
        log.warning("ffprobe failed: %s", e)
    return 0, 0


def _main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Camera lens intrinsics (chessboard calibration)")
    ap.add_argument("mode", choices=["capture", "calibrate"])
    ap.add_argument("--camera", default=config.CAMERA_NAME)
    ap.add_argument("--cols", type=int, default=9, help="inner corners across (default 9)")
    ap.add_argument("--rows", type=int, default=6, help="inner corners down (default 6)")
    ap.add_argument("--count", type=int, default=30, help="views to capture (capture mode)")
    ap.add_argument("--interval", type=float, default=1.5, help="seconds between captures")
    ap.add_argument("--square", type=float, default=0.025, help="square size in metres")
    args = ap.parse_args()

    if args.mode == "capture":
        capture(args.camera, args.count, args.cols, args.rows, args.interval)
    else:
        calibrate(args.camera, args.cols, args.rows, args.square)


if __name__ == "__main__":
    _main()
