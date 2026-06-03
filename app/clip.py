"""
Encode an mp4 clip + save entry/exit/trajectory stills from a PassResult.
The clip is re-encoded from the buffered pipeline frames, and all stills come from
those same frames — so clip and stills are inherently time-aligned at the pipeline
resolution (no separate HQ recording or seek-matching).
"""
from __future__ import annotations
import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import config
from .speed import PassResult

log = logging.getLogger(__name__)


def save_pass_stills(result: PassResult, session_id: str) -> dict[str, str | None]:
    frames_dir = Path(config.DATA_DIR) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{session_id}_{result.track_id}"

    def _save(arr: np.ndarray | None, name: str) -> str | None:
        if arr is None:
            return None
        path = frames_dir / name
        cv2.imwrite(str(path), arr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return f"frames/{name}"

    return {
        "entry_frame_path":        _save(result.entry_frame,        f"{prefix}_entry.jpg"),
        "exit_frame_path":         _save(result.exit_frame,         f"{prefix}_exit.jpg"),
        "entry_frame_raw_path":    _save(result.entry_frame_raw,    f"{prefix}_entry_raw.jpg"),
        "exit_frame_raw_path":     _save(result.exit_frame_raw,     f"{prefix}_exit_raw.jpg"),
        "trajectory_overlay_path": _save(result.trajectory_frame,   f"{prefix}_traj.jpg"),
    }


def encode_clip(
    frames: list[tuple[np.ndarray, float]],
    session_id: str,
    result: PassResult | None,
    fps: float,
) -> str | None:
    """
    Re-encode an annotated mp4 clip from the buffered pipeline frames.
    Returns relative path 'clips/<name>.mp4' or None on failure.
    """
    if not frames:
        return None

    clips_dir = Path(config.DATA_DIR) / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    out_path = clips_dir / f"{session_id}.mp4"

    h, w = frames[0][0].shape[:2]
    speed_mph = result.sw_speed_mph if result else None

    cmd = [
        config.FFMPEG_BIN, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",   # required for browser H.264 compatibility
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for frame_bgr, _ in frames:
            # Clean clip — no zone overlay. Keep only the speed readout.
            annotated = frame_bgr
            if speed_mph is not None:
                annotated = frame_bgr.copy()
                cv2.putText(
                    annotated,
                    f"{speed_mph:.1f} mph",
                    (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 128), 2,
                )
            proc.stdin.write(annotated.tobytes())
        proc.stdin.close()
        proc.wait(timeout=120)
    except Exception as e:
        log.warning("ffmpeg encode error: %s", e)
        try:
            proc.kill()
        except Exception:
            pass
        return None

    if not out_path.exists() or out_path.stat().st_size == 0:
        log.warning("ffmpeg produced empty clip for session %s", session_id)
        return None

    size_kb = out_path.stat().st_size // 1024
    log.info("Clip saved: %s (%d KB)", out_path.name, size_kb)
    return f"clips/{out_path.name}"
