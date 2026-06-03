"""
Camera ingest: ffmpeg subprocess → rawvideo frames at DETECT_FPS.
Motion gate: cv2 background subtraction activates/deactivates pass sessions.

The ingest loop yields PassSession objects to the pipeline.
Each PassSession holds a list of (frame_bgr, t_ms) pairs accumulated during motion.
"""
from __future__ import annotations
import logging
import subprocess
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config
from . import profiling

log = logging.getLogger(__name__)

# Motion gate tuning
_MOG2_HISTORY  = 200
_MOG2_THRESH   = 60
_MOTION_AREA_MIN_PCT = 0.003   # 0.3% of frame must move to trigger
_GATE_OPEN_FRAMES   = 3        # consecutive motion frames to open gate
_GATE_CLOSE_FRAMES  = int(config.DETECT_FPS * 1.5)  # ~1.5s quiet to close
_PREBUFFER_FRAMES   = int(config.DETECT_FPS * 1.5)   # 1.5s pre-motion buffer

# Reconnect
_RECONNECT_DELAY_S = 3.0

# Safety cap on buffered frames per motion event (bounds memory — see config)
_MAX_SESSION_FRAMES = max(1, int(config.DETECT_FPS * config.MAX_SESSION_SECONDS))


@dataclass
class PassSession:
    frames: list[tuple[np.ndarray, float]] = field(default_factory=list)  # (frame_bgr, t_ms)
    start_wall: float = field(default_factory=time.time)


def _ffmpeg_proc(fps: float, w: int = 0, h: int = 0) -> subprocess.Popen:
    """
    Open ffmpeg subprocess reading RTSPS at target fps, outputting rawvideo BGR24 to stdout.
    If w/h are 0, let ffmpeg keep native resolution.
    """
    vf_parts = [f"fps={fps}"]
    if w and h:
        vf_parts.append(f"scale={w}:{h}")
    vf = ",".join(vf_parts)

    cmd = [config.FFMPEG_BIN, "-loglevel", "warning"]
    # Hardware decode (input option — must precede -i). Offloads H.264 decode to the GPU;
    # frames are auto-downloaded to system memory, then scale + bgr24 convert run on CPU.
    if config.FFMPEG_HWACCEL:
        cmd += ["-hwaccel", config.FFMPEG_HWACCEL]
    cmd += [
        "-timeout", "5000000",         # 5s I/O timeout (replaces deprecated -stimeout)
        "-rtsp_transport", "tcp",
        "-fflags", "+discardcorrupt",  # survive missing reference frames at stream start
        "-i", config.CAMERA_RTSP,
        "-map", "0:v:0",              # select video stream explicitly (camera has multiple audio tracks)
        "-vf", vf,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    log.info("Starting ffmpeg: %s", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10 * 1024 * 1024,    # 10 MB read buffer — needed for pipe reads to return full frames
    )


def ingest_loop(on_session_ready, stop_event=None):
    """
    Blocking loop. Call from a background thread.
    Calls on_session_ready(PassSession) for each complete pass session.
    Stops when stop_event is set (or never if None).
    """
    while stop_event is None or not stop_event.is_set():
        try:
            _run_once(on_session_ready, stop_event)
        except Exception as e:
            log.error("Ingest error (reconnecting in %.0fs): %s", _RECONNECT_DELAY_S, e)
        if stop_event is None or not stop_event.is_set():
            time.sleep(_RECONNECT_DELAY_S)


def _run_once(on_session_ready, stop_event):
    # Pipeline resolution: when scaling is configured the rawvideo output is exactly that
    # size, so we know the frame dimensions without a second connection. Otherwise probe.
    scale_w, scale_h = config.PIPELINE_WIDTH, config.PIPELINE_HEIGHT
    if scale_w and scale_h:
        frame_w, frame_h = scale_w, scale_h
    else:
        frame_w, frame_h = _probe_resolution()
        if not frame_w:
            raise RuntimeError("Could not probe RTSP resolution (camera unreachable?)")
    log.info("Pipeline resolution: %dx%d", frame_w, frame_h)

    proc = _ffmpeg_proc(fps=config.DETECT_FPS, w=scale_w, h=scale_h)
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=_MOG2_HISTORY, varThreshold=_MOG2_THRESH, detectShadows=False
    )

    t_start = time.time()

    # Ring buffer for pre-trigger frames
    prebuf: list[tuple[np.ndarray, float]] = []
    motion_count  = 0
    quiet_count   = 0
    gate_open     = False
    session: PassSession | None = None

    try:
        while stop_event is None or not stop_event.is_set():
            frame_bytes = frame_w * frame_h * 3
            # Read exactly frame_bytes — accumulate across multiple reads
            _t = profiling.start()
            raw = b""
            while len(raw) < frame_bytes:
                chunk = proc.stdout.read(frame_bytes - len(raw))
                if not chunk:
                    break
                raw += chunk
            profiling.record("pipe_read", _t)
            if len(raw) < frame_bytes:
                log.warning("ffmpeg stream ended (read %d/%d bytes)", len(raw), frame_bytes)
                break

            t_ms   = (time.time() - t_start) * 1000.0
            _t = profiling.start()
            frame  = np.frombuffer(raw, dtype=np.uint8).reshape((frame_h, frame_w, 3)).copy()
            profiling.record("frame_copy", _t)
            _t = profiling.start()
            # Motion gate runs on a downscaled frame — it only needs to know that a
            # fraction of the scene moved, not full 1080p detail.
            if config.MOTION_SCALE and config.MOTION_SCALE < 1.0:
                small = cv2.resize(frame, None, fx=config.MOTION_SCALE, fy=config.MOTION_SCALE,
                                   interpolation=cv2.INTER_AREA)
            else:
                small = frame
            fgmask = fgbg.apply(small)
            motion_px = int(np.count_nonzero(fgmask))
            sh, sw = small.shape[:2]
            profiling.record("mog2", _t)
            is_moving = motion_px > sw * sh * _MOTION_AREA_MIN_PCT

            if is_moving:
                motion_count += 1
                quiet_count   = 0
            else:
                quiet_count  += 1
                motion_count  = 0

            # Maintain pre-trigger ring buffer
            prebuf.append((frame, t_ms))
            if len(prebuf) > _PREBUFFER_FRAMES:
                prebuf.pop(0)

            profiling.maybe_flush()

            if not gate_open and motion_count >= _GATE_OPEN_FRAMES:
                gate_open = True
                session   = PassSession()
                session.frames.extend(prebuf)
                log.info("Motion gate OPENED (motion_px=%d)", motion_px)

            if gate_open:
                session.frames.append((frame, t_ms))

                # Force-close on quiet OR when the buffer hits the safety cap
                capped = len(session.frames) >= _MAX_SESSION_FRAMES
                if quiet_count >= _GATE_CLOSE_FRAMES or capped:
                    gate_open    = False
                    motion_count = 0
                    quiet_count  = 0
                    why = "CAP HIT" if capped else "quiet"
                    log.info("Motion gate CLOSED (%s, %d frames in session)", why, len(session.frames))
                    if len(session.frames) >= 5:
                        on_session_ready(session)
                    session = None
                    prebuf.clear()

    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
        # If gate was open when we stopped, flush partial session
        if gate_open and session and len(session.frames) >= 5:
            log.info("Flushing partial session on disconnect (%d frames)", len(session.frames))
            on_session_ready(session)


def _probe_resolution() -> tuple[int, int]:
    """Use ffprobe to get stream width/height. Returns (w, h) or (0, 0) on failure."""
    try:
        import subprocess as sp
        r = sp.run(
            [
                config.FFPROBE_BIN,
                "-v", "error",
                "-rtsp_transport", "tcp",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                config.CAMERA_RTSP,
            ],
            capture_output=True, text=True, timeout=15
        )
        parts = r.stdout.strip().split(",")
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
    except Exception as e:
        log.warning("ffprobe failed: %s", e)
    return 0, 0
