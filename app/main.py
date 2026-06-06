"""
Speedwatch standalone entry point.
Runs: ingest loop (background thread) + FastAPI server (uvicorn, port 8765).
"""
from __future__ import annotations
import asyncio
import contextlib
import logging
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from . import config, db
from . import profiling
from . import detect as detect_mod
from . import speed as speed_mod
from . import clip as clip_mod
from . import alpr as alpr_mod
from . import embed as embed_mod
from . import vehicles as vehicles_mod
from .ingest import PassSession, ingest_loop
from .server import app, set_pipeline_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(config.DATA_DIR) / "logs" / "speedwatch.log"),
    ],
)
log = logging.getLogger(__name__)

_stop_event = threading.Event()


def _process_session(session: PassSession):
    """Called from the ingest thread for each completed pass session."""
    t0 = time.time()
    frames = session.frames
    if not frames:
        return

    # Resolve calibration synchronously (event loop runs separately)
    cal = asyncio.run_coroutine_threadsafe(
        db.get_active_calibration(config.CAMERA_NAME),
        _main_loop,
    ).result(timeout=5)

    if cal is None:
        log.warning("No active calibration — session dropped (set one up at /calibrate)")
        set_pipeline_status("idle (no calibration)")
        return

    log.info("Processing motion event (%d frames)", len(frames))
    set_pipeline_status("processing motion event")

    # Run YOLO + tracking over the buffered frames BEFORE persisting anything.
    # Detection/tracking is strided down to ~DETECT_INFER_FPS — far cheaper than the
    # 30fps capture rate and plenty for tracking/speed. Clips still use ALL frames (30fps).
    fps = config.DETECT_FPS
    infer_fps = config.DETECT_INFER_FPS or fps
    stride = max(1, round(fps / infer_fps))
    eff_fps = fps / stride
    h, w = frames[0][0].shape[:2]
    tracker = speed_mod.LiveTracker(cal, eff_fps)
    t_session_start = frames[0][1]

    for i, (frame_bgr, t_ms_rel) in enumerate(frames):
        if stride > 1 and (i % stride):
            continue
        detections = detect_mod.run(frame_bgr)
        _t = profiling.start()
        tracker.update(detections, frame_bgr, t_ms_rel - t_session_start)
        profiling.record("track", _t)

    passes = tracker.finalize()

    # Drop empty events entirely — no clip, no DB row, no stills.
    if not passes:
        set_pipeline_status("idle — last motion: no vehicle")
        log.info("No qualifying pass — motion event discarded (%d frames)", len(frames))
        return

    # We have at least one pass — now create the session row and persist.
    session_id = asyncio.run_coroutine_threadsafe(
        db.create_session(config.CAMERA_NAME, session.start_wall, cal.get("id")),
        _main_loop,
    ).result(timeout=5)
    log.info("Session %d: %d qualifying pass(es)", session_id, len(passes))

    # Encode clip from buffered frames (clean — speed label only, no zone overlay)
    _t = profiling.start()
    clip_path = clip_mod.encode_clip(frames, str(session_id), passes[0], fps)
    profiling.record("clip_encode", _t)

    for result in passes:
        _t = profiling.start()
        stills = clip_mod.save_pass_stills(result, str(session_id))
        profiling.record("stills", _t)

        # Number-plate recognition: read the plate inside the car box on both the entry
        # and exit raw frames; the higher-confidence read wins (the plate-visible end
        # depends on travel direction). Remember the winning frame/bbox for the car crop.
        plate = plate_conf = None
        best_frame, best_bbox = None, None
        _t = profiling.start()
        for frame_raw, bbox in (
            (result.entry_frame_raw, result.entry_bbox_xyxy),
            (result.exit_frame_raw,  result.exit_bbox_xyxy),
        ):
            hit = alpr_mod.read_plate(frame_raw, bbox)
            if hit and (plate_conf is None or hit[1] > plate_conf):
                plate, plate_conf = hit
                best_frame, best_bbox = frame_raw, bbox
        profiling.record("alpr", _t)
        if plate:
            log.info("track %d: plate %s (conf %.2f)", result.track_id, plate, plate_conf)

        # Car crop (from the plate-winning frame, else the entry frame) — used as the
        # vehicle's representative image and the vision-description input.
        crop_frame = best_frame if best_frame is not None else result.entry_frame_raw
        crop_bbox  = best_bbox  if best_bbox  is not None else result.entry_bbox_xyxy
        car_crop_path = clip_mod.save_car_crop(crop_frame, crop_bbox, str(session_id), result.track_id)

        # Vehicle Re-ID embedding of the car crop (for visual grouping of look-alikes /
        # plateless cars). No-ops if the Re-ID model isn't installed.
        emb_blob = None
        if crop_frame is not None and crop_bbox is not None:
            _t = profiling.start()
            x1, y1, x2, y2 = (int(v) for v in crop_bbox)
            car_img = crop_frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            vec = embed_mod.embed_bgr(car_img)
            if vec is not None:
                emb_blob = embed_mod.to_blob(vec)
            profiling.record("embed", _t)

        pass_row = {
            "session_id":              session_id,
            "track_id":                result.track_id,
            "sw_speed_mph":            result.sw_speed_mph,
            "sw_speed_mps":            result.sw_speed_mps,
            "frames_in_zone":          result.frames_in_zone,
            "dwell_seconds":           result.dwell_seconds,
            "median_heading_deg":      result.median_heading_deg,
            "per_frame_speed_cv":      result.per_frame_speed_cv,
            "id_switches":             result.id_switches,
            "bbox_area_cv":            result.bbox_area_cv,
            "net_displacement_m":      result.net_displacement_m,
            "confidence":              result.confidence,
            "clip_path":               clip_path,
            "plate":                   plate,
            "plate_confidence":        plate_conf,
            "car_crop_path":           car_crop_path,
            "embedding":               emb_blob,
            **stills,
        }
        asyncio.run_coroutine_threadsafe(
            db.insert_pass(pass_row), _main_loop
        ).result(timeout=10)

        # Vehicle upkeep (only for plated passes): register the vehicle, keep the best
        # car crop as its representative image, and auto-describe once it's a regular.
        # Fire-and-forget on the event loop so the ~9s vision call never blocks ingest.
        if plate:
            async def _veh_upkeep(raw=plate, img=car_crop_path, pc=plate_conf or 0.0):
                try:
                    pl = await db.canonical_plate(raw)   # attach to merged vehicle if aliased
                    await db.ensure_vehicle(pl)
                    if img:
                        await db.update_vehicle_rep_image(pl, img, pc)
                    await vehicles_mod.maybe_describe(pl)
                except Exception as e:
                    log.warning("vehicle upkeep failed for %s: %s", raw, e)
            asyncio.run_coroutine_threadsafe(_veh_upkeep(), _main_loop)

    asyncio.run_coroutine_threadsafe(
        db.finalize_session(session_id, time.time(), len(frames), "done", clip_path),
        _main_loop,
    ).result(timeout=5)

    elapsed = time.time() - t0
    set_pipeline_status(f"idle — last session {session_id} ({len(passes)} pass{'es' if len(passes) != 1 else ''})")
    log.info("Session %d done in %.1fs", session_id, elapsed)


def _ingest_thread():
    set_pipeline_status("connecting to camera…")
    detect_mod.warmup()
    log.info("Detection warmup done — provider: %s", detect_mod.active_provider())
    alpr_mod.warmup()
    embed_mod.warmup()
    set_pipeline_status("idle (waiting for motion)")
    ingest_loop(_process_session, stop_event=_stop_event)


async def _startup():
    await db.init_db()
    Path(config.DATA_DIR, "frames").mkdir(parents=True, exist_ok=True)
    Path(config.DATA_DIR, "clips").mkdir(parents=True, exist_ok=True)
    Path(config.DATA_DIR, "logs").mkdir(parents=True, exist_ok=True)
    t = threading.Thread(target=_ingest_thread, daemon=True, name="ingest")
    t.start()
    log.info("Speedwatch standalone started on port %d", config.PORT)


@contextlib.asynccontextmanager
async def _lifespan(application: FastAPI):
    global _main_loop
    _main_loop = asyncio.get_event_loop()
    async with contextlib.AsyncExitStack() as stack:
        # The MCP server is mounted as a sub-app; Starlette does not run a mounted app's own
        # lifespan, so we run its streamable-HTTP session manager from the parent lifespan.
        if config.MCP_ENABLED:
            from . import mcp_server as _mcp
            _mcp.streamable_app()  # ensure the session manager is created
            await stack.enter_async_context(_mcp.session_manager_lifespan())
            log.info("MCP server mounted at /mcp")
        await _startup()
        yield


app.router.lifespan_context = _lifespan


def main():
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        log_level="warning",
        reload=False,
    )


if __name__ == "__main__":
    main()
