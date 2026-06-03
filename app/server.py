"""
FastAPI web UI and REST API for speedwatch standalone.
Routes:
  GET  /              — pass list
  GET  /session/<id>  — session detail + pass labelling
  GET  /review        — low-confidence unlabelled passes
  GET  /calibrate     — calibration UI
  POST /api/calibration
  POST /api/passes/<id>/label
  GET  /snapshot/<camera>  — grab one live frame via ffmpeg
  Static: /frames/, /clips/
"""
from __future__ import annotations
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import config, db
from . import calibration as cal_mod

log = logging.getLogger(__name__)

app = FastAPI(title="Speedwatch")

_WEB_DIR      = Path(__file__).parent / "web"
_TEMPLATES_DIR = _WEB_DIR / "templates"
_STATIC_DIR   = _WEB_DIR / "static"
_FRAMES_DIR   = Path(config.DATA_DIR) / "frames"
_CLIPS_DIR    = Path(config.DATA_DIR) / "clips"

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
app.mount("/static",  StaticFiles(directory=str(_STATIC_DIR)),  name="static")
app.mount("/frames",  StaticFiles(directory=str(_FRAMES_DIR)),  name="frames")
app.mount("/clips",   StaticFiles(directory=str(_CLIPS_DIR)),   name="clips")

# Pipeline status (set by main.py)
_pipeline_status: str = "starting"

def set_pipeline_status(s: str):
    global _pipeline_status
    _pipeline_status = s


# ── Template helpers ───────────────────────────────────────────────────────────

def _fmt_ts(ts):
    if ts is None:
        return "—"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%d %b %Y %H:%M:%S")
    except Exception:
        return str(ts)

def _fmt_mph(v):
    if v is None:
        return "—"
    try:
        return f"{float(v):.1f}"
    except Exception:
        return "—"

def _confidence_class(c):
    if c is None:
        return "secondary"
    if c >= 0.7:
        return "success"
    if c >= 0.4:
        return "warning"
    return "danger"

templates.env.globals["fmt_ts"]           = _fmt_ts
templates.env.globals["fmt_mph"]          = _fmt_mph
templates.env.globals["confidence_class"] = _confidence_class
templates.env.globals["abs"]              = abs
templates.env.globals["round"]            = round


# ── Web routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, order_by: str = "created_at", label: str = "", page: int = 1):
    limit  = 50
    offset = (page - 1) * limit
    passes = await db.list_passes(limit=limit, offset=offset, label=label or None, order_by=order_by)
    total  = await db.count_passes(label=label or None)
    has_cal = (await db.get_active_calibration(config.CAMERA_NAME)) is not None
    return templates.TemplateResponse("list.html", {
        "request": request,
        "passes": passes,
        "total": total,
        "page": page,
        "limit": limit,
        "order_by": order_by,
        "label_filter": label,
        "has_calibration": has_cal,
        "pipeline_status": _pipeline_status,
    })


@app.get("/session/{session_id}", response_class=HTMLResponse)
async def session_detail(request: Request, session_id: int):
    sessions = await db.list_sessions(limit=1, offset=0)
    session  = next((s for s in await db.list_sessions(limit=1000) if s["id"] == session_id), None)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    passes = await _get_passes_for_session(session_id)
    return templates.TemplateResponse("detail.html", {
        "request": request,
        "session": session,
        "passes": passes,
    })


async def _get_passes_for_session(session_id: int) -> list[dict]:
    import aiosqlite
    from . import db as db_mod
    async with aiosqlite.connect(db_mod.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            "SELECT * FROM pass WHERE session_id=? ORDER BY sw_speed_mph DESC",
            (session_id,),
        )
        return [dict(r) for r in await cur.fetchall()]


@app.get("/calibrate", response_class=HTMLResponse)
async def calibrate_page(request: Request):
    cals = await db.list_calibrations(config.CAMERA_NAME)
    return templates.TemplateResponse("calibrate.html", {
        "request": request,
        "snapshot_url": f"/snapshot/{config.CAMERA_NAME}",
        "camera": config.CAMERA_NAME,
        "calibrations": cals,
        "queue_size": 0,
    })


@app.get("/review", response_class=HTMLResponse)
async def review_page(request: Request, page: int = 1):
    limit  = 50
    offset = (page - 1) * limit
    passes = await db.list_passes(
        limit=limit, offset=offset,
        unlabelled_only=True,
        order_by="confidence",
    )
    total  = await db.count_passes(unlabelled_only=True)
    return templates.TemplateResponse("review.html", {
        "request": request,
        "passes": passes,
        "total": total,
        "page": page,
        "limit": limit,
    })


# ── REST API ───────────────────────────────────────────────────────────────────

@app.post("/api/calibration")
async def api_save_calibration(request: Request):
    body          = await request.json()
    camera        = body.get("camera", config.CAMERA_NAME)
    zone_pixels   = body.get("zone_pixels")
    edge_distances= body.get("edge_distances_m")
    diagonal_m    = body.get("diagonal_m")
    frame_w       = int(body.get("frame_w", 1280))
    frame_h       = int(body.get("frame_h", 720))
    notes         = body.get("notes", "")

    if not zone_pixels or not edge_distances or diagonal_m is None:
        raise HTTPException(status_code=422, detail="Missing required calibration fields")

    try:
        cal = cal_mod.build_calibration(
            camera=camera,
            zone_pixels_norm=zone_pixels,
            edge_distances_m=edge_distances,
            diagonal_m=float(diagonal_m),
            frame_w=frame_w,
            frame_h=frame_h,
            notes=notes,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    cal_id = await db.save_calibration(cal)
    return JSONResponse({"ok": True, "calibration_id": cal_id,
                         "residual_max_px": cal["residual_max_px"]})


_PASS_FILE_COLS = (
    "entry_frame_path", "exit_frame_path",
    "entry_frame_raw_path", "exit_frame_raw_path",
    "trajectory_overlay_path", "hq_frame_path", "hq_exit_frame_path",
)

def _delete_data_file(rel_path: str | None):
    if not rel_path:
        return
    try:
        (Path(config.DATA_DIR) / rel_path).unlink(missing_ok=True)
    except Exception as e:
        log.warning("Could not delete %s: %s", rel_path, e)


@app.delete("/api/sessions/{session_id}")
async def api_delete_session(session_id: int):
    session, passes = await db.delete_session(session_id)
    _delete_data_file(session.get("clip_path"))
    for p in passes:
        for col in _PASS_FILE_COLS:
            _delete_data_file(p.get(col))
    return JSONResponse({"ok": True})


@app.delete("/api/passes/{pass_id}")
async def api_delete_pass(pass_id: int):
    p = await db.delete_pass(pass_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Pass not found")
    for col in _PASS_FILE_COLS:
        _delete_data_file(p.get(col))
    return JSONResponse({"ok": True})


@app.post("/api/passes/{pass_id}/label")
async def api_label_pass(
    pass_id: int,
    label: str = Form(default=""),
    notes: str = Form(default=""),
    corrected_mph: str = Form(default=""),
):
    corrected = float(corrected_mph) if corrected_mph.strip() else None
    await db.update_pass_label(pass_id, label or None, notes or None, corrected)
    return JSONResponse({"ok": True})


@app.get("/api/status")
async def api_status():
    from . import detect
    return JSONResponse({
        "pipeline": _pipeline_status,
        "provider": detect.active_provider(),
        "model": config.YOLO_MODEL,
        "camera": config.CAMERA_NAME,
    })


# ── Snapshot endpoint (live frame via ffmpeg) ──────────────────────────────────

@app.get("/snapshot/{camera}")
async def snapshot(camera: str):
    """Grab one JPEG frame from the RTSPS substream."""
    try:
        rtsp_url = config.CAMERA_RTSP
        cmd = [
            config.FFMPEG_BIN,
            "-loglevel", "error",
            "-timeout", "5000000",
            "-rtsp_transport", "tcp",
            "-i", rtsp_url,
            "-map", "0:v:0",
            "-vframes", "1",
            "-f", "image2",
            "-vcodec", "mjpeg",
            "pipe:1",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        if not stdout:
            raise HTTPException(status_code=503, detail="No frame from camera")
        return Response(content=stdout, media_type="image/jpeg")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Camera snapshot timed out")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Snapshot error: {e}")
