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
import csv
import io
import logging
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import config, db
from . import calibration as cal_mod
from . import intrinsics as intr_mod
from . import embed as embed_mod
from . import vehicles as vehicles_mod
from . import anpr_mask as anpr_mask_mod
from . import stats as stats_mod
from . import report as report_mod

log = logging.getLogger(__name__)

app = FastAPI(title="Speedwatch")

_WEB_DIR      = Path(__file__).parent / "web"
_TEMPLATES_DIR = _WEB_DIR / "templates"
_STATIC_DIR   = _WEB_DIR / "static"
_FRAMES_DIR   = Path(config.DATA_DIR) / "frames"
_CLIPS_DIR    = Path(config.DATA_DIR) / "clips"
_EXPORTS_DIR  = Path(config.DATA_DIR) / "exports"
for _d in (_FRAMES_DIR, _CLIPS_DIR, _EXPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
app.mount("/static",  StaticFiles(directory=str(_STATIC_DIR)),  name="static")
app.mount("/frames",  StaticFiles(directory=str(_FRAMES_DIR)),  name="frames")
app.mount("/clips",   StaticFiles(directory=str(_CLIPS_DIR)),   name="clips")
app.mount("/exports", StaticFiles(directory=str(_EXPORTS_DIR)), name="exports")

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


def _speed_band(v):
    """UI band for a speed: 'ok' (green) / 'warn' (amber) / 'over' (red) / '' (unknown)."""
    if v is None:
        return ""
    try:
        v = float(v)
    except Exception:
        return ""
    if v > config.SPEED_LIMIT_MPH:
        return "over"
    if v >= config.SPEED_WARN_MPH:
        return "warn"
    return "ok"


def _fmt_ago(ts):
    """Compact relative time, e.g. 'just now', '12 min', '3 h', '2 d'."""
    if ts is None:
        return "—"
    try:
        delta = time.time() - float(ts)
    except Exception:
        return "—"
    if delta < 0:
        delta = 0
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)} min"
    if delta < 86400:
        return f"{int(delta // 3600)} h"
    if delta < 7 * 86400:
        return f"{int(delta // 86400)} d"
    return _fmt_ts(ts).split(" ")[0]


def _day_label(ts):
    """'Today' / 'Yesterday' / 'DD Mon YYYY' for grouping rows by calendar day."""
    if ts is None:
        return "—"
    try:
        d = datetime.fromtimestamp(float(ts)).date()
    except Exception:
        return "—"
    today = datetime.now().date()
    delta = (today - d).days
    if delta == 0:
        return "Today"
    if delta == 1:
        return "Yesterday"
    return d.strftime("%d %b %Y")


templates.env.globals["fmt_ts"]           = _fmt_ts
templates.env.globals["fmt_mph"]          = _fmt_mph
templates.env.globals["fmt_ago"]          = _fmt_ago
templates.env.globals["day_label"]        = _day_label
templates.env.globals["confidence_class"] = _confidence_class
templates.env.globals["speed_band"]       = _speed_band
templates.env.globals["abs"]              = abs
templates.env.globals["round"]            = round
templates.env.globals["SPEED_LIMIT_MPH"]  = config.SPEED_LIMIT_MPH
templates.env.globals["SPEED_WARN_MPH"]   = config.SPEED_WARN_MPH
templates.env.globals["APP_VERSION"]      = config.APP_VERSION


# ── Web routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    stats = await db.dashboard_stats(midnight, config.SPEED_WARN_MPH, config.SPEED_LIMIT_MPH)
    recent = await db.recent_fast_passes(limit=8)
    leaders = await db.list_vehicles(order_by="max_mph", min_passes=1)
    leaders = leaders[:8]
    has_cal = (await db.get_active_calibration(config.CAMERA_NAME)) is not None
    road_speeds = await db.all_pass_speeds()
    speed_summary = stats_mod.speed_summary(road_speeds, config.SPEED_WARN_MPH, config.SPEED_LIMIT_MPH)
    markers = []
    if speed_summary:
        markers.append((speed_summary["p85"], "#4fc3f7", f"85th {speed_summary['p85']:g}"))
    dist_svg = stats_mod.speed_distribution_svg(
        road_speeds, warn=config.SPEED_WARN_MPH, limit=config.SPEED_LIMIT_MPH, markers=markers)
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats": stats,
        "recent": recent,
        "leaders": leaders,
        "has_calibration": has_cal,
        "pipeline_status": _pipeline_status,
        "speed_summary": speed_summary,
        "dist_svg": dist_svg,
    })


@app.get("/passes", response_class=HTMLResponse)
async def passes_list(request: Request, order_by: str = "created_at", label: str = "",
                      page: int = 1, min_mph: float = 0, plateless: int = 0):
    limit  = 50
    offset = (page - 1) * limit
    min_mph_f = min_mph if min_mph and min_mph > 0 else None
    plateless_b = bool(plateless)
    passes = await db.list_passes(limit=limit, offset=offset, label=label or None,
                                  order_by=order_by, min_mph=min_mph_f, plateless=plateless_b)
    total  = await db.count_passes(label=label or None, min_mph=min_mph_f, plateless=plateless_b)
    has_cal = (await db.get_active_calibration(config.CAMERA_NAME)) is not None
    return templates.TemplateResponse("list.html", {
        "request": request,
        "passes": passes,
        "total": total,
        "page": page,
        "limit": limit,
        "order_by": order_by,
        "label_filter": label,
        "min_mph": min_mph_f or 0,
        "plateless": 1 if plateless_b else 0,
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
    known_plates = [r["plate"] for r in await db.list_distinct_plates()]
    reid_available = embed_mod.available()
    if reid_available:
        for p in passes:
            p["similar"] = await vehicles_mod.similar_vehicles_for_pass(p["id"])
    return templates.TemplateResponse("detail.html", {
        "request": request,
        "session": session,
        "passes": passes,
        "known_plates": known_plates,
        "reid_available": reid_available,
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
    intrinsics = intr_mod.load(config.CAMERA_NAME)
    try:
        static_v = int((_STATIC_DIR / "calibrate.js").stat().st_mtime)
    except OSError:
        static_v = 0
    return templates.TemplateResponse("calibrate.html", {
        "request": request,
        "snapshot_url": f"/snapshot/{config.CAMERA_NAME}",
        "camera": config.CAMERA_NAME,
        "calibrations": cals,
        "intrinsics": intrinsics,
        "anpr_mask": anpr_mask_mod.load(config.CAMERA_NAME),
        "static_v": static_v,
        "queue_size": 0,
    })


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Read-only view of the running configuration (edited via .env + service restart)."""
    from . import detect
    intr = intr_mod.load(config.CAMERA_NAME)
    active_cal = await db.get_active_calibration(config.CAMERA_NAME)
    groups = [
        ("Camera & pipeline", [
            ("Camera name", config.CAMERA_NAME),
            ("Pipeline resolution", f"{config.PIPELINE_WIDTH}×{config.PIPELINE_HEIGHT}"),
            ("Detect FPS / infer FPS", f"{config.DETECT_FPS:g} / {config.DETECT_INFER_FPS:g}"),
            ("Max session seconds", f"{config.MAX_SESSION_SECONDS:g}"),
            ("ffmpeg hwaccel", config.FFMPEG_HWACCEL or "(software)"),
        ]),
        ("Speed", [
            ("Legal limit (red)", f"{config.SPEED_LIMIT_MPH:g} mph"),
            ("Warn threshold (amber)", f"{config.SPEED_WARN_MPH:g} mph"),
            ("Min dt for a valid pass", f"{config.MIN_DT_SECONDS:g} s"),
            ("Active zone calibration", "yes" if active_cal else "none"),
            ("Lens intrinsics", "calibrated" if intr else "none"),
        ]),
        ("Detection & inference", [
            ("YOLO model", config.YOLO_MODEL),
            ("ONNX provider", detect.active_provider()),
            ("Object classes", config.OBJECT_CLASSES),
        ]),
        ("ANPR", [
            ("Enabled", "yes" if config.ENABLE_ALPR else "no"),
            ("Detector model", config.ALPR_DETECTOR_MODEL),
            ("OCR model", config.ALPR_OCR_MODEL),
            ("Min OCR confidence", f"{config.ALPR_MIN_CONF:g}"),
        ]),
        ("Vehicle enrichment", [
            ("Vision model", config.VEHICLE_VISION_MODEL),
            ("Auto-describe", "yes" if config.VEHICLE_AUTODESCRIBE else "no"),
            ("Describe after N passes", str(config.VEHICLE_MIN_PASSES)),
            ("Re-ID model", config.REID_MODEL),
            ("Re-ID available", "yes" if embed_mod.available() else "no (model not installed)"),
            ("Re-ID same-car threshold", f"{config.REID_SIM_THRESHOLD:g}"),
        ]),
        ("Server", [
            ("Bind", f"{config.HOST}:{config.PORT}"),
            ("Data dir", config.DATA_DIR),
            ("Version", config.APP_VERSION),
        ]),
    ]
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "groups": groups,
        "pipeline_status": _pipeline_status,
    })


@app.get("/vehicles", response_class=HTMLResponse)
async def vehicles_page(request: Request, order_by: str = "max_mph", min_passes: int = 1,
                        confirmed: int = 0):
    vehicles = await db.list_vehicles(order_by=order_by, min_passes=min_passes)
    if confirmed:
        vehicles = [v for v in vehicles if v.get("confirmed_speeder")]
    return templates.TemplateResponse("vehicles.html", {
        "request": request,
        "vehicles": vehicles,
        "order_by": order_by,
        "min_passes": min_passes,
        "confirmed": 1 if confirmed else 0,
        "describe_threshold": config.VEHICLE_MIN_PASSES,
    })


@app.get("/vehicle/{plate}", response_class=HTMLResponse)
async def vehicle_detail(request: Request, plate: str):
    # If this plate has been merged into another, show the canonical page.
    canon = await db.canonical_plate(plate)
    if canon != plate:
        return RedirectResponse(url=f"/vehicle/{canon}", status_code=303)
    vehicle = await db.get_vehicle(plate)
    if vehicle is None:
        raise HTTPException(status_code=404, detail="No passes recorded for that plate")
    passes = await db.list_passes_for_plate(plate)
    aliases = await db.aliases_of(plate)
    candidates = await db.list_distinct_plates()
    target_desc = vehicle.get("user_description") or vehicle.get("description")
    suggestions = vehicles_mod.similar_plates(plate, candidates, target_desc=target_desc)

    # Visually-similar vehicles (Re-ID) — appearance corroboration / merge candidates.
    visual_raw = await vehicles_mod.visual_similar_vehicles(plate)
    visual = []
    if visual_raw:
        repmap = {v["plate"]: v["rep_image_path"] for v in await db.list_vehicles(min_passes=1)}
        visual = [{"plate": pl, "sim": s, "rep_image": repmap.get(pl),
                   "strong": s >= config.REID_SIM_THRESHOLD} for pl, s in visual_raw]

    # Speed distribution: this vehicle's passes overlaid on the whole road, + percentile rank.
    veh_speeds = [p.get("sw_speed_mph") for p in passes]
    road_speeds = await db.all_pass_speeds()
    veh_summary = stats_mod.speed_summary(veh_speeds, config.SPEED_WARN_MPH, config.SPEED_LIMIT_MPH)
    road_sorted = sorted([s for s in road_speeds if s is not None])
    median_rank = (stats_mod.percentile_rank(road_sorted, veh_summary["median"])
                   if veh_summary else None)
    dist_svg = stats_mod.speed_distribution_svg(
        road_speeds, highlight=veh_speeds,
        warn=config.SPEED_WARN_MPH, limit=config.SPEED_LIMIT_MPH,
        highlight_label="median")  # the violet curve is labelled in the card legend
    # Per-pass percentile rank vs the whole road (for the pass table).
    pass_ranks = {p["id"]: stats_mod.percentile_rank(road_sorted, p.get("sw_speed_mph"))
                  for p in passes}

    return templates.TemplateResponse("vehicle.html", {
        "request": request,
        "vehicle": vehicle,
        "passes": passes,
        "aliases": aliases,
        "suggestions": suggestions,
        "visual": visual,
        "reid_available": embed_mod.available(),
        "veh_summary": veh_summary,
        "median_rank": median_rank,
        "dist_svg": dist_svg,
        "pass_ranks": pass_ranks,
    })


@app.get("/unidentified", response_class=HTMLResponse)
async def unidentified_page(request: Request):
    clusters = await vehicles_mod.unidentified_clusters()
    known_plates = [r["plate"] for r in await db.list_distinct_plates()]
    needing = len(await db.passes_needing_embedding())
    return templates.TemplateResponse("unidentified.html", {
        "request": request,
        "clusters": clusters,
        "known_plates": known_plates,
        "embed_available": embed_mod.available(),
        "needing_embedding": needing,
    })


@app.get("/evidence", response_class=HTMLResponse)
async def evidence_page(request: Request):
    passes = await db.list_evidence_passes()
    return templates.TemplateResponse("evidence.html", {
        "request": request,
        "passes": passes,
        "total": len(passes),
        "evidence_floor": config.EVIDENCE_MIN_MPH,
        "council": config.REPORT_COUNCIL,
    })


@app.get("/evidence/report", response_class=HTMLResponse)
async def evidence_report(request: Request, refresh: int = 0):
    """Print-ready council evidence report: road-wide speed profile + per-offender profiles +
    an AI-assisted narrative. Save-as-PDF from the browser. The narrative is cached on a data
    signature; ?refresh=1 forces it to be regenerated."""
    ctx = await report_mod.build_report_context(refresh=bool(refresh))
    return templates.TemplateResponse("report.html", {"request": request, **ctx})


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

    # Bake in lens intrinsics if this camera has been lens-calibrated (distortion correction).
    intrinsics = intr_mod.load(camera)

    try:
        cal = cal_mod.build_calibration(
            camera=camera,
            zone_pixels_norm=zone_pixels,
            edge_distances_m=edge_distances,
            diagonal_m=float(diagonal_m),
            frame_w=frame_w,
            frame_h=frame_h,
            notes=notes,
            intrinsics=intrinsics,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    cal_id = await db.save_calibration(cal)
    return JSONResponse({"ok": True, "calibration_id": cal_id,
                         "residual_max_px": cal["residual_max_px"],
                         "lens_corrected": intrinsics is not None})


@app.post("/api/calibration/reapply-lens")
async def api_reapply_lens(request: Request):
    """
    Rebuild the active zone calibration from its stored corners + measurements, baking in
    the current lens intrinsics. No re-clicking needed — the clicked corners are unchanged,
    so this is identical to re-saving the same calibration with lens correction on.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    camera = body.get("camera", config.CAMERA_NAME)

    active = await db.get_active_calibration(camera)
    if active is None:
        raise HTTPException(status_code=404, detail="No active zone calibration to re-apply to")
    intrinsics = intr_mod.load(camera)
    if intrinsics is None:
        raise HTTPException(status_code=422, detail="No lens intrinsics saved for this camera")

    try:
        cal = cal_mod.build_calibration(
            camera=camera,
            zone_pixels_norm=active["zone_pixels"],
            edge_distances_m=active["edge_distances_m"],
            diagonal_m=active["diagonal_m"],
            frame_w=active["frame_w"],
            frame_h=active["frame_h"],
            notes=active.get("notes") or "",
            intrinsics=intrinsics,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    cal_id = await db.save_calibration(cal)
    return JSONResponse({"ok": True, "calibration_id": cal_id,
                         "residual_max_px": cal["residual_max_px"], "lens_corrected": True})


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


@app.post("/api/passes/{pass_id}/confirm")
async def api_confirm_pass(pass_id: int, request: Request):
    """Flag/unflag a single pass as confirmed-speeder evidence. Body: {confirmed: bool}."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    confirmed = bool(body.get("confirmed", True))
    await db.set_pass_confirmed(pass_id, confirmed)
    return JSONResponse({"ok": True, "confirmed": confirmed})


@app.post("/api/passes/{pass_id}/assign")
async def api_assign_pass(pass_id: int, request: Request):
    """Attach a pass to a vehicle by plate (for plateless or mis-read passes the user can
    visually confirm). Body: {plate: str}. An empty plate clears the assignment."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    plate = (body.get("plate") or "").strip().upper()
    if not plate:
        await db.unassign_pass(pass_id)
        return JSONResponse({"ok": True, "assigned": None})
    canon = await db.assign_pass_to_vehicle(pass_id, plate)
    return JSONResponse({"ok": True, "assigned": canon})


@app.post("/api/passes/assign-bulk")
async def api_assign_bulk(request: Request):
    """Assign many passes to one vehicle at once. Body: {pass_ids: [int], plate: str}.
    Used by the Unidentified review queue to promote a cluster to a (new or existing) vehicle."""
    body = await request.json()
    plate = (body.get("plate") or "").strip().upper()
    pass_ids = body.get("pass_ids") or []
    if not plate or not pass_ids:
        raise HTTPException(status_code=422, detail="plate and pass_ids required")
    canon = None
    for pid in pass_ids:
        canon = await db.assign_pass_to_vehicle(int(pid), plate)
    return JSONResponse({"ok": True, "plate": canon, "count": len(pass_ids)})


@app.post("/api/passes/group-unidentified")
async def api_group_unidentified(request: Request):
    """Group plateless passes into a new no-plate vehicle record. Body: {pass_ids: [int]}.
    Allocates a synthetic id (UNK-NNNN) and assigns the passes to it, so visually-identical
    cars with no readable plate still get a single vehicle record. Rename later via merge /
    recompute-plate once a plate is found."""
    body = await request.json()
    pass_ids = body.get("pass_ids") or []
    if not pass_ids:
        raise HTTPException(status_code=422, detail="pass_ids required")
    plate = await db.next_synthetic_plate()
    for pid in pass_ids:
        await db.assign_pass_to_vehicle(int(pid), plate)
    return JSONResponse({"ok": True, "plate": plate, "count": len(pass_ids)})


@app.post("/api/embeddings/backfill")
async def api_embeddings_backfill():
    """Compute Re-ID embeddings for existing passes that have a crop but no embedding."""
    return JSONResponse(await vehicles_mod.backfill_embeddings())


@app.post("/api/vehicle/{plate}/confirm")
async def api_confirm_vehicle(plate: str, request: Request):
    """Flag/unflag a whole vehicle as a confirmed speeder. Body: {confirmed: bool}.
    Every pass of the (canonical) vehicle then counts as evidence."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    confirmed = bool(body.get("confirmed", True))
    canon = await db.canonical_plate(plate)
    await db.set_vehicle_confirmed(canon, confirmed)
    return JSONResponse({"ok": True, "confirmed": confirmed, "plate": canon})


# ── Confirmed-speeder evidence export ──────────────────────────────────────────

_EVIDENCE_COLS = [
    ("evidence_no", "Evidence #"),
    ("datetime", "Date / time"),
    ("camera", "Camera"),
    ("plate", "Plate"),
    ("measured_mph", "Measured speed (mph)"),
    ("corrected_mph", "Reviewed speed (mph)"),
    ("confidence_pct", "System confidence (%)"),
    ("dwell_seconds", "Time in zone (s)"),
    ("frames_in_zone", "Frames measured"),
    ("notes", "Reviewer notes"),
    ("clip_file", "Video file"),
    ("entry_image", "Entry still"),
    ("exit_image", "Exit still"),
]


def _evidence_rows(passes: list[dict]) -> list[dict]:
    """Flatten DB rows into manifest-friendly dicts (shared by CSV and ZIP)."""
    rows = []
    for i, p in enumerate(passes, start=1):
        try:
            dt = datetime.fromtimestamp(float(p["start_ts"])).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            dt = str(p.get("start_ts"))
        rows.append({
            "evidence_no": i,
            "datetime": dt,
            "camera": p.get("camera") or "",
            "plate": p.get("canonical_plate") or p.get("plate") or "(no plate read)",
            "measured_mph": f"{p['sw_speed_mph']:.1f}" if p.get("sw_speed_mph") is not None else "",
            "corrected_mph": f"{p['user_corrected_mph']:.1f}" if p.get("user_corrected_mph") else "",
            "confidence_pct": f"{(p.get('confidence') or 0) * 100:.0f}",
            "dwell_seconds": f"{p.get('dwell_seconds') or 0:.2f}",
            "frames_in_zone": p.get("frames_in_zone") or 0,
            "notes": p.get("user_notes") or "",
            "_clip": p.get("clip_path") or p.get("session_clip_path"),
            "_entry": p.get("entry_frame_raw_path") or p.get("entry_frame_path"),
            "_exit": p.get("exit_frame_raw_path") or p.get("exit_frame_path"),
            "_traj": p.get("trajectory_overlay_path"),
            "_car": p.get("car_crop_path"),
            "_pass_id": p.get("id"),
        })
    return rows


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "", str(s)) or "x"


def _build_evidence_zip(rows: list[dict]) -> bytes:
    """Build an evidence ZIP: one folder per pass (clip + stills) + a manifest CSV."""
    data_dir = Path(config.DATA_DIR)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        man = io.StringIO()
        w = csv.writer(man)
        w.writerow([label for _, label in _EVIDENCE_COLS])
        for r in rows:
            ts = r["datetime"].replace(":", "").replace("-", "").replace(" ", "_")
            folder = f"{r['evidence_no']:03d}_{ts}_{_safe(r['plate'])}"
            clip_file = entry_img = exit_img = ""
            attachments = [("_clip", "clip"), ("_entry", "entry"),
                           ("_exit", "exit"), ("_traj", "trajectory"), ("_car", "car")]
            for key, label in attachments:
                rel = r.get(key)
                if not rel:
                    continue
                src = data_dir / rel
                if not src.exists():
                    continue
                arc = f"{folder}/{label}{src.suffix}"
                z.write(src, arc)
                if key == "_clip":
                    clip_file = arc
                elif key == "_entry":
                    entry_img = arc
                elif key == "_exit":
                    exit_img = arc
            row_out = {**r, "clip_file": clip_file, "entry_image": entry_img, "exit_image": exit_img}
            w.writerow([row_out.get(k, "") for k, _ in _EVIDENCE_COLS])
        z.writestr("manifest.csv", man.getvalue())
        z.writestr("README.txt",
                   "Speedwatch confirmed-speeder evidence bundle.\n"
                   "Each numbered folder contains the video clip and stills for one vehicle pass.\n"
                   "manifest.csv lists every pass with date/time, plate, measured speed and file names.\n")
    return buf.getvalue()


@app.get("/api/evidence/export.csv")
async def evidence_export_csv():
    rows = _evidence_rows(await db.list_evidence_passes())
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow([label for _, label in _EVIDENCE_COLS])
    for r in rows:
        w.writerow([r.get(k, "") for k, _ in _EVIDENCE_COLS])
    fname = f"speedwatch_evidence_{datetime.now():%Y%m%d}.csv"
    return Response(content=out.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@app.get("/api/evidence/export.zip")
async def evidence_export_zip():
    rows = _evidence_rows(await db.list_evidence_passes())
    if not rows:
        raise HTTPException(status_code=404, detail="No confirmed-speeder passes to export")
    # Zip building is blocking (file I/O + compression) — run off the event loop.
    blob = await asyncio.to_thread(_build_evidence_zip, rows)
    fname = f"speedwatch_evidence_{datetime.now():%Y%m%d}.zip"
    return Response(content=blob, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@app.post("/api/vehicle/{plate}/meta")
async def api_vehicle_meta(
    plate: str,
    user_description: str = Form(default=""),
    user_notes: str = Form(default=""),
):
    await db.update_vehicle_meta(plate, user_description or None, user_notes or None)
    return JSONResponse({"ok": True})


@app.post("/api/vehicle/{plate}/rep-image")
async def api_vehicle_rep_image(plate: str, request: Request):
    """Pin a vehicle's representative thumbnail to a chosen pass's image. Body: {pass_id}.
    Fixes a stale/wrong rep image (e.g. after manually cleaning up a mis-grouped pass)."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    pass_id = body.get("pass_id")
    if not pass_id:
        raise HTTPException(status_code=422, detail="pass_id required")
    p = await db.get_pass(int(pass_id))
    if p is None:
        raise HTTPException(status_code=404, detail="Pass not found")
    img = p.get("car_crop_path") or p.get("entry_frame_path") or p.get("entry_frame_raw_path")
    if not img:
        raise HTTPException(status_code=422, detail="This pass has no image to use as a thumbnail")
    canon = await db.canonical_plate(plate)
    await db.set_vehicle_rep_image_manual(canon, img)
    return JSONResponse({"ok": True, "rep_image_path": img, "plate": canon})


@app.post("/api/vehicle/{plate}/recompute-plate")
async def api_vehicle_recompute_plate(plate: str):
    """Re-read the plate from every image across all of this vehicle's passes and suggest a
    combined best guess. Does not apply it — the UI confirms, then applies via /merge."""
    if not config.ENABLE_ALPR:
        raise HTTPException(status_code=503, detail="ANPR is disabled")
    canon = await db.canonical_plate(plate)
    return JSONResponse(await vehicles_mod.recompute_plate(canon))


@app.post("/api/vehicle/{plate}/describe")
async def api_vehicle_describe(plate: str):
    """Manually (re)generate the AI vehicle description from its representative crop."""
    desc = await vehicles_mod.describe(plate, force=True)
    if desc is None:
        raise HTTPException(status_code=503,
                            detail="Vision model unavailable, or no car image for this plate yet")
    return JSONResponse({"ok": True, "description": desc})


@app.post("/api/vehicle/{plate}/merge")
async def api_vehicle_merge(plate: str, request: Request):
    """Group plates as one vehicle. Body: {canonical: str, aliases: [str, ...]}."""
    body = await request.json()
    canonical = (body.get("canonical") or plate).strip().upper()
    aliases = [a.strip().upper() for a in body.get("aliases", []) if a.strip()]
    if not canonical:
        raise HTTPException(status_code=422, detail="canonical plate required")
    root = await db.add_plate_aliases(canonical, aliases)
    return JSONResponse({"ok": True, "canonical": root})


@app.post("/api/vehicle/{plate}/unmerge")
async def api_vehicle_unmerge(plate: str, request: Request):
    """Remove one alias ({alias: str}) or all aliases of this canonical ({all: true})."""
    body = await request.json()
    if body.get("all"):
        await db.unmerge_plate(plate)
    elif body.get("alias"):
        await db.remove_plate_alias(body["alias"].strip().upper())
    return JSONResponse({"ok": True})


@app.post("/api/vehicles/backfill-images")
async def api_vehicles_backfill():
    """Re-derive + save car crops for plated passes recorded before car_crop_path existed."""
    return JSONResponse(await vehicles_mod.backfill_car_images())


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

async def _grab_snapshot_jpeg() -> bytes:
    """Grab one JPEG frame from the camera stream via ffmpeg. Raises HTTPException on failure."""
    cmd = [
        config.FFMPEG_BIN,
        "-loglevel", "error",
        "-timeout", "5000000",
        "-rtsp_transport", "tcp",
        "-i", config.CAMERA_RTSP,
        "-map", "0:v:0",
        "-vframes", "1",
        "-f", "image2",
        "-vcodec", "mjpeg",
        "pipe:1",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Camera snapshot timed out")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Snapshot error: {e}")
    if not stdout:
        raise HTTPException(status_code=503, detail="No frame from camera")
    return stdout


@app.get("/snapshot/{camera}")
async def snapshot(camera: str):
    """Grab one JPEG frame from the RTSPS substream."""
    return Response(content=await _grab_snapshot_jpeg(), media_type="image/jpeg")


# ── Lens distortion tuner (manual plumb-line calibration) ──────────────────────
# The snapshot used for tuning is cached so dragging the sliders re-warps the same frame
# instead of re-hitting the camera each time. "refresh" forces a fresh grab.

_lens_frame_cache: dict[str, "np.ndarray"] = {}


async def _lens_frame(camera: str, refresh: bool = False):
    import numpy as np
    import cv2
    arr = _lens_frame_cache.get(camera)
    if arr is None or refresh:
        jpeg = await _grab_snapshot_jpeg()
        arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise HTTPException(status_code=503, detail="Could not decode snapshot")
        _lens_frame_cache[camera] = arr
    return arr


def _lens_K(k1: float, k2: float, f_frac: float, w: int, h: int):
    import numpy as np
    f = f_frac * w
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    return K, dist


@app.get("/api/lens/preview")
async def lens_preview(camera: str, k1: float = 0.0, k2: float = 0.0,
                       f: float = 0.5, refresh: int = 0):
    """Return the cached snapshot undistorted with the given parameters (downscaled JPEG)."""
    import cv2
    arr = await _lens_frame(camera, refresh=bool(refresh))
    h, w = arr.shape[:2]
    max_w = 1280
    if w > max_w:
        s = max_w / w
        arr = cv2.resize(arr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        h, w = arr.shape[:2]
    K, dist = _lens_K(k1, k2, f, w, h)   # f_frac is resolution-independent
    und = cv2.undistort(arr, K, dist)
    ok, buf = cv2.imencode(".jpg", und, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise HTTPException(status_code=500, detail="encode failed")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.post("/api/anpr-mask/add")
async def api_anpr_mask_add(request: Request):
    """Append one ANPR ignore rectangle (normalised [x1,y1,x2,y2], 0..1) to a camera's existing
    set. Used by the plate-recompute UI to mask a fixed parked/background plate in one click.
    Takes effect on the next vehicle (mtime-cached, no restart)."""
    body = await request.json()
    camera = body.get("camera", config.CAMERA_NAME)
    rect = body.get("rect")
    if not rect or len(rect) != 4:
        raise HTTPException(status_code=422, detail="rect [x1,y1,x2,y2] required")
    existing = anpr_mask_mod.load(camera)
    anpr_mask_mod.save(camera, list(existing) + [rect])
    saved = anpr_mask_mod.load(camera)
    return JSONResponse({"ok": True, "count": len(saved)})


@app.post("/api/anpr-mask/save")
async def api_anpr_mask_save(request: Request):
    """Save the per-camera ANPR ignore zones. Body: {camera, rects:[[x1,y1,x2,y2],...]}
    in normalised 0..1 coords. Takes effect on the next vehicle (mtime-cached, no restart)."""
    body = await request.json()
    camera = body.get("camera", config.CAMERA_NAME)
    rects = body.get("rects", [])
    anpr_mask_mod.save(camera, rects)
    saved = anpr_mask_mod.load(camera)
    return JSONResponse({"ok": True, "count": len(saved)})


@app.post("/api/lens/save")
async def lens_save(camera: str, request: Request):
    """Persist manually-tuned lens intrinsics (at the full snapshot resolution)."""
    body = await request.json()
    k1 = float(body.get("k1", 0.0))
    k2 = float(body.get("k2", 0.0))
    f_frac = float(body.get("f", 0.5))
    arr = await _lens_frame(camera)
    h, w = arr.shape[:2]
    intr = intr_mod.build_manual(camera, k1, k2, f_frac, w, h)
    path = intr_mod.save(camera, intr)
    log.info("Saved manual lens intrinsics for %s (k1=%.4f k2=%.4f f=%.3f) → %s",
             camera, k1, k2, f_frac, path)
    return JSONResponse({"ok": True, "image_w": w, "image_h": h})


# ── Authentication ─────────────────────────────────────────────────────────────
# Installs the session + auth-gate middleware, the Google SSO routes (/login,
# /auth/callback, /logout) and the admin Users screen. Registered last so it wraps
# every route defined above. See app/auth.py for the role model.
from . import auth as _auth  # noqa: E402
_auth.install(app, templates)


# ── MCP server + API-key access (for an external agent, e.g. Hermes) ─────────────
# Mounts the MCP server at /mcp and installs the API-key gate. The gate is added AFTER the
# SSO middleware so it ends up outermost and runs first: it requires a key for /mcp and lets a
# key-bearing agent reach the media/export routes without a browser session. Managed at /keys.
if config.MCP_ENABLED:
    from . import mcp_server as _mcp  # noqa: E402
    from . import apikey as _apikey   # noqa: E402
    app.mount("/mcp", _mcp.streamable_app())
    _apikey.install(app)

    @app.get("/keys", response_class=HTMLResponse)
    async def keys_page(request: Request):
        keys = await db.list_api_keys()
        return templates.TemplateResponse("keys.html", {
            "request": request,
            "keys": keys,
            "mcp_path": "/mcp",
            "public_base": config.MCP_PUBLIC_BASE_URL,
            "pipeline_status": _pipeline_status,
        })

    @app.post("/api/keys")
    async def api_create_key(request: Request, name: str = Form(...)):
        name = name.strip()
        if not name:
            raise HTTPException(status_code=422, detail="A name is required")
        actor = _auth.current_user(request)
        raw = await db.create_api_key(name, created_by=actor["email"] if actor else None)
        # Returned ONCE — the raw key is never recoverable after this response.
        return JSONResponse({"ok": True, "name": name, "key": raw})

    @app.post("/api/keys/revoke")
    async def api_revoke_key(request: Request, key_id: int = Form(...)):
        await db.revoke_api_key(key_id)
        return JSONResponse({"ok": True})
