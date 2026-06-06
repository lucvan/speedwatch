"""
MCP (Model Context Protocol) server for speedwatch.

Mounted at /mcp on the main app (see server.py). Exposes READ-ONLY evidence data + image/clip
URLs and an on-demand clip-bundle builder, so an external agent (e.g. Hermes) can author an
enriched council report with its own research + models. SpeedWatch stays the source of truth
for the measured figures; the agent only adds prose and supporting research.

Auth is by API key (see app/apikey.py + the /keys admin page). The same key authorises the
media/export routes, so the agent can fetch the images and clips a tool points it at. Media
travels as absolute URLs built from `config.MCP_PUBLIC_BASE_URL` (Hermes reaches this host via
host.docker.internal).
"""
from __future__ import annotations
import asyncio
import secrets
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from . import config, db
from . import stats as stats_mod

_INSTRUCTIONS = (
    "Speedwatch measures vehicle speeds on a residential road from a single camera (homography "
    "+ ANPR). Use these tools to pull the measured evidence for a traffic-calming report: "
    "road_stats for the headline distribution, list_offenders for persistent speeders, "
    "vehicle_profile(plate) for one vehicle's full pass history, speed_distribution for embeddable "
    "SVG charts, and request_clip_bundle to assemble a downloadable evidence pack. All figures are "
    "ground truth — do not invent or alter numbers; you supply the narrative and research, not the data."
)

mcp = FastMCP(
    "speedwatch",
    instructions=_INSTRUCTIONS,
    stateless_http=True,
    streamable_http_path="/",
)


def _media_url(rel: str | None) -> str | None:
    """Absolute URL (reachable by the agent) for a stored media path like 'frames/x.jpg'."""
    if not rel:
        return None
    return f"{config.MCP_PUBLIC_BASE_URL}/{str(rel).lstrip('/')}"


def _eff_speed(p: dict) -> float | None:
    """The speed to reason about: reviewer-corrected if present, else the measured speed."""
    return p.get("user_corrected_mph") if p.get("user_corrected_mph") is not None else p.get("sw_speed_mph")


# ── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool()
async def road_stats() -> dict:
    """Road-wide speed profile aggregated over every measured pass. The report's headline
    figures: median, mean, 85th/95th percentile, fastest, count + % over the limit, the
    observation window, and the number of distinct vehicles and repeat offenders."""
    limit, warn = config.SPEED_LIMIT_MPH, config.SPEED_WARN_MPH
    road = await db.all_pass_speeds()
    summary = stats_mod.speed_summary(road, warn=warn, limit=limit) or {}
    vehicles = await db.list_vehicles(order_by="max_mph", min_passes=1)
    first_seen = min((v["first_seen"] for v in vehicles if v.get("first_seen")), default=None)
    last_seen = max((v["last_seen"] for v in vehicles if v.get("last_seen")), default=None)
    days = None
    if first_seen and last_seen:
        days = max(1, round((float(last_seen) - float(first_seen)) / 86400) or 1)
    over = await db.vehicle_over_limit_counts(limit)
    offenders = sum(
        1 for v in vehicles
        if over.get(v["plate"], 0) >= config.REPORT_REPEAT_MIN_OVER or v.get("confirmed_speeder")
    )
    return {
        "road_name": config.REPORT_ROAD_NAME,
        "location": config.REPORT_LOCATION,
        "council": config.REPORT_COUNCIL,
        "speed_limit_mph": limit,
        "warn_mph": warn,
        "evidence_floor_mph": config.EVIDENCE_MIN_MPH,
        "observation": {"first_seen": first_seen, "last_seen": last_seen, "days": days},
        "distinct_vehicles": len(vehicles),
        "repeat_offender_count": offenders,
        "summary": summary,  # n, min, mean, median, p85, p95, max, over_limit, over_pct
    }


@mcp.tool()
async def list_offenders() -> list[dict]:
    """Persistent offenders — the same rule the council report uses: vehicles with at least
    REPORT_REPEAT_MIN_OVER passes over the limit, OR manually confirmed. Each entry has summary
    stats + a representative image URL. Call vehicle_profile(plate) for the full pass history."""
    limit = config.SPEED_LIMIT_MPH
    vehicles = await db.list_vehicles(order_by="max_mph", min_passes=1)
    over = await db.vehicle_over_limit_counts(limit)
    road = sorted([s for s in await db.all_pass_speeds() if s is not None])
    out = []
    for v in vehicles:
        plate = v["plate"]
        over_cnt = over.get(plate, 0)
        confirmed = bool(v.get("confirmed_speeder"))
        if over_cnt < config.REPORT_REPEAT_MIN_OVER and not confirmed:
            continue
        out.append({
            "plate": plate,
            "is_synthetic": plate.startswith(db.SYNTH_PLATE_PREFIX),
            "passes": v["passes"],
            "max_mph": round(v["max_mph"], 1) if v.get("max_mph") is not None else None,
            "avg_mph": round(v["avg_mph"], 1) if v.get("avg_mph") is not None else None,
            "passes_over_limit": over_cnt,
            "confirmed": confirmed,
            "auto_flagged": over_cnt >= config.REPORT_REPEAT_MIN_OVER,
            "avg_percentile_vs_road": stats_mod.percentile_rank(road, v.get("avg_mph")),
            "description": v.get("user_description") or v.get("description"),
            "notes": v.get("user_notes"),
            "first_seen": v.get("first_seen"),
            "last_seen": v.get("last_seen"),
            "image_url": _media_url(v.get("rep_image_path")),
        })
    out.sort(key=lambda o: (o["max_mph"] or 0), reverse=True)
    return out


@mcp.tool()
async def vehicle_profile(plate: str) -> dict:
    """Full profile for one vehicle by (canonical) plate: summary stats, description/notes,
    representative image, median percentile vs the road, and EVERY pass (time, speed, whether
    it was over the limit, percentile vs the road, confidence, and clip + still URLs). Use to
    build a persistent-offender section of the report. Aliases resolve to the canonical plate."""
    plate = await db.canonical_plate(plate.strip().upper())
    v = await db.get_vehicle(plate)
    if v is None:
        return {"error": f"no vehicle found for plate {plate}"}
    passes = await db.list_passes_for_plate(plate)
    road = sorted([s for s in await db.all_pass_speeds() if s is not None])
    speeds = [_eff_speed(p) for p in passes if _eff_speed(p) is not None]
    summary = stats_mod.speed_summary(speeds, warn=config.SPEED_WARN_MPH, limit=config.SPEED_LIMIT_MPH)
    pass_list = []
    for p in passes:
        s = _eff_speed(p)
        pass_list.append({
            "pass_id": p["id"],
            "session_id": p["session_id"],
            "timestamp": p.get("start_ts"),
            "speed_mph": round(s, 1) if s is not None else None,
            "over_limit": (s is not None and s > config.SPEED_LIMIT_MPH),
            "percentile_vs_road": stats_mod.percentile_rank(road, s),
            "confidence": p.get("confidence"),
            "plate_confidence": p.get("plate_confidence"),
            "reviewer_label": p.get("user_label"),
            "clip_url": _media_url(p.get("clip_path") or p.get("session_clip_path")),
            "car_crop_url": _media_url(p.get("car_crop_path")),
            "entry_still_url": _media_url(p.get("entry_frame_path")),
            "exit_still_url": _media_url(p.get("exit_frame_path")),
        })
    return {
        "plate": plate,
        "is_synthetic": plate.startswith(db.SYNTH_PLATE_PREFIX),
        "description": v.get("user_description") or v.get("description"),
        "notes": v.get("user_notes"),
        "image_url": _media_url(v.get("rep_image_path")),
        "summary": summary,
        "median_percentile_vs_road": (
            stats_mod.percentile_rank(road, summary["median"]) if summary else None
        ),
        "first_seen": v.get("first_seen"),
        "last_seen": v.get("last_seen"),
        "passes": pass_list,
    }


@mcp.tool()
async def pass_detail(pass_id: int) -> dict:
    """All recorded detail for a single pass, including media URLs (clip, entry/exit stills,
    trajectory overlay, car crop) and the measurement-quality fields."""
    p = await db.get_pass(pass_id)
    if p is None:
        return {"error": f"no pass with id {pass_id}"}
    s = _eff_speed(p)
    return {
        "pass_id": p["id"],
        "session_id": p["session_id"],
        "plate": p.get("plate"),
        "assigned_plate": p.get("assigned_plate"),
        "plate_confidence": p.get("plate_confidence"),
        "measured_mph": p.get("sw_speed_mph"),
        "corrected_mph": p.get("user_corrected_mph"),
        "speed_mph": round(s, 1) if s is not None else None,
        "over_limit": (s is not None and s > config.SPEED_LIMIT_MPH),
        "confidence": p.get("confidence"),
        "dwell_seconds": p.get("dwell_seconds"),
        "frames_in_zone": p.get("frames_in_zone"),
        "reviewer_label": p.get("user_label"),
        "reviewer_notes": p.get("user_notes"),
        "clip_url": _media_url(p.get("clip_path")),
        "entry_still_url": _media_url(p.get("entry_frame_path")),
        "exit_still_url": _media_url(p.get("exit_frame_path")),
        "trajectory_url": _media_url(p.get("trajectory_overlay_path")),
        "car_crop_url": _media_url(p.get("car_crop_path")),
    }


@mcp.tool()
async def speed_distribution(plate: str | None = None) -> dict:
    """An inline-SVG speed-distribution chart ready to embed in an HTML report. Without a plate:
    the whole-road distribution with the 85th-percentile marker. With a plate: that vehicle's
    passes overlaid as ticks on the road distribution. Returns the SVG markup plus the summary."""
    road = await db.all_pass_speeds()
    warn, limit = config.SPEED_WARN_MPH, config.SPEED_LIMIT_MPH
    if plate:
        plate = await db.canonical_plate(plate.strip().upper())
        passes = await db.list_passes_for_plate(plate)
        hl = [_eff_speed(p) for p in passes if _eff_speed(p) is not None]
        svg = stats_mod.speed_distribution_svg(
            road, highlight=hl, warn=warn, limit=limit,
            highlight_label=plate, width=720, height=260,
        )
        return {"plate": plate, "svg": svg,
                "summary": stats_mod.speed_summary(hl, warn=warn, limit=limit)}
    summary = stats_mod.speed_summary(road, warn=warn, limit=limit) or {}
    markers = []
    if summary.get("p85") is not None:
        markers.append((summary["p85"], "#4fc3f7", f"85th {summary['p85']:g}"))
    svg = stats_mod.speed_distribution_svg(
        road, warn=warn, limit=limit, markers=markers, width=720, height=300)
    return {"svg": svg, "summary": summary}


@mcp.tool()
async def request_clip_bundle(plates: list[str] | None = None,
                              pass_ids: list[int] | None = None) -> dict:
    """Build an evidence clip bundle (a ZIP: one folder per pass with the video clip + stills,
    plus a manifest.csv) for the given canonical plates and/or pass ids, and return a download
    URL. Use this to attach a supporting evidence pack to the report. Provide at least one of
    `plates` or `pass_ids`."""
    if not plates and not pass_ids:
        return {"error": "provide at least one of plates or pass_ids"}
    # Imported here (not at module load) to avoid a circular import: server imports this module.
    from .server import _evidence_rows, _build_evidence_zip

    rows = await db.passes_for_export(pass_ids=pass_ids, plates=plates)
    if not rows:
        return {"error": "no passes matched the given plates / pass_ids"}
    blob = await asyncio.to_thread(_build_evidence_zip, _evidence_rows(rows))
    exports = Path(config.DATA_DIR) / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    fname = f"clips_{int(time.time())}_{secrets.token_urlsafe(6)}.zip"
    (exports / fname).write_bytes(blob)
    return {
        "download_url": _media_url(f"exports/{fname}"),
        "passes": len(rows),
        "bytes": len(blob),
    }


# ── App / lifespan plumbing (used by server.py + main.py) ────────────────────

_app = None


def streamable_app():
    """The ASGI app to mount at /mcp. Creates the session manager lazily on first call."""
    global _app
    if _app is None:
        _app = mcp.streamable_http_app()
    return _app


def session_manager_lifespan():
    """Context manager that runs the streamable-HTTP session manager task. Must be entered by
    the parent app's lifespan (a mounted sub-app's own lifespan is not invoked by Starlette).
    `streamable_app()` must have been called first (it creates the session manager)."""
    return mcp.session_manager.run()
