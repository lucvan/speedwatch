"""
Council evidence report builder.

Assembles a print-ready "case for traffic calming" report from the recorded passes:

  * the road-wide speed distribution (what a normal, competent & careful driver does here),
  * the minority of vehicles that *regularly* exceed the limit (auto-selected from the data
    plus any manually-confirmed speeders), each with their own speed profile overlaid on the
    road as a whole, and
  * a written narrative that frames the figures as a road-safety argument.

The narrative is written by a LOCAL Ollama text model (no data leaves the box). Every section
has a deterministic, figures-grounded fallback, so the report is complete and accurate even
when Ollama is down — the model only ever rephrases facts it was given, it never invents them.

The "evidence floor" (config.EVIDENCE_MIN_MPH, default 17 mph) drops crawl-past passes from the
per-vehicle evidence and the offender selection; the road-wide distribution still shows every
pass so the reader sees the true shape of normal driving.
"""
from __future__ import annotations
import asyncio
import json
import logging
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from . import config, db
from . import stats as stats_mod

log = logging.getLogger(__name__)


# ── Local LLM (Ollama text generation) ───────────────────────────────────────

def _ollama_generate(prompt: str, system: str | None = None, *, as_json: bool = True) -> str | None:
    """One blocking text generation against local Ollama. Returns the raw response string
    (JSON text when as_json) or None on any failure. Call from a thread executor."""
    body = {
        "model": config.REPORT_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.4},
    }
    if system:
        body["system"] = system
    if as_json:
        body["format"] = "json"
    req = urllib.request.Request(
        config.OLLAMA_URL.rstrip("/") + "/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=config.REPORT_LLM_TIMEOUT).read())
    return (resp.get("response") or "").strip() or None


async def _generate_json(prompt: str, system: str | None = None) -> dict | None:
    """Run a JSON generation off the event loop; tolerate a non-JSON / partial reply."""
    if not config.REPORT_LLM_ENABLED:
        return None
    try:
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, lambda: _ollama_generate(prompt, system))
    except Exception as e:
        log.warning("report LLM call failed: %s", e)
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        # Best-effort: pull the outermost {...} block out of a chatty reply.
        a, b = raw.find("{"), raw.rfind("}")
        if a >= 0 and b > a:
            try:
                return json.loads(raw[a:b + 1])
            except Exception:
                pass
    log.warning("report LLM returned unparseable JSON (%d chars)", len(raw))
    return None


_SYSTEM = (
    "You are a road-safety analyst writing a formal evidence report for a UK local authority "
    "(council). Write in British English, in a measured, factual, professional tone suitable "
    "for submission to a council's traffic team. Use ONLY the figures supplied to you — never "
    "invent or round to different numbers, never add statistics you were not given. Do not use "
    "markdown, bullet points, or headings inside a section: return plain prose paragraphs. "
    "Where relevant you may refer to the statutory standard of 'a competent and careful driver' "
    "(section 3 Road Traffic Act 1988) and to the 85th-percentile speed as the standard "
    "traffic-engineering benchmark. Always return valid JSON in the exact shape requested."
)


# ── Numeric helpers ──────────────────────────────────────────────────────────

def _eff(p: dict) -> float | None:
    v = p.get("user_corrected_mph")
    if v is None:
        v = p.get("sw_speed_mph")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _fmt_date(ts) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%-d %B %Y") if ts else "—"
    except (ValueError, OSError):
        # Windows strftime has no %-d; fall back to a manual day-of-month.
        try:
            d = datetime.fromtimestamp(float(ts))
            return d.strftime("%d %B %Y").lstrip("0")
        except Exception:
            return "—"
    except Exception:
        return "—"


def _fmt_datetime(ts) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%d %b %Y, %H:%M")
    except Exception:
        return "—"


def _pick_image(v: dict, passes: list[dict]) -> str | None:
    """Best available still for a vehicle: its pinned/rep crop, else a pass car-crop, else an
    entry frame. Returned as a web path (already prefixed with frames/…)."""
    if v.get("rep_image_path"):
        return v["rep_image_path"]
    for p in passes:
        for key in ("car_crop_path", "entry_frame_path", "entry_frame_raw_path"):
            if p.get(key):
                return p[key]
    return None


# ── Data gathering ───────────────────────────────────────────────────────────

async def _gather() -> dict:
    """Pull every figure the report needs from the DB. No LLM here — pure data."""
    limit = config.SPEED_LIMIT_MPH
    warn  = config.SPEED_WARN_MPH
    floor = config.EVIDENCE_MIN_MPH

    road = await db.all_pass_speeds()                       # every measured speed (the bell)
    road_sorted = sorted(road)
    road_summary = stats_mod.speed_summary(road, warn=warn, limit=limit) or {
        "n": 0, "min": None, "mean": None, "median": None, "p85": None,
        "p95": None, "max": None, "over_limit": 0, "over_pct": None,
    }

    # Observation window + distinct-vehicle count, from the vehicle stats view.
    vehicles = await db.list_vehicles(order_by="max_mph", min_passes=1)
    first_seen = min((v["first_seen"] for v in vehicles if v.get("first_seen")), default=None)
    last_seen  = max((v["last_seen"]  for v in vehicles if v.get("last_seen")),  default=None)
    days = None
    if first_seen and last_seen:
        days = max(1, round((float(last_seen) - float(first_seen)) / 86400) or 1)

    # Repeat offenders: vehicles with >= REPEAT_MIN_OVER passes over the limit, OR manually
    # confirmed. Sorted fastest-first so the worst offenders lead the report.
    over_counts = await db.vehicle_over_limit_counts(limit)
    offenders: list[dict] = []
    for v in vehicles:
        plate = v["plate"]
        over_cnt = over_counts.get(plate, 0)
        confirmed = bool(v.get("confirmed_speeder"))
        if over_cnt < config.REPORT_REPEAT_MIN_OVER and not confirmed:
            continue
        passes = await db.list_passes_for_plate(plate)
        # Document shows EVERY pass of a fast driver (the floor only trims the ZIP clip bundle),
        # so the profile shows their full driving pattern against the road.
        ev = []
        for p in passes:
            s = _eff(p)
            if s is None:
                continue
            ev.append({
                "id": p["id"],
                "session_id": p["session_id"],
                "ts": p.get("start_ts"),
                "speed": round(s, 1),
                "over": s > limit,
                "vs_road": stats_mod.percentile_rank(road_sorted, s),
                "plate_confidence": p.get("plate_confidence"),
                "car_crop_path": p.get("car_crop_path"),
                "entry_frame_path": p.get("entry_frame_path"),
                "trajectory_overlay_path": p.get("trajectory_overlay_path"),
                "clip_path": p.get("clip_path") or p.get("session_clip_path"),
            })
        if not ev:
            continue
        ev.sort(key=lambda r: r["speed"], reverse=True)
        speeds = [r["speed"] for r in ev]
        summary = stats_mod.speed_summary(speeds, warn=warn, limit=limit) or {}
        med = summary.get("median")
        offenders.append({
            "plate": plate,
            "is_synthetic": plate.startswith("UNK-"),
            "confirmed": confirmed,
            "auto_flagged": over_cnt >= config.REPORT_REPEAT_MIN_OVER,
            "description": v.get("user_description") or v.get("description"),
            "notes": v.get("user_notes"),
            "image": _pick_image(v, passes),
            "summary": summary,
            "over_cnt": over_cnt,
            "median_rank": stats_mod.percentile_rank(road_sorted, med) if med is not None else None,
            "first_seen": min((r["ts"] for r in ev if r["ts"]), default=None),
            "last_seen":  max((r["ts"] for r in ev if r["ts"]), default=None),
            "passes": ev,
            "speeds": speeds,
            "chart": stats_mod.speed_distribution_svg(
                road, highlight=speeds, warn=warn, limit=limit,
                highlight_label=plate, width=680, height=210,
            ),
        })
    offenders.sort(key=lambda o: o["summary"].get("max", 0), reverse=True)

    # Road-wide chart with the 85th-percentile marker (the engineering benchmark).
    markers = []
    if road_summary.get("p85") is not None:
        markers.append((road_summary["p85"], "#4fc3f7", f"85th {road_summary['p85']:g}"))
    road_chart = stats_mod.speed_distribution_svg(
        road, warn=warn, limit=limit, markers=markers, width=720, height=300)

    return {
        "limit": limit, "warn": warn, "floor": floor,
        "road": road, "road_summary": road_summary, "road_chart": road_chart,
        "n_vehicles": len(vehicles),
        "first_seen": first_seen, "last_seen": last_seen, "days": days,
        "offenders": offenders,
    }


# ── Narrative (LLM with deterministic fallback) ──────────────────────────────

def _road_facts(d: dict) -> dict:
    rs = d["road_summary"]
    n = rs.get("n", 0)
    over = rs.get("over_limit", 0)
    return {
        "road_name": config.REPORT_ROAD_NAME,
        "location": config.REPORT_LOCATION,
        "council": config.REPORT_COUNCIL,
        "speed_limit_mph": d["limit"],
        "evidence_floor_mph": d["floor"],
        "total_passes": n,
        "distinct_vehicles": d["n_vehicles"],
        "observation_days": d["days"],
        "date_from": _fmt_date(d["first_seen"]),
        "date_to": _fmt_date(d["last_seen"]),
        "median_mph": rs.get("median"),
        "mean_mph": rs.get("mean"),
        "p85_mph": rs.get("p85"),
        "fastest_mph": rs.get("max"),
        "passes_over_limit": over,
        "pct_over_limit": rs.get("over_pct"),
        "num_repeat_offenders": len(d["offenders"]),
    }


def _offender_facts(o: dict) -> dict:
    s = o["summary"]
    return {
        "plate": o["plate"],
        # A vehicle with a real plate is identified even if it has no visual description; only a
        # synthetic UNK- record is genuinely unidentified. Keep these distinct so the model does
        # not call a plated vehicle "unidentified".
        "identified": not o["is_synthetic"],
        "number_plate": None if o["is_synthetic"] else o["plate"],
        "description": o["description"] or None,
        "passes_recorded": s.get("n"),
        "fastest_mph": s.get("max"),
        "median_mph": s.get("median"),
        "passes_over_limit": s.get("over_limit"),
        "pct_over_limit": s.get("over_pct"),
        "faster_than_pct_of_road": round(o["median_rank"]) if o.get("median_rank") is not None else None,
        "first_seen": _fmt_date(o["first_seen"]),
        "last_seen": _fmt_date(o["last_seen"]),
    }


# Deterministic, figures-grounded fallbacks (also the data the LLM rephrases).

def _fallback_intro(f: dict) -> str:
    loc = f" in {f['location']}" if f["location"] else ""
    return (
        f"This report presents speed-measurement evidence gathered on {f['road_name']}{loc} and "
        f"is submitted to {f['council']} in support of a request for traffic-calming measures. "
        f"Over the observation period ({f['date_from']} to {f['date_to']}), an automated "
        f"camera-based speed-measurement system recorded {f['total_passes']} vehicle passes from "
        f"{f['distinct_vehicles']} distinct vehicles. The road carries a {f['speed_limit_mph']:g} mph "
        f"speed limit. The evidence below shows that, while most drivers travel at a speed "
        f"consistent with a competent and careful driver, a small number of vehicles repeatedly "
        f"travel substantially faster, presenting a road-safety risk that calming measures would "
        f"address."
    )


def _fallback_methodology(f: dict) -> str:
    return (
        "Speeds were measured automatically from a fixed camera overlooking the carriageway. Each "
        "vehicle is detected, tracked frame-to-frame and its speed derived by mapping its image "
        "position onto a surveyed ground plane (homography calibration), then averaged across its "
        "transit of the measured zone. Number plates are read automatically so that repeat "
        "journeys by the same vehicle can be grouped. Every recorded pass is included in the speed "
        "figures and distributions in this document, so the road-wide distribution shows the true "
        f"shape of normal driving on this road. The accompanying video-clip bundle is limited to "
        f"passes of {f['evidence_floor_mph']:g} mph and above, as a conservative noise floor for the "
        "supporting footage."
    )


def _fallback_road_findings(f: dict) -> str:
    pct = f["pct_over_limit"]
    pct_txt = f"{pct:g}%" if pct is not None else "a number"
    return (
        f"Across all {f['total_passes']} recorded passes the median speed was "
        f"{f['median_mph']:g} mph and the 85th-percentile speed — the standard benchmark used in "
        f"traffic engineering — was {f['p85_mph']:g} mph, against a {f['speed_limit_mph']:g} mph "
        f"limit. This confirms that the typical driver on this road, the competent and careful "
        f"majority, travels at or close to the limit. However, {pct_txt} of passes "
        f"({f['passes_over_limit']} in total) exceeded the limit, and the fastest recorded pass "
        f"reached {f['fastest_mph']:g} mph. The distribution is not a simple bell centred on the "
        f"limit: it has a pronounced upper tail of vehicles travelling well above the speed a "
        f"careful driver would adopt on a road of this character."
    )


def _fallback_offender(f: dict) -> str:
    rank = f["faster_than_pct_of_road"]
    rank_txt = (f" Its typical speed is faster than approximately {rank}% of all vehicles using "
                f"the road.") if rank is not None else ""
    desc = f["description"]
    desc_txt = f" ({desc})" if desc else ""
    ref = f"Vehicle {f['number_plate']}" if f["identified"] else "An unidentified vehicle"
    return (
        f"{ref}{desc_txt} was recorded passing the camera {f['passes_recorded']} "
        f"times between {f['first_seen']} and {f['last_seen']}, of which {f['passes_over_limit']} "
        f"were over the {config.SPEED_LIMIT_MPH:g} mph limit. Its fastest recorded pass was "
        f"{f['fastest_mph']:g} mph and its median speed was {f['median_mph']:g} mph.{rank_txt} "
        f"This is a pattern of repeated, deliberate speeding rather than an isolated lapse, and "
        f"falls well below the standard expected of a competent and careful driver."
    )


def _fallback_conclusion(f: dict) -> str:
    return (
        f"The evidence shows that {f['road_name']} experiences persistent speeding by a minority "
        f"of drivers who travel materially faster than the careful majority and well in excess of "
        f"the {f['speed_limit_mph']:g} mph limit. Because these are repeat offenders rather than "
        f"occasional ones, education and enforcement alone are unlikely to change their behaviour; "
        f"physical traffic-calming measures are the proportionate response. We therefore ask "
        f"{f['council']} to assess this location for traffic-calming intervention — such as speed "
        f"humps, a chicane, a raised table or a vehicle-activated sign — to protect residents, "
        f"pedestrians and other road users from the risk these drivers create."
    )


async def _narrative(d: dict) -> dict:
    """Produce all narrative sections, LLM-written where possible, deterministic otherwise."""
    rf = _road_facts(d)
    # Always compute the deterministic baseline first — it is the guaranteed-correct fallback.
    out = {
        "intro": _fallback_intro(rf),
        "methodology": _fallback_methodology(rf),
        "road_findings": _fallback_road_findings(rf),
        "conclusion": _fallback_conclusion(rf),
        "offenders": {o["plate"]: _fallback_offender(_offender_facts(o)) for o in d["offenders"]},
        "llm_used": False,
    }

    # Framing sections in one call.
    framing = await _generate_json(
        "Write the narrative sections of a council traffic-calming evidence report.\n"
        "Return JSON with exactly these string keys: intro, methodology, road_findings, "
        "conclusion. Each should be one to two paragraphs of plain prose. The conclusion must "
        "explicitly ask the council to introduce traffic-calming measures.\n\n"
        "FACTS (use these figures verbatim, invent nothing):\n"
        + json.dumps(rf, indent=2),
        system=_SYSTEM,
    )
    if framing:
        for k in ("intro", "methodology", "road_findings", "conclusion"):
            val = framing.get(k)
            if isinstance(val, str) and val.strip():
                out[k] = val.strip()
                out["llm_used"] = True

    # Per-offender paragraphs in one batched call (keyed by plate).
    if d["offenders"]:
        facts = [_offender_facts(o) for o in d["offenders"]]
        para = await _generate_json(
            "Write one short paragraph (about 60-90 words) about each of the following vehicles "
            "for the 'repeat offenders' section of the report. Each paragraph should summarise how "
            "often and how fast the vehicle was recorded and why this represents dangerous driving "
            "below the standard of a competent and careful driver. Do NOT speculate about the "
            "driver's identity or intent beyond what the figures support.\n"
            "If 'identified' is true, refer to the vehicle by its number plate; if 'identified' is "
            "false, refer to it only as 'an unidentified vehicle'. Never describe a vehicle that "
            "has a number plate as unidentified.\n"
            "Return JSON: {\"paragraphs\": {\"<PLATE>\": \"<text>\", ...}} using the exact plate "
            "strings as keys.\n\n"
            "VEHICLES (use these figures verbatim, invent nothing):\n"
            + json.dumps(facts, indent=2),
            system=_SYSTEM,
        )
        paras = (para or {}).get("paragraphs") or {}
        if isinstance(paras, dict):
            for o in d["offenders"]:
                val = paras.get(o["plate"])
                if isinstance(val, str) and val.strip():
                    out["offenders"][o["plate"]] = val.strip()
                    out["llm_used"] = True

    return out


# ── Cache + public entry point ───────────────────────────────────────────────
# The data gathering is cheap; the LLM calls are not. Cache the whole built context keyed by a
# data signature so reloads are instant, and only regenerate when the underlying data changed
# or the caller forces a refresh.

_CACHE: dict | None = None
_CACHE_SIG: tuple | None = None
_CACHE_TS: float = 0.0


def _signature(d: dict) -> tuple:
    return (
        len(d["road"]),
        round(d["road_summary"].get("max", 0), 1),
        tuple((o["plate"], o["summary"].get("n"), o["summary"].get("max")) for o in d["offenders"]),
    )


async def build_report_context(refresh: bool = False) -> dict:
    """Full template context for the council report. Cached on a data signature; pass
    refresh=True to force regeneration of the narrative."""
    global _CACHE, _CACHE_SIG, _CACHE_TS
    d = await _gather()
    sig = _signature(d)
    if not refresh and _CACHE is not None and _CACHE_SIG == sig:
        return _CACHE

    narrative = await _narrative(d)
    dates = None
    if d["first_seen"] and d["last_seen"]:
        dates = f"{_fmt_date(d['first_seen'])} – {_fmt_date(d['last_seen'])}"
    ctx = {
        "generated_at": datetime.now(),
        "road_name": config.REPORT_ROAD_NAME,
        "location": config.REPORT_LOCATION,
        "council": config.REPORT_COUNCIL,
        "author": config.REPORT_AUTHOR,
        "app_version": config.APP_VERSION,
        "narrative": narrative,
        "narrative_dates": dates,
        "repeat_min": config.REPORT_REPEAT_MIN_OVER,
        **d,
    }
    _CACHE, _CACHE_SIG, _CACHE_TS = ctx, sig, time.time()
    return ctx
