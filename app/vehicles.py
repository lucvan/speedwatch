"""
Vehicle enrichment: describe a car (colour/make/body) from its best car crop using a
local Ollama multimodal model. Triggered automatically for plates seen more than
config.VEHICLE_MIN_PASSES times, and on demand from the UI.

Local-only by design — neighbours' vehicle imagery never leaves the box. The HTTP call
runs in a thread executor so it never blocks the asyncio loop or the ingest thread.
"""
from __future__ import annotations
import asyncio
import base64
import json
import logging
import urllib.request
from pathlib import Path

from . import config, db

log = logging.getLogger(__name__)


def _ollama_describe(image_bytes: bytes) -> str | None:
    b64 = base64.b64encode(image_bytes).decode()
    body = json.dumps({
        "model": config.VEHICLE_VISION_MODEL,
        "prompt": config.VEHICLE_VISION_PROMPT,
        "images": [b64],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        config.OLLAMA_URL.rstrip("/") + "/api/generate",
        data=body, headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=config.VEHICLE_VISION_TIMEOUT).read())
    return (resp.get("response") or "").strip() or None


async def describe(plate: str, force: bool = False) -> str | None:
    """Generate (or regenerate, if force) the AI description for a plate's rep image."""
    meta = await db.get_vehicle_meta(plate)
    if not meta or not meta.get("rep_image_path"):
        return None
    if meta.get("description") and not force:
        return meta["description"]
    img_path = Path(config.DATA_DIR) / meta["rep_image_path"]
    if not img_path.exists():
        log.warning("vehicle %s: rep image missing (%s)", plate, img_path)
        return None
    img_bytes = img_path.read_bytes()
    try:
        desc = await asyncio.get_event_loop().run_in_executor(None, _ollama_describe, img_bytes)
    except Exception as e:
        log.warning("vehicle %s: vision describe failed: %s", plate, e)
        return None
    if desc:
        await db.set_vehicle_description(plate, desc)
        log.info("vehicle %s described: %s", plate, desc)
    return desc


async def maybe_describe(plate: str) -> None:
    """Auto-describe once a plate crosses the frequent-vehicle threshold."""
    if not config.VEHICLE_AUTODESCRIBE:
        return
    meta = await db.get_vehicle_meta(plate)
    if meta and meta.get("description"):
        return  # already described
    if await db.vehicle_pass_count(plate) > config.VEHICLE_MIN_PASSES:
        await describe(plate)


# ── Fuzzy plate matching (OCR-confusable aware) ──────────────────────────────

_CONFUSABLE = [set("0OQDU"), set("1ILT"), set("8B"), set("5S"), set("2Z"),
               set("6G"), set("9B"), set("4A"), set("7T"), set("VY"), set("EF")]


def _confusable(a: str, b: str) -> bool:
    return a == b or any(a in g and b in g for g in _CONFUSABLE)


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _soft_dist(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if len(a) == len(b):
        return sum(0.0 if x == y else (0.4 if _confusable(x, y) else 1.0) for x, y in zip(a, b))
    return float(_lev(a, b))


def similar_plates(target: str, candidates: list[str],
                   max_dist: float | None = None) -> list[tuple[str, float]]:
    """Candidates within OCR-aware soft edit-distance of target, closest first."""
    md = config.ALIAS_SUGGEST_MAXDIST if max_dist is None else max_dist
    out = [(c, _soft_dist(target, c)) for c in candidates if c and c != target]
    out = [(c, round(d, 2)) for c, d in out if d <= md]
    out.sort(key=lambda x: x[1])
    return out


# ── Backfill car crops for passes recorded before car_crop_path existed ──────

def _backfill_one_sync(pass_row: dict) -> tuple[str | None, bool]:
    """Re-derive the car box from a stored full-frame still and save a crop.
    Picks the box whose plate matches the recorded plate; else the largest car.
    Returns (crop_relpath, exact_plate_match)."""
    import cv2
    from . import detect as detect_mod, alpr as alpr_mod, clip as clip_mod
    target = (pass_row.get("plate") or "").upper()
    pid = pass_row["id"]
    for key in ("entry_frame_raw_path", "exit_frame_raw_path"):
        rel = pass_row.get(key)
        if not rel:
            continue
        full = Path(config.DATA_DIR) / rel
        if not full.exists():
            continue
        img = cv2.imread(str(full))
        if img is None:
            continue
        dets = detect_mod.run(img)
        if not dets:
            continue
        match_box = None
        best_box, best_area = None, 0.0
        for d in dets:
            x1, y1, x2, y2 = d.bbox_xyxy
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area, best_box = area, d.bbox_xyxy
            hit = alpr_mod.read_plate(img, d.bbox_xyxy)
            if hit and hit[0] == target:
                match_box = d.bbox_xyxy
                break
        box = match_box if match_box is not None else best_box
        if box is None:
            continue
        crop_rel = clip_mod.save_car_crop(img, box, f"bf{pid}", 0)
        if crop_rel:
            return crop_rel, match_box is not None
    return None, False


async def backfill_car_images(limit: int | None = None) -> dict:
    rows = await db.passes_missing_car_crop()
    if limit:
        rows = rows[:limit]
    loop = asyncio.get_event_loop()
    updated = matched = 0
    for row in rows:
        try:
            crop_rel, exact = await loop.run_in_executor(None, _backfill_one_sync, row)
        except Exception as e:
            log.warning("backfill pass %s failed: %s", row.get("id"), e)
            continue
        if not crop_rel:
            continue
        await db.set_pass_car_crop(row["id"], crop_rel)
        canon = await db.canonical_plate(row["plate"])
        await db.ensure_vehicle(canon)
        await db.update_vehicle_rep_image(canon, crop_rel, row.get("plate_confidence") or 0.0)
        updated += 1
        matched += 1 if exact else 0
    result = {"updated": updated, "exact_matches": matched, "total_missing": len(rows)}
    log.info("backfill car images: %s", result)
    return result
