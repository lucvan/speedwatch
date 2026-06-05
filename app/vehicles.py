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
import re
import urllib.request
from pathlib import Path

from . import config, db
from . import embed as embed_mod

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


# ── Appearance (description) matching ────────────────────────────────────────
# The car's colour/make/model description is an independent signal from the plate OCR:
# two OCR-variant plates that describe the same car are very likely the same vehicle.
# It corroborates (and loosens) plate similarity — it never stands alone, since many
# distinct cars share a make/model/colour.

_DESC_STOP = {"unknown", "car", "vehicle", "a", "an", "the", "and", "or", "with"}
_DESC_MATCH_MIN_SIM    = 0.5   # token Jaccard threshold for "same appearance"
_DESC_MATCH_MIN_SHARED = 2     # need ≥2 shared significant tokens (e.g. make + model)
_DESC_RANK_BONUS       = 0.75  # how much an appearance match improves ranking
_DESC_DIST_SLACK       = 1.5   # extra plate-distance allowed when appearance corroborates


def _desc_tokens(desc: str | None) -> set[str]:
    if not desc:
        return set()
    return {w for w in re.findall(r"[a-z0-9]+", desc.lower()) if w not in _DESC_STOP}


def _desc_match(target_tokens: set[str], cand_desc: str | None) -> tuple[bool, float]:
    """Return (is_match, jaccard_similarity) between a target's tokens and a candidate."""
    tb = _desc_tokens(cand_desc)
    if not target_tokens or not tb:
        return False, 0.0
    inter = target_tokens & tb
    sim = len(inter) / len(target_tokens | tb)
    is_match = len(inter) >= _DESC_MATCH_MIN_SHARED and sim >= _DESC_MATCH_MIN_SIM
    return is_match, sim


def similar_plates(target: str, candidates, target_desc: str | None = None,
                   max_dist: float | None = None) -> list[dict]:
    """
    Candidates likely to be the same car as `target`, best first.

    `candidates` may be plain plate strings or dicts with 'plate' and an optional
    'user_description'/'description'. When descriptions are present, a candidate whose
    appearance matches the target is ranked higher and allowed at a looser plate distance
    (since the matching make/model/colour corroborates that the OCR just misread the plate).

    Returns dicts: {plate, dist, description, desc_sim, desc_match}.
    """
    md = config.ALIAS_SUGGEST_MAXDIST if max_dist is None else max_dist
    tgt_tokens = _desc_tokens(target_desc)

    out: list[dict] = []
    for c in candidates:
        if isinstance(c, dict):
            plate = c.get("plate")
            desc  = c.get("user_description") or c.get("description")
        else:
            plate, desc = c, None
        if not plate or plate == target:
            continue

        d = _soft_dist(target, plate)
        matched, sim = _desc_match(tgt_tokens, desc)
        # Include if the plate is close, OR the appearance matches and the plate is
        # within a looser bound (don't suggest wildly different plates on looks alone).
        if d <= md or (matched and d <= md + _DESC_DIST_SLACK):
            out.append({
                "plate": plate,
                "dist": round(d, 2),
                "description": desc,
                "desc_sim": round(sim, 2),
                "desc_match": matched,
            })

    # Rank by plate distance, with an appearance match pulling a candidate up the list.
    out.sort(key=lambda x: x["dist"] - (_DESC_RANK_BONUS if x["desc_match"] else 0.0))
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


# ── Recompute a vehicle's plate from every image across all its passes ────────
# Combining many independent ANPR reads beats any single read: low-confidence or
# partially-occluded plates often agree once aggregated. We score whole-string reads by
# summed confidence and also build a per-position character-vote consensus, then suggest
# the better-supported of the two for the user to confirm.

def _collect_reads_sync(pass_row: dict) -> list[tuple[str, float]]:
    """Run ANPR over every stored image of one pass and return all plate reads."""
    import cv2
    from . import detect as detect_mod, alpr as alpr_mod
    reads: list[tuple[str, float]] = []

    # The car crop is already a tight car image — OCR it whole.
    crop_rel = pass_row.get("car_crop_path")
    if crop_rel:
        p = Path(config.DATA_DIR) / crop_rel
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                h, w = img.shape[:2]
                reads += alpr_mod.read_plates(img, (0, 0, w, h))

    # The raw full-frame stills — detect cars first, then OCR each box.
    for key in ("entry_frame_raw_path", "exit_frame_raw_path", "entry_frame_path", "exit_frame_path"):
        rel = pass_row.get(key)
        if not rel:
            continue
        p = Path(config.DATA_DIR) / rel
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        for d in detect_mod.run(img):
            reads += alpr_mod.read_plates(img, d.bbox_xyxy)
    return reads


def _aggregate_plate_reads(reads: list[tuple[str, float]]) -> dict | None:
    """Combine many (plate, confidence) reads into a single suggestion.
    Returns {suggested, consensus, candidates[], num_reads} or None if there are no reads."""
    if not reads:
        return None
    from collections import defaultdict
    score: dict[str, float] = defaultdict(float)
    count: dict[str, int]   = defaultdict(int)
    maxc:  dict[str, float] = defaultdict(float)
    for text, conf in reads:
        score[text] += conf
        count[text] += 1
        maxc[text]   = max(maxc[text], conf)
    candidates = [{"plate": t, "count": count[t],
                   "total_conf": round(score[t], 3), "max_conf": round(maxc[t], 3)}
                  for t in score]
    candidates.sort(key=lambda c: (-c["total_conf"], -c["max_conf"]))

    # Per-position character vote among reads of the dominant (most-confidently-seen) length.
    len_score: dict[int, float] = defaultdict(float)
    for text, conf in reads:
        len_score[len(text)] += conf
    dom_len = max(len_score, key=lambda L: len_score[L])
    same_len = [(t, c) for t, c in reads if len(t) == dom_len]
    consensus = None
    if same_len:
        cols = [defaultdict(float) for _ in range(dom_len)]
        for text, conf in same_len:
            for i, ch in enumerate(text):
                cols[i][ch] += conf
        consensus = "".join(max(col, key=lambda k: col[k]) for col in cols)

    # Prefer the character-vote consensus when several reads of that length back it;
    # otherwise fall back to the single best-scoring whole-string read.
    if consensus and len(same_len) >= 2:
        suggested = consensus
    else:
        suggested = candidates[0]["plate"]

    return {"suggested": suggested, "consensus": consensus,
            "candidates": candidates[:8], "num_reads": len(reads)}


async def recompute_plate(plate: str) -> dict:
    """Re-read the plate for a vehicle by running ANPR over every image of every pass and
    combining the results. Returns a suggestion dict (does not apply it — the caller
    confirms and applies via the merge flow)."""
    rows = await db.list_passes_for_plate(plate)
    loop = asyncio.get_event_loop()
    all_reads: list[tuple[str, float]] = []
    for row in rows:
        try:
            all_reads += await loop.run_in_executor(None, _collect_reads_sync, row)
        except Exception as e:
            log.warning("recompute_plate: reads for pass %s failed: %s", row.get("id"), e)
    agg = _aggregate_plate_reads(all_reads)
    if agg is None:
        return {"suggested": None, "consensus": None, "candidates": [],
                "num_reads": 0, "num_passes": len(rows), "current": plate}
    agg["num_passes"] = len(rows)
    agg["current"] = plate
    log.info("recompute_plate %s → %s (%d reads, %d passes)",
             plate, agg["suggested"], agg["num_reads"], len(rows))
    return agg


# ── Vehicle Re-ID embeddings: backfill, clustering, visual similarity ─────────

def _embed_crop_sync(crop_rel: str):
    import cv2
    p = Path(config.DATA_DIR) / crop_rel
    if not p.exists():
        return None
    img = cv2.imread(str(p))
    if img is None:
        return None
    return embed_mod.embed_bgr(img)


async def backfill_embeddings(limit: int | None = None) -> dict:
    """Compute Re-ID embeddings for passes that have a car crop but no embedding yet."""
    if not embed_mod.available():
        return {"error": "Re-ID model not installed", "updated": 0, "total": 0}
    rows = await db.passes_needing_embedding()
    if limit:
        rows = rows[:limit]
    loop = asyncio.get_event_loop()
    updated = 0
    for row in rows:
        crop = row.get("car_crop_path")
        if not crop:
            continue
        try:
            vec = await loop.run_in_executor(None, _embed_crop_sync, crop)
        except Exception as e:
            log.warning("embed backfill pass %s failed: %s", row.get("id"), e)
            continue
        if vec is None:
            continue
        await db.set_pass_embedding(row["id"], embed_mod.to_blob(vec))
        updated += 1
    result = {"updated": updated, "total": len(rows)}
    log.info("backfill embeddings: %s", result)
    return result


# Floor for *showing* a visual candidate. Lower than REID_SIM_THRESHOLD (the "likely same
# car" bar) so the closest look-alikes are always offered for review; callers flag which
# clear the threshold as strong matches.
_VISUAL_FLOOR = 0.4


async def _vehicle_means() -> dict[str, "np.ndarray"]:
    """Per-vehicle mean Re-ID embedding, keyed by canonical plate."""
    rows = await db.embedded_vehicle_passes()
    from collections import defaultdict
    vecs: dict[str, list] = defaultdict(list)
    for r in rows:
        if r.get("plate") and r.get("embedding") is not None:
            vecs[r["plate"]].append(embed_mod.from_blob(r["embedding"]))
    return {pl: embed_mod.mean_vector(vs) for pl, vs in vecs.items()}


def _rank_against_means(target, means: dict, exclude: str | None = None,
                        max_results: int = 6, floor: float = _VISUAL_FLOOR) -> list[tuple[str, float]]:
    """Closest vehicle means to `target` (cosine), above `floor`, closest first."""
    if target is None:
        return []
    sims = [(pl, embed_mod.cosine(target, m)) for pl, m in means.items()
            if pl != exclude and m is not None]
    sims = [(pl, round(s, 3)) for pl, s in sims if s >= floor]
    sims.sort(key=lambda x: -x[1])
    return sims[:max_results]


async def _attach_rep_images(ranked: list[tuple[str, float]]) -> list[dict]:
    """Turn (plate, sim) pairs into display dicts with a representative image + strong flag."""
    if not ranked:
        return []
    repmap = {v["plate"]: v["rep_image_path"] for v in await db.list_vehicles(min_passes=1)}
    return [{"plate": pl, "sim": s, "rep_image": repmap.get(pl),
             "strong": s >= config.REID_SIM_THRESHOLD} for pl, s in ranked]


async def unidentified_clusters() -> list[dict]:
    """Group plateless passes (no effective plate) into visually-similar clusters for review.
    Each cluster also carries `similar`: the closest existing vehicles by appearance, so a
    cluster can be merged into an existing vehicle (not only promoted to a new one)."""
    if not embed_mod.available():
        return []
    rows = await db.plateless_embedded_passes()
    if not rows:
        return []
    items = [(r["id"], embed_mod.from_blob(r["embedding"])) for r in rows]
    vec_by_id = {i: v for i, v in items}
    by_id = {r["id"]: r for r in rows}
    groups = embed_mod.cluster(items)
    means = await _vehicle_means()
    out = []
    for members in groups:
        passes = [by_id[i] for i in members]
        rep = next((p.get("car_crop_path") or p.get("entry_frame_path")
                    for p in passes if p.get("car_crop_path") or p.get("entry_frame_path")), None)
        cmean = embed_mod.mean_vector([vec_by_id[i] for i in members])
        similar = await _attach_rep_images(_rank_against_means(cmean, means, max_results=4))
        out.append({
            "pass_ids": members,
            "size": len(members),
            "passes": passes,
            "top_speed": max((p.get("sw_speed_mph") or 0) for p in passes),
            "rep_crop": rep,
            "similar": similar,
        })
    return out


async def visual_similar_vehicles(target_plate: str, max_results: int = 6,
                                  min_sim: float | None = None) -> list[tuple[str, float]]:
    """Identified vehicles whose mean Re-ID embedding is closest to the target's. Returns the
    top look-alikes above a low discovery floor (not the strict same-car threshold), closest
    first, as (plate, cosine_sim) — the caller flags which clear REID_SIM_THRESHOLD."""
    if not embed_mod.available():
        return []
    means = await _vehicle_means()
    floor = _VISUAL_FLOOR if min_sim is None else min_sim
    return _rank_against_means(means.get(target_plate), means, exclude=target_plate,
                               max_results=max_results, floor=floor)


async def similar_vehicles_for_pass(pass_id: int, max_results: int = 6) -> list[dict]:
    """Existing vehicles whose appearance best matches this single pass's car-crop embedding.
    Returns [{plate, sim, rep_image, strong}] (closest first), or [] if the pass has no
    embedding or the Re-ID model is absent. Excludes the pass's own current vehicle."""
    if not embed_mod.available():
        return []
    p = await db.get_pass(pass_id)
    if not p or p.get("embedding") is None:
        return []
    vec = embed_mod.from_blob(p["embedding"])
    raw = p.get("assigned_plate") or p.get("plate")
    eff = await db.canonical_plate(raw) if raw else None
    means = await _vehicle_means()
    ranked = _rank_against_means(vec, means, exclude=eff, max_results=max_results)
    return await _attach_rep_images(ranked)
