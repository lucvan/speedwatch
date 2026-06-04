"""
Fresh SQLite schema.  No Frigate/MQTT tables.
Tables: calibration, pass_session, pass.
"""
from __future__ import annotations
import aiosqlite
import json
import time
from pathlib import Path

from . import config

DB_PATH = Path(config.DATA_DIR) / "speedwatch.db"

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS calibration (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    camera           TEXT    NOT NULL,
    zone_pixels      TEXT    NOT NULL,
    edge_distances_m TEXT    NOT NULL,
    diagonal_m       REAL    NOT NULL,
    world_coords     TEXT    NOT NULL,
    homography       TEXT    NOT NULL,
    road_axis_deg    REAL    NOT NULL,
    residual_max_px  REAL    NOT NULL,
    frame_w          INTEGER NOT NULL DEFAULT 1280,
    frame_h          INTEGER NOT NULL DEFAULT 720,
    camera_matrix    TEXT,
    dist_coeffs      TEXT,
    intrinsics_rms   REAL,
    active           INTEGER NOT NULL DEFAULT 0,
    notes            TEXT,
    created_at       REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS pass_session (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    camera           TEXT    NOT NULL,
    start_ts         REAL    NOT NULL,
    end_ts           REAL,
    frames_seen      INTEGER NOT NULL DEFAULT 0,
    status           TEXT    NOT NULL DEFAULT 'processing',  -- processing | done | empty
    calibration_id   INTEGER REFERENCES calibration(id),
    clip_path        TEXT,
    created_at       REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session_start ON pass_session(start_ts DESC);

CREATE TABLE IF NOT EXISTS pass (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id               INTEGER NOT NULL REFERENCES pass_session(id) ON DELETE CASCADE,
    track_id                 INTEGER NOT NULL,
    sw_speed_mph             REAL    NOT NULL,
    sw_speed_mps             REAL    NOT NULL,
    frames_in_zone           INTEGER NOT NULL,
    dwell_seconds            REAL    NOT NULL,
    median_heading_deg       REAL    NOT NULL,
    per_frame_speed_cv       REAL    NOT NULL,
    id_switches              INTEGER NOT NULL DEFAULT 0,
    bbox_area_cv             REAL    NOT NULL,
    net_displacement_m       REAL    NOT NULL,
    confidence               REAL    NOT NULL,
    entry_frame_path         TEXT,
    exit_frame_path          TEXT,
    entry_frame_raw_path     TEXT,
    exit_frame_raw_path      TEXT,
    trajectory_overlay_path  TEXT,
    clip_path                TEXT,
    hq_frame_path            TEXT,
    hq_exit_frame_path       TEXT,
    plate                    TEXT,
    plate_confidence         REAL,
    car_crop_path            TEXT,
    user_label               TEXT,
    user_corrected_mph       REAL,
    user_notes               TEXT,
    confirmed_speeder        INTEGER NOT NULL DEFAULT 0,
    assigned_plate           TEXT,
    embedding                BLOB,
    created_at               REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pass_session    ON pass(session_id);
CREATE INDEX IF NOT EXISTS idx_pass_label      ON pass(user_label);
CREATE INDEX IF NOT EXISTS idx_pass_confidence ON pass(confidence);
CREATE INDEX IF NOT EXISTS idx_pass_speed      ON pass(sw_speed_mph DESC);
CREATE INDEX IF NOT EXISTS idx_pass_plate      ON pass(plate);

CREATE TABLE IF NOT EXISTS vehicle (
    plate            TEXT PRIMARY KEY,
    description      TEXT,        -- AI vision description (make/colour/body)
    described_at     REAL,        -- when the AI description was generated
    user_description TEXT,        -- manually entered description
    user_notes       TEXT,        -- free-form notes / metadata
    rep_image_path   TEXT,        -- representative car crop (best plate read)
    rep_plate_conf   REAL,        -- plate confidence of the rep image's pass
    confirmed_speeder INTEGER NOT NULL DEFAULT 0,  -- user-flagged: all passes count as evidence
    created_at       REAL NOT NULL
);

-- Maps an OCR-variant plate to its canonical plate so passes group as one vehicle.
-- Resolved at query time (raw pass.plate is never rewritten — merges are reversible).
CREATE TABLE IF NOT EXISTS plate_alias (
    alias      TEXT PRIMARY KEY,
    canonical  TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alias_canonical ON plate_alias(canonical);
"""


async def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.executescript(_SCHEMA)
        # Migrate existing tables
        for col in ("hq_frame_path", "hq_exit_frame_path", "plate", "car_crop_path"):
            try:
                await conn.execute(f"ALTER TABLE pass ADD COLUMN {col} TEXT")
            except Exception:
                pass
        try:
            await conn.execute("ALTER TABLE pass ADD COLUMN plate_confidence REAL")
        except Exception:
            pass
        # Confirmed-speeder evidence flags (pass-level and vehicle-level).
        try:
            await conn.execute("ALTER TABLE pass ADD COLUMN confirmed_speeder INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass
        try:
            await conn.execute("ALTER TABLE vehicle ADD COLUMN confirmed_speeder INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass
        # Manual vehicle assignment: attach a plateless / mis-read pass to a vehicle by plate.
        try:
            await conn.execute("ALTER TABLE pass ADD COLUMN assigned_plate TEXT")
        except Exception:
            pass
        # Vehicle Re-ID image embedding (float32 vector as BLOB) for visual grouping.
        try:
            await conn.execute("ALTER TABLE pass ADD COLUMN embedding BLOB")
        except Exception:
            pass
        for col, default in (("frame_w", 1280), ("frame_h", 720)):
            try:
                await conn.execute(
                    f"ALTER TABLE calibration ADD COLUMN {col} INTEGER NOT NULL DEFAULT {default}"
                )
            except Exception:
                pass
        # Lens intrinsics baked into a calibration (added with distortion-correction support).
        for col in ("camera_matrix", "dist_coeffs"):
            try:
                await conn.execute(f"ALTER TABLE calibration ADD COLUMN {col} TEXT")
            except Exception:
                pass
        try:
            await conn.execute("ALTER TABLE calibration ADD COLUMN intrinsics_rms REAL")
        except Exception:
            pass
        # Recompute homography for degenerate quads (pt3_y < 0.5)
        import json as _json
        rows = await (await conn.execute(
            "SELECT id, zone_pixels, edge_distances_m, diagonal_m, world_coords, frame_w, frame_h FROM calibration"
        )).fetchall()
        for row in rows:
            wc = _json.loads(row[4])
            if wc[3][1] < 0.5:
                try:
                    from . import calibration as _cal
                    zp   = _json.loads(row[1])
                    ed   = _json.loads(row[2])
                    diag = row[3]
                    fw, fh = row[5] or 1280, row[6] or 720
                    new_world = _cal.reconstruct_quad(ed, diag)
                    H_new, res = _cal.compute_homography(zp, new_world, fw, fh)
                    await conn.execute(
                        "UPDATE calibration SET world_coords=?, homography=?, residual_max_px=? WHERE id=?",
                        (_json.dumps(new_world), _json.dumps(H_new.tolist()), res, row[0]),
                    )
                except Exception:
                    pass
        await conn.commit()


# ── calibration ────────────────────────────────────────────────────────────────

async def save_calibration(cal: dict) -> int:
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute("UPDATE calibration SET active=0 WHERE camera=?", (cal["camera"],))
        cur = await conn.execute(
            """INSERT INTO calibration
               (camera, zone_pixels, edge_distances_m, diagonal_m, world_coords,
                homography, road_axis_deg, residual_max_px, frame_w, frame_h,
                camera_matrix, dist_coeffs, intrinsics_rms,
                active, notes, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,?)""",
            (
                cal["camera"],
                json.dumps(cal["zone_pixels"]),
                json.dumps(cal["edge_distances_m"]),
                cal["diagonal_m"],
                json.dumps(cal["world_coords"]),
                json.dumps(cal["homography"]),
                cal["road_axis_deg"],
                cal["residual_max_px"],
                cal.get("frame_w", 1280),
                cal.get("frame_h", 720),
                json.dumps(cal["camera_matrix"]) if cal.get("camera_matrix") else None,
                json.dumps(cal["dist_coeffs"]) if cal.get("dist_coeffs") else None,
                cal.get("intrinsics_rms"),
                cal.get("notes", ""),
                now,
            ),
        )
        cal_id = cur.lastrowid
        await conn.commit()
    return cal_id


def _decode_cal_row(d: dict) -> dict:
    d["zone_pixels"]      = json.loads(d["zone_pixels"])
    d["edge_distances_m"] = json.loads(d["edge_distances_m"])
    d["world_coords"]     = json.loads(d["world_coords"])
    d["homography"]       = json.loads(d["homography"])
    d["camera_matrix"]    = json.loads(d["camera_matrix"]) if d.get("camera_matrix") else None
    d["dist_coeffs"]      = json.loads(d["dist_coeffs"]) if d.get("dist_coeffs") else None
    return d


async def get_active_calibration(camera: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            "SELECT * FROM calibration WHERE camera=? AND active=1 ORDER BY created_at DESC LIMIT 1",
            (camera,),
        )
        row = await cur.fetchone()
        return _decode_cal_row(dict(row)) if row else None


async def list_calibrations(camera: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            "SELECT * FROM calibration WHERE camera=? ORDER BY created_at DESC",
            (camera,),
        )
        return [_decode_cal_row(dict(r)) for r in await cur.fetchall()]


# ── pass_session ───────────────────────────────────────────────────────────────

async def create_session(camera: str, start_ts: float, calibration_id: int | None) -> int:
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as conn:
        cur = await conn.execute(
            "INSERT INTO pass_session (camera, start_ts, calibration_id, status, created_at) VALUES (?,?,?,?,?)",
            (camera, start_ts, calibration_id, "processing", now),
        )
        sid = cur.lastrowid
        await conn.commit()
    return sid


async def finalize_session(session_id: int, end_ts: float, frames_seen: int,
                           status: str, clip_path: str | None):
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            "UPDATE pass_session SET end_ts=?, frames_seen=?, status=?, clip_path=? WHERE id=?",
            (end_ts, frames_seen, status, clip_path, session_id),
        )
        await conn.commit()


async def list_sessions(limit: int = 50, offset: int = 0) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            "SELECT * FROM pass_session ORDER BY start_ts DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [dict(r) for r in await cur.fetchall()]


async def count_sessions() -> int:
    async with aiosqlite.connect(DB_PATH) as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM pass_session")
        row = await cur.fetchone()
        return row[0] if row else 0


# ── pass ───────────────────────────────────────────────────────────────────────

async def insert_pass(p: dict) -> int:
    p.setdefault("created_at", time.time())
    cols = list(p.keys())
    placeholders = ", ".join("?" for _ in cols)
    sql = f"INSERT INTO pass ({', '.join(cols)}) VALUES ({placeholders})"
    async with aiosqlite.connect(DB_PATH) as conn:
        cur = await conn.execute(sql, [p[c] for c in cols])
        await conn.commit()
        return cur.lastrowid


async def list_passes(
    limit: int = 50,
    offset: int = 0,
    label: str | None = None,
    unlabelled_only: bool = False,
    order_by: str = "created_at",
    min_mph: float | None = None,
    plateless: bool = False,
) -> list[dict]:
    conditions, params = [], []
    if label:
        conditions.append("p.user_label=?")
        params.append(label)
    if unlabelled_only:
        conditions.append("p.user_label IS NULL")
    if min_mph is not None:
        conditions.append("p.sw_speed_mph >= ?")
        params.append(min_mph)
    if plateless:
        conditions.append(f"({_EFF} IS NULL OR {_EFF} = '')")
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    order_sql = {
        "created_at":   "p.created_at DESC",
        "sw_speed_mph": "p.sw_speed_mph DESC",
        "confidence":   "p.confidence ASC",
    }.get(order_by, "p.created_at DESC")
    sql = f"""
        SELECT p.*, s.start_ts, s.camera, s.clip_path as session_clip_path,
               {_EFF} AS eff_plate,
               v.user_description AS veh_user_description,
               v.description      AS veh_description
        FROM pass p
        JOIN pass_session s ON s.id = p.session_id
        LEFT JOIN vehicle v ON v.plate = {_EFF}
        {where}
        ORDER BY {order_sql}
        LIMIT ? OFFSET ?
    """
    params += [limit, offset]
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(sql, params)
        return [dict(r) for r in await cur.fetchall()]


async def count_passes(label: str | None = None, unlabelled_only: bool = False,
                       min_mph: float | None = None, plateless: bool = False) -> int:
    conditions, params = [], []
    if label:
        conditions.append("p.user_label=?")
        params.append(label)
    if unlabelled_only:
        conditions.append("p.user_label IS NULL")
    if min_mph is not None:
        conditions.append("p.sw_speed_mph >= ?")
        params.append(min_mph)
    if plateless:
        conditions.append(f"({_EFF} IS NULL OR {_EFF} = '')")
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    async with aiosqlite.connect(DB_PATH) as conn:
        cur = await conn.execute(f"SELECT COUNT(*) FROM pass p {where}", params)
        row = await cur.fetchone()
        return row[0] if row else 0


async def update_pass_label(pass_id: int, label: str | None, notes: str | None, corrected_mph: float | None):
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            "UPDATE pass SET user_label=?, user_notes=?, user_corrected_mph=? WHERE id=?",
            (label, notes, corrected_mph, pass_id),
        )
        await conn.commit()


async def set_pass_confirmed(pass_id: int, confirmed: bool) -> None:
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            "UPDATE pass SET confirmed_speeder=? WHERE id=?",
            (1 if confirmed else 0, pass_id),
        )
        await conn.commit()


async def get_pass(pass_id: int) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute("SELECT * FROM pass WHERE id=?", (pass_id,))
        row = await cur.fetchone()
        return dict(row) if row else None


async def delete_session(session_id: int) -> tuple[dict, list[dict]]:
    """Delete session + cascade passes. Returns (session_row, pass_rows) for file cleanup."""
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute("SELECT * FROM pass_session WHERE id=?", (session_id,))
        session_row = await cur.fetchone()
        session = dict(session_row) if session_row else {}
        cur = await conn.execute("SELECT * FROM pass WHERE session_id=?", (session_id,))
        passes = [dict(r) for r in await cur.fetchall()]
        await conn.execute("DELETE FROM pass_session WHERE id=?", (session_id,))
        await conn.commit()
    return session, passes


async def delete_pass(pass_id: int) -> dict | None:
    """Delete one pass row. Returns the row for file cleanup, or None if not found."""
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute("SELECT * FROM pass WHERE id=?", (pass_id,))
        row = await cur.fetchone()
        if row is None:
            return None
        p = dict(row)
        await conn.execute("DELETE FROM pass WHERE id=?", (pass_id,))
        await conn.commit()
    return p


# ── vehicles (grouped by plate) ──────────────────────────────────────────────

# Leaderboard / per-plate stats are computed live from the pass table; the vehicle
# table only holds enrichment (AI + manual descriptions, representative image).

def _eff_plate(alias: str = "p") -> str:
    """SQL expression for a pass's effective (canonical) plate. A manual `assigned_plate`
    wins (resolved through any alias), then the OCR plate's alias, then the raw OCR plate.
    This is what lets plateless / mis-read passes be attached to a vehicle."""
    return (
        "COALESCE("
        f"(SELECT ap.canonical FROM plate_alias ap WHERE ap.alias = {alias}.assigned_plate),"
        f"{alias}.assigned_plate,"
        f"(SELECT op.canonical FROM plate_alias op WHERE op.alias = {alias}.plate),"
        f"{alias}.plate)"
    )


_EFF = _eff_plate("p")

_VEHICLE_STATS_SQL = f"""
    SELECT {_EFF}                         AS plate,
           COUNT(*)                       AS passes,
           MAX(p.sw_speed_mph)            AS max_mph,
           AVG(p.sw_speed_mph)            AS avg_mph,
           MIN(p.sw_speed_mph)            AS min_mph,
           MAX(s.start_ts)                AS last_seen,
           MIN(s.start_ts)                AS first_seen
    FROM pass p JOIN pass_session s ON s.id = p.session_id
    WHERE {_EFF} IS NOT NULL AND {_EFF} <> ''
    GROUP BY {_EFF}
"""


async def list_vehicles(order_by: str = "max_mph", min_passes: int = 1) -> list[dict]:
    order_sql = {
        "max_mph":  "max_mph DESC",
        "avg_mph":  "avg_mph DESC",
        "passes":   "passes DESC",
        "last_seen": "last_seen DESC",
    }.get(order_by, "max_mph DESC")
    sql = f"""
        SELECT st.*, v.description, v.user_description, v.user_notes, v.confirmed_speeder,
               COALESCE(v.rep_image_path, (
                   SELECT pc.car_crop_path FROM pass pc
                   WHERE {_eff_plate('pc')} = st.plate
                     AND pc.car_crop_path IS NOT NULL AND pc.car_crop_path <> ''
                   ORDER BY COALESCE(pc.plate_confidence, 0) DESC LIMIT 1
               )) AS rep_image_path
        FROM ( {_VEHICLE_STATS_SQL} ) st
        LEFT JOIN vehicle v ON v.plate = st.plate
        WHERE st.passes >= ?
        ORDER BY {order_sql}
    """
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(sql, (min_passes,))
        return [dict(r) for r in await cur.fetchall()]


async def get_vehicle(plate: str) -> dict | None:
    """Stats + enrichment for one plate. None if no passes carry that plate."""
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            f"SELECT * FROM ( {_VEHICLE_STATS_SQL} ) WHERE plate = ?", (plate,)
        )
        st = await cur.fetchone()
        if st is None:
            return None
        v = dict(st)
        cur = await conn.execute("SELECT * FROM vehicle WHERE plate=?", (plate,))
        meta = await cur.fetchone()
        if meta:
            v.update({k: meta[k] for k in meta.keys() if k != "plate"})
        # Fall back to a representative pass crop if no rep image is stored (e.g. a vehicle
        # created by a merge-to-new-plate, or one built only from assigned plateless passes).
        if not v.get("rep_image_path"):
            cur = await conn.execute(
                f"""SELECT pc.car_crop_path FROM pass pc
                    WHERE {_eff_plate('pc')} = ?
                      AND pc.car_crop_path IS NOT NULL AND pc.car_crop_path <> ''
                    ORDER BY COALESCE(pc.plate_confidence, 0) DESC LIMIT 1""",
                (plate,),
            )
            crop = await cur.fetchone()
            if crop:
                v["rep_image_path"] = crop[0]
        return v


async def list_passes_for_plate(plate: str) -> list[dict]:
    """All passes whose canonical plate is `plate` (includes merged aliases)."""
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            f"""SELECT p.*, s.start_ts, s.clip_path AS session_clip_path
                FROM pass p JOIN pass_session s ON s.id = p.session_id
                WHERE {_EFF} = ? ORDER BY s.start_ts DESC""",
            (plate,),
        )
        return [dict(r) for r in await cur.fetchall()]


async def canonical_plate(plate: str) -> str:
    async with aiosqlite.connect(DB_PATH) as conn:
        cur = await conn.execute("SELECT canonical FROM plate_alias WHERE alias=?", (plate,))
        row = await cur.fetchone()
        return row[0] if row else plate


async def aliases_of(canonical: str) -> list[str]:
    async with aiosqlite.connect(DB_PATH) as conn:
        cur = await conn.execute("SELECT alias FROM plate_alias WHERE canonical=? ORDER BY alias", (canonical,))
        return [r[0] for r in await cur.fetchall()]


async def add_plate_aliases(canonical: str, aliases: list[str]) -> str:
    """Point each alias plate at `canonical` (resolving chains). Returns the root canonical."""
    async with aiosqlite.connect(DB_PATH) as conn:
        cur = await conn.execute("SELECT canonical FROM plate_alias WHERE alias=?", (canonical,))
        r = await cur.fetchone()
        root = r[0] if r else canonical          # if canonical is itself aliased, use its root
        now = time.time()
        for a in aliases:
            if not a or a == root:
                continue
            # repoint anything currently pointing at `a` to the root, then alias a→root
            await conn.execute("UPDATE plate_alias SET canonical=? WHERE canonical=?", (root, a))
            await conn.execute(
                "INSERT OR REPLACE INTO plate_alias (alias, canonical, created_at) VALUES (?,?,?)",
                (a, root, now),
            )
        # the root must not be an alias of anything
        await conn.execute("DELETE FROM plate_alias WHERE alias=?", (root,))
        await conn.commit()
    await ensure_vehicle(root)
    # A merge to a brand-new corrected plate starts with no rep image — inherit one from
    # the now-grouped passes so the merged vehicle still has a thumbnail.
    await ensure_vehicle_rep(root)
    return root


async def remove_plate_alias(alias: str) -> None:
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute("DELETE FROM plate_alias WHERE alias=?", (alias,))
        await conn.commit()


async def unmerge_plate(canonical: str) -> None:
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute("DELETE FROM plate_alias WHERE canonical=?", (canonical,))
        await conn.commit()


async def list_distinct_plates() -> list[dict]:
    """Distinct canonical plates with pass counts + vehicle descriptions — for fuzzy
    match suggestions (plate similarity corroborated by appearance)."""
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            f"""SELECT {_EFF} AS plate, COUNT(*) AS passes,
                       v.description AS description, v.user_description AS user_description
                FROM pass p
                LEFT JOIN vehicle v ON v.plate = {_EFF}
                WHERE {_EFF} IS NOT NULL AND {_EFF} <> ''
                GROUP BY {_EFF}""",
        )
        return [dict(r) for r in await cur.fetchall()]


async def passes_missing_car_crop() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            """SELECT id, plate, plate_confidence, entry_frame_raw_path, exit_frame_raw_path
               FROM pass
               WHERE plate IS NOT NULL AND plate <> '' AND (car_crop_path IS NULL OR car_crop_path = '')"""
        )
        return [dict(r) for r in await cur.fetchall()]


async def set_pass_car_crop(pass_id: int, path: str) -> None:
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute("UPDATE pass SET car_crop_path=? WHERE id=?", (path, pass_id))
        await conn.commit()


async def vehicle_pass_count(plate: str) -> int:
    async with aiosqlite.connect(DB_PATH) as conn:
        cur = await conn.execute(
            "SELECT COUNT(*) FROM pass WHERE plate=?", (plate,)
        )
        row = await cur.fetchone()
        return row[0] if row else 0


async def ensure_vehicle(plate: str) -> None:
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO vehicle (plate, created_at) VALUES (?, ?)",
            (plate, time.time()),
        )
        await conn.commit()


async def update_vehicle_rep_image(plate: str, image_path: str, plate_conf: float) -> None:
    """Set the representative car image if this pass has the best plate confidence so far."""
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            """UPDATE vehicle SET rep_image_path=?, rep_plate_conf=?
               WHERE plate=? AND (rep_plate_conf IS NULL OR ? > rep_plate_conf)""",
            (image_path, plate_conf, plate, plate_conf),
        )
        await conn.commit()


async def ensure_vehicle_rep(plate: str) -> None:
    """Give a vehicle a representative image if it has none, derived from its own passes'
    car crops (best plate confidence wins). Used after a merge-to-new-plate or after a
    plateless pass is assigned, so the merged/assigned vehicle inherits a thumbnail."""
    await ensure_vehicle(plate)
    async with aiosqlite.connect(DB_PATH) as conn:
        cur = await conn.execute("SELECT rep_image_path FROM vehicle WHERE plate=?", (plate,))
        row = await cur.fetchone()
        if row and row[0]:
            return
        cur = await conn.execute(
            f"""SELECT pc.car_crop_path, pc.plate_confidence FROM pass pc
                WHERE {_eff_plate('pc')} = ?
                  AND pc.car_crop_path IS NOT NULL AND pc.car_crop_path <> ''
                ORDER BY COALESCE(pc.plate_confidence, 0) DESC LIMIT 1""",
            (plate,),
        )
        crop = await cur.fetchone()
        if crop:
            await conn.execute(
                "UPDATE vehicle SET rep_image_path=?, rep_plate_conf=COALESCE(rep_plate_conf, ?) WHERE plate=?",
                (crop[0], crop[1] or 0.0, plate),
            )
            await conn.commit()


async def assign_pass_to_vehicle(pass_id: int, plate: str) -> str:
    """Manually attach a pass (typically plateless or mis-read) to a vehicle by plate.
    Returns the canonical plate the pass now belongs to. Reversible via unassign_pass."""
    canon = await canonical_plate(plate.strip().upper())
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute("UPDATE pass SET assigned_plate=? WHERE id=?", (canon, pass_id))
        await conn.commit()
    await ensure_vehicle(canon)
    await ensure_vehicle_rep(canon)
    return canon


async def unassign_pass(pass_id: int) -> None:
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute("UPDATE pass SET assigned_plate=NULL WHERE id=?", (pass_id,))
        await conn.commit()


# ── Vehicle Re-ID embeddings (visual grouping) ───────────────────────────────

async def set_pass_embedding(pass_id: int, blob: bytes) -> None:
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute("UPDATE pass SET embedding=? WHERE id=?", (blob, pass_id))
        await conn.commit()


async def passes_needing_embedding() -> list[dict]:
    """Passes that have a car crop but no embedding yet (for backfill)."""
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            """SELECT id, car_crop_path, entry_frame_raw_path, exit_frame_raw_path
               FROM pass
               WHERE embedding IS NULL
                 AND car_crop_path IS NOT NULL AND car_crop_path <> ''"""
        )
        return [dict(r) for r in await cur.fetchall()]


async def plateless_embedded_passes() -> list[dict]:
    """Unidentified passes (no effective plate) that have an embedding — clustering input."""
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            f"""SELECT p.id, p.embedding, p.car_crop_path, p.entry_frame_path,
                       p.session_id, p.sw_speed_mph, p.confidence, s.start_ts
                FROM pass p JOIN pass_session s ON s.id = p.session_id
                WHERE {_EFF} IS NULL AND p.embedding IS NOT NULL
                ORDER BY s.start_ts DESC"""
        )
        return [dict(r) for r in await cur.fetchall()]


async def embedded_vehicle_passes() -> list[dict]:
    """(effective_plate, embedding) for every identified pass that has an embedding —
    used to build per-vehicle mean embeddings for visual-similarity suggestions."""
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            f"""SELECT {_EFF} AS plate, p.embedding
                FROM pass p
                WHERE {_EFF} IS NOT NULL AND p.embedding IS NOT NULL"""
        )
        return [dict(r) for r in await cur.fetchall()]


async def set_vehicle_description(plate: str, description: str) -> None:
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            "UPDATE vehicle SET description=?, described_at=? WHERE plate=?",
            (description, time.time(), plate),
        )
        await conn.commit()


async def update_vehicle_meta(plate: str, user_description: str | None,
                              user_notes: str | None) -> None:
    await ensure_vehicle(plate)
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            "UPDATE vehicle SET user_description=?, user_notes=? WHERE plate=?",
            (user_description, user_notes, plate),
        )
        await conn.commit()


async def get_vehicle_meta(plate: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute("SELECT * FROM vehicle WHERE plate=?", (plate,))
        row = await cur.fetchone()
        return dict(row) if row else None


# ── Confirmed-speeder evidence ───────────────────────────────────────────────

async def set_vehicle_confirmed(plate: str, confirmed: bool) -> None:
    """Flag a vehicle (by canonical plate) as a confirmed speeder. Every pass of that
    vehicle then counts as evidence, on top of any individually-confirmed passes."""
    await ensure_vehicle(plate)
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            "UPDATE vehicle SET confirmed_speeder=? WHERE plate=?",
            (1 if confirmed else 0, plate),
        )
        await conn.commit()


# A pass is evidence if it was flagged directly, OR its (canonical) vehicle is flagged.
_EVIDENCE_WHERE = """
    p.confirmed_speeder = 1
    OR EXISTS (
        SELECT 1 FROM vehicle v
        WHERE v.plate = COALESCE((SELECT canonical FROM plate_alias WHERE alias = p.plate), p.plate)
          AND v.confirmed_speeder = 1
    )
"""


async def list_evidence_passes() -> list[dict]:
    """All passes that count as confirmed-speeder evidence, newest first, with the data
    needed for display and for the export bundle."""
    sql = f"""
        SELECT p.*, s.start_ts, s.camera, s.clip_path AS session_clip_path,
               COALESCE((SELECT canonical FROM plate_alias WHERE alias = p.plate), p.plate) AS canonical_plate,
               p.confirmed_speeder AS pass_confirmed
        FROM pass p
        JOIN pass_session s ON s.id = p.session_id
        WHERE {_EVIDENCE_WHERE}
        ORDER BY s.start_ts DESC
    """
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(sql)
        return [dict(r) for r in await cur.fetchall()]


async def count_evidence_passes() -> int:
    sql = f"SELECT COUNT(*) FROM pass p WHERE {_EVIDENCE_WHERE}"
    async with aiosqlite.connect(DB_PATH) as conn:
        cur = await conn.execute(sql)
        row = await cur.fetchone()
        return row[0] if row else 0


# ── Dashboard ────────────────────────────────────────────────────────────────

async def dashboard_stats(since_ts: float, warn_mph: float, limit_mph: float) -> dict:
    """One round of aggregate counters for the landing dashboard. `since_ts` marks the
    start of "today" (local midnight, computed by the caller)."""
    async with aiosqlite.connect(DB_PATH) as conn:
        async def scalar(sql, params=()):
            cur = await conn.execute(sql, params)
            row = await cur.fetchone()
            return row[0] if row and row[0] is not None else 0

        total_passes   = await scalar("SELECT COUNT(*) FROM pass")
        passes_today   = await scalar(
            "SELECT COUNT(*) FROM pass p JOIN pass_session s ON s.id=p.session_id "
            "WHERE s.start_ts >= ?", (since_ts,))
        fastest_all    = await scalar("SELECT MAX(sw_speed_mph) FROM pass")
        fastest_today  = await scalar(
            "SELECT MAX(p.sw_speed_mph) FROM pass p JOIN pass_session s ON s.id=p.session_id "
            "WHERE s.start_ts >= ?", (since_ts,))
        over_today     = await scalar(
            "SELECT COUNT(*) FROM pass p JOIN pass_session s ON s.id=p.session_id "
            "WHERE s.start_ts >= ? AND p.sw_speed_mph > ?", (since_ts, limit_mph))
        vehicles_total = await scalar(
            f"SELECT COUNT(*) FROM ( {_VEHICLE_STATS_SQL} )")
        confirmed_veh  = await scalar(
            "SELECT COUNT(*) FROM vehicle WHERE confirmed_speeder = 1")
        evidence_total = await scalar(
            f"SELECT COUNT(*) FROM pass p WHERE {_EVIDENCE_WHERE}")
        # Work queues
        plateless      = await scalar(
            f"SELECT COUNT(*) FROM pass p WHERE {_EFF} IS NULL OR {_EFF} = ''")
        low_conf_unlbl = await scalar(
            "SELECT COUNT(*) FROM pass WHERE user_label IS NULL AND confidence < 0.4")
        needing_embed  = await scalar(
            "SELECT COUNT(*) FROM pass WHERE embedding IS NULL "
            "AND car_crop_path IS NOT NULL AND car_crop_path <> ''")
    return {
        "total_passes": total_passes,
        "passes_today": passes_today,
        "fastest_all": fastest_all or None,
        "fastest_today": fastest_today or None,
        "over_today": over_today,
        "vehicles_total": vehicles_total,
        "confirmed_vehicles": confirmed_veh,
        "evidence_total": evidence_total,
        "plateless": plateless,
        "low_conf_unlabelled": low_conf_unlbl,
        "needing_embedding": needing_embed,
    }


async def recent_fast_passes(limit: int = 8) -> list[dict]:
    """Most recent passes (newest first) with thumbnail + plate, for the dashboard strip."""
    sql = f"""
        SELECT p.id, p.session_id, p.sw_speed_mph, p.confidence,
               p.trajectory_overlay_path, p.entry_frame_path, p.car_crop_path,
               s.start_ts, {_EFF} AS eff_plate
        FROM pass p JOIN pass_session s ON s.id = p.session_id
        ORDER BY s.start_ts DESC
        LIMIT ?
    """
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(sql, (limit,))
        return [dict(r) for r in await cur.fetchall()]
