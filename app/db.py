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
    user_label               TEXT,
    user_corrected_mph       REAL,
    user_notes               TEXT,
    created_at               REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pass_session    ON pass(session_id);
CREATE INDEX IF NOT EXISTS idx_pass_label      ON pass(user_label);
CREATE INDEX IF NOT EXISTS idx_pass_confidence ON pass(confidence);
CREATE INDEX IF NOT EXISTS idx_pass_speed      ON pass(sw_speed_mph DESC);
"""


async def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.executescript(_SCHEMA)
        # Migrate existing tables
        for col in ("hq_frame_path", "hq_exit_frame_path", "plate"):
            try:
                await conn.execute(f"ALTER TABLE pass ADD COLUMN {col} TEXT")
            except Exception:
                pass
        try:
            await conn.execute("ALTER TABLE pass ADD COLUMN plate_confidence REAL")
        except Exception:
            pass
        for col, default in (("frame_w", 1280), ("frame_h", 720)):
            try:
                await conn.execute(
                    f"ALTER TABLE calibration ADD COLUMN {col} INTEGER NOT NULL DEFAULT {default}"
                )
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
                active, notes, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,1,?,?)""",
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
) -> list[dict]:
    conditions, params = [], []
    if label:
        conditions.append("p.user_label=?")
        params.append(label)
    if unlabelled_only:
        conditions.append("p.user_label IS NULL")
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    order_sql = {
        "created_at":   "p.created_at DESC",
        "sw_speed_mph": "p.sw_speed_mph DESC",
        "confidence":   "p.confidence ASC",
    }.get(order_by, "p.created_at DESC")
    sql = f"""
        SELECT p.*, s.start_ts, s.camera, s.clip_path as session_clip_path
        FROM pass p
        JOIN pass_session s ON s.id = p.session_id
        {where}
        ORDER BY {order_sql}
        LIMIT ? OFFSET ?
    """
    params += [limit, offset]
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(sql, params)
        return [dict(r) for r in await cur.fetchall()]


async def count_passes(label: str | None = None, unlabelled_only: bool = False) -> int:
    conditions, params = [], []
    if label:
        conditions.append("user_label=?")
        params.append(label)
    if unlabelled_only:
        conditions.append("user_label IS NULL")
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    async with aiosqlite.connect(DB_PATH) as conn:
        cur = await conn.execute(f"SELECT COUNT(*) FROM pass {where}", params)
        row = await cur.fetchone()
        return row[0] if row else 0


async def update_pass_label(pass_id: int, label: str | None, notes: str | None, corrected_mph: float | None):
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            "UPDATE pass SET user_label=?, user_notes=?, user_corrected_mph=? WHERE id=?",
            (label, notes, corrected_mph, pass_id),
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
