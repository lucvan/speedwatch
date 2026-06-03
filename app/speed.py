"""
Speed estimation: ByteTrack + homography qualification math.
Lifted from old pipeline.py and adapted for live-stream (not clip) use.

Call flow:
  tracker = LiveTracker(calibration, fps)
  for frame in pass_session:
      tracker.update(detections, frame_bgr, t_ms)
  results = tracker.finalize()  # -> list[PassResult]
"""
from __future__ import annotations
import dataclasses
import logging
import math
import statistics
from typing import NamedTuple

import cv2
import numpy as np

from . import calibration as cal_mod
from . import config
from .detect import Detection

log = logging.getLogger(__name__)

_MPH_PER_MPS = 2.23694


@dataclasses.dataclass
class PassResult:
    track_id: int
    sw_speed_mph: float
    sw_speed_mps: float
    frames_in_zone: int
    dwell_seconds: float
    median_heading_deg: float
    per_frame_speed_cv: float
    id_switches: int
    bbox_area_cv: float
    net_displacement_m: float
    confidence: float
    entry_t_ms: float = 0.0                     # t_ms of first in-zone frame (tracker-relative)
    exit_t_ms: float = 0.0                      # t_ms of last in-zone frame (tracker-relative)
    entry_bbox_xyxy: tuple = (0, 0, 0, 0)       # car bbox in the entry_frame_raw (for ALPR crop)
    exit_bbox_xyxy: tuple = (0, 0, 0, 0)        # car bbox in the exit_frame_raw (for ALPR crop)
    entry_frame: np.ndarray | None     = dataclasses.field(default=None, repr=False)
    exit_frame: np.ndarray | None      = dataclasses.field(default=None, repr=False)
    entry_frame_raw: np.ndarray | None = dataclasses.field(default=None, repr=False)
    exit_frame_raw: np.ndarray | None  = dataclasses.field(default=None, repr=False)
    trajectory_frame: np.ndarray | None = dataclasses.field(default=None, repr=False)


class _FrameRecord(NamedTuple):
    t_ms: float
    wx: float
    wy: float
    bx_center: float
    by_center: float
    bbox_xyxy: tuple
    bbox_area: float
    in_zone: bool


def _trimmed_mean(values: list[float], trim_pct: float = 0.20) -> float:
    if not values:
        return 0.0
    s   = sorted(values)
    cut = int(len(s) * trim_pct)
    trimmed = s[cut:-cut] if cut and len(s) > 2 * cut else s
    return sum(trimmed) / len(trimmed)


def _cv(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = statistics.mean(values)
    return 0.0 if m == 0 else statistics.stdev(values) / m


def _compute_confidence(
    frames_in_zone: int,
    dwell_seconds: float,
    per_frame_speed_cv: float,
    id_switches: int,
) -> float:
    frames_score = min(frames_in_zone / 20.0, 1.0)
    dwell_score  = min(dwell_seconds / 1.5, 1.0)
    cv_score     = max(0.0, 1.0 - min(per_frame_speed_cv, 1.0))
    switch_score = 1.0 if id_switches == 0 else max(0.0, 1.0 - id_switches * 0.25)
    return round((frames_score + dwell_score + cv_score + switch_score) / 4.0, 3)


class LiveTracker:
    """
    Accumulates per-frame detections for one pass session, then qualifies+scores
    on finalize().  Reusable: reset() for the next session.
    """

    def __init__(self, calibration: dict, fps: float):
        import supervision as sv
        self._cal  = calibration
        self._H    = cal_mod.homography_from_dict(calibration)
        self._zone = cal_mod.ZonePolygon(calibration["zone_pixels"])
        self._cal_w = int(calibration.get("frame_w", 1280))
        self._cal_h = int(calibration.get("frame_h", 720))
        self._fps   = fps

        self._tracker = sv.ByteTrack(
            frame_rate=max(1, int(round(fps))),
            track_activation_threshold=0.30,
            lost_track_buffer=max(2, int(fps * 2)),
            minimum_matching_threshold=0.80,
            minimum_consecutive_frames=2,
        )
        self._reset_state()

    def _reset_state(self):
        self._track_history: dict[int, list[_FrameRecord]] = {}
        self._id_switch_counts: dict[int, int] = {}
        self._entry_raw: dict[int, np.ndarray] = {}
        self._last_inzone: dict[int, np.ndarray] = {}
        self._frame_w: int | None = None
        self._frame_h: int | None = None

    def reset(self):
        import supervision as sv
        self._tracker.reset()
        self._reset_state()

    def update(self, detections: list[Detection], frame_bgr: np.ndarray, t_ms: float):
        import supervision as sv

        h, w = frame_bgr.shape[:2]
        if self._frame_w is None:
            self._frame_w, self._frame_h = w, h

        # Convert to supervision Detections
        if detections:
            xyxy      = np.array([[d.bbox_xyxy[0], d.bbox_xyxy[1], d.bbox_xyxy[2], d.bbox_xyxy[3]] for d in detections], dtype=np.float32)
            class_ids = np.array([d.class_id for d in detections], dtype=int)
            confs     = np.array([d.conf for d in detections], dtype=np.float32)
            sv_dets   = sv.Detections(xyxy=xyxy, class_id=class_ids, confidence=confs)
        else:
            sv_dets = sv.Detections.empty()

        sv_dets = self._tracker.update_with_detections(sv_dets)
        if sv_dets.tracker_id is None:
            return

        for i, tid in enumerate(sv_dets.tracker_id):
            tid = int(tid)
            bx1, by1, bx2, by2 = (float(v) for v in sv_dets.xyxy[i])
            bc_x      = (bx1 + bx2) / 2.0
            bc_y      = by2
            bbox_area = (bx2 - bx1) * (by2 - by1)
            in_zone   = self._zone.contains_px(bc_x, bc_y, w, h)
            wx, wy    = cal_mod.pixel_to_world(
                bc_x * self._cal_w / w,
                bc_y * self._cal_h / h,
                self._H,
            )
            rec = _FrameRecord(
                t_ms=t_ms, wx=wx, wy=wy,
                bx_center=bc_x, by_center=bc_y,
                bbox_xyxy=(bx1, by1, bx2, by2),
                bbox_area=bbox_area, in_zone=in_zone,
            )
            if tid not in self._track_history:
                self._track_history[tid] = []
                self._id_switch_counts[tid] = 0
            self._track_history[tid].append(rec)
            if in_zone:
                if tid not in self._entry_raw:
                    self._entry_raw[tid] = frame_bgr.copy()
                self._last_inzone[tid] = frame_bgr

    def finalize(self) -> list[PassResult]:
        exit_raw: dict[int, np.ndarray] = {
            tid: f.copy() for tid, f in self._last_inzone.items()
        }
        vid_w = self._frame_w or 1280
        vid_h = self._frame_h or 720
        passes: list[PassResult] = []

        for tid, history in self._track_history.items():
            in_zone_recs = [r for r in history if r.in_zone]

            if len(in_zone_recs) < 5:
                log.info("track %d: SKIP in_zone_frames=%d < 5", tid, len(in_zone_recs))
                continue

            dwell_s = (in_zone_recs[-1].t_ms - in_zone_recs[0].t_ms) / 1000.0
            if dwell_s < config.MIN_DT_SECONDS:
                log.info("track %d: SKIP dwell=%.3fs < %.3fs", tid, dwell_s, config.MIN_DT_SECONDS)
                continue

            bbox_cv = _cv([r.bbox_area for r in in_zone_recs])
            if bbox_cv > 0.6:
                log.info("track %d: SKIP bbox_area_cv=%.2f > 0.6", tid, bbox_cv)
                continue

            speeds_mps: list[float] = []
            headings: list[float] = []
            total_path = 0.0

            for j in range(1, len(in_zone_recs)):
                r0, r1 = in_zone_recs[j - 1], in_zone_recs[j]
                dt_s = (r1.t_ms - r0.t_ms) / 1000.0
                if dt_s <= 0:
                    continue
                dx   = r1.wx - r0.wx
                dy   = r1.wy - r0.wy
                dist = math.hypot(dx, dy)
                v    = dist / dt_s
                if 0.5 <= v <= 60.0:
                    speeds_mps.append(v)
                    headings.append(math.degrees(math.atan2(dy, dx)))
                    total_path += dist

            if len(speeds_mps) < 3:
                log.info("track %d: SKIP only %d valid speed samples", tid, len(speeds_mps))
                continue

            median_heading = statistics.median(headings)
            heading_std    = statistics.stdev(headings)
            if heading_std > 60.0:
                log.info("track %d: SKIP heading_std=%.1f° > 60°", tid, heading_std)
                continue

            net_disp = math.hypot(
                in_zone_recs[-1].wx - in_zone_recs[0].wx,
                in_zone_recs[-1].wy - in_zone_recs[0].wy,
            )
            if total_path > 0 and net_disp < 0.5 * total_path:
                log.info("track %d: SKIP net_disp=%.2fm < 50%% of path", tid, net_disp)
                continue

            speed_cv  = _cv(speeds_mps)
            speed_mps = _trimmed_mean(speeds_mps)
            speed_mph = speed_mps * _MPH_PER_MPS

            confidence = _compute_confidence(
                frames_in_zone=len(in_zone_recs),
                dwell_seconds=dwell_s,
                per_frame_speed_cv=speed_cv,
                id_switches=self._id_switch_counts.get(tid, 0),
            )

            trail_px  = [(int(r.bx_center), int(r.by_center)) for r in in_zone_recs]
            entry_raw = self._entry_raw.get(tid)
            exit_raw_ = exit_raw.get(tid)

            entry_ann = cal_mod.draw_pass_annotations(
                entry_raw, in_zone_recs[0].bbox_xyxy, tid, speed_mph, self._zone
            ) if entry_raw is not None else None
            exit_ann = cal_mod.draw_pass_annotations(
                exit_raw_, in_zone_recs[-1].bbox_xyxy, tid, speed_mph, self._zone
            ) if exit_raw_ is not None else None

            if exit_raw_ is not None:
                traj = cal_mod.draw_zone_on_frame(exit_raw_.copy(), self._zone)
                for k in range(1, len(trail_px)):
                    cv2.line(traj, trail_px[k - 1], trail_px[k], (0, 200, 255), 2)
                for pt in trail_px[::max(1, len(trail_px) // 20)]:
                    cv2.circle(traj, pt, 3, (0, 200, 255), -1)
                mx, my = trail_px[len(trail_px) // 2]
                cv2.putText(traj, f"{speed_mph:.1f} mph",
                            (mx + 6, my - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            else:
                traj = None

            passes.append(PassResult(
                track_id=tid,
                entry_t_ms=in_zone_recs[0].t_ms,
                exit_t_ms=in_zone_recs[-1].t_ms,
                entry_bbox_xyxy=in_zone_recs[0].bbox_xyxy,
                exit_bbox_xyxy=in_zone_recs[-1].bbox_xyxy,
                sw_speed_mph=round(speed_mph, 2),
                sw_speed_mps=round(speed_mps, 3),
                frames_in_zone=len(in_zone_recs),
                dwell_seconds=round(dwell_s, 3),
                median_heading_deg=round(median_heading, 1),
                per_frame_speed_cv=round(speed_cv, 3),
                id_switches=self._id_switch_counts.get(tid, 0),
                bbox_area_cv=round(bbox_cv, 3),
                net_displacement_m=round(net_disp, 3),
                confidence=confidence,
                entry_frame=entry_ann,
                exit_frame=exit_ann,
                entry_frame_raw=entry_raw,
                exit_frame_raw=exit_raw_,
                trajectory_frame=traj,
            ))
            log.info("track %d: PASS %.1f mph  frames=%d  dwell=%.2fs  conf=%.2f",
                     tid, speed_mph, len(in_zone_recs), dwell_s, confidence)

        return passes
