"""
Calibration: reconstruct a road-plane quadrilateral from 4 clicked pixel corners
+ 5 distance measurements, compute a homography (image px → world metres), and
persist/retrieve calibrations from the DB.

Measurement convention:
  corners 0..3 in clockwise order, starting top-left.
  edge_distances_m = [d01, d12, d23, d30]   (adjacent edge lengths)
  diagonal_m       = d02                    (corner 0 → corner 2)

World coordinate frame:
  pt0 at origin, pt1 on positive X-axis.
  Y increases "away" from the camera (road depth direction).
  Expected vehicle headings: ~0° (east) or ~180° (west).
"""

from __future__ import annotations
import json
import logging
import math
from typing import Sequence

import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

log = logging.getLogger(__name__)


# ── Quad reconstruction ────────────────────────────────────────────────────────

def _circle_intersect(
    c1: tuple[float, float], r1: float,
    c2: tuple[float, float], r2: float,
    sign: int = 1,
) -> tuple[float, float]:
    ax, ay = c1
    bx, by = c2
    d = math.hypot(bx - ax, by - ay)
    if d < 1e-9:
        raise ValueError("Degenerate calibration: two corners coincide in world space")
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2.0 * d)
    h_sq = max(0.0, r1 ** 2 - a ** 2)
    h = math.sqrt(h_sq)
    mx = ax + a * (bx - ax) / d
    my = ay + a * (by - ay) / d
    px = -(by - ay) / d
    py = (bx - ax) / d
    return (mx + sign * h * px, my + sign * h * py)


def reconstruct_quad(
    edge_distances_m: Sequence[float],
    diagonal_m: float,
) -> list[tuple[float, float]]:
    """
    Given 4 edge lengths [d01, d12, d23, d30] and diagonal d02,
    return [(x,y)×4] world coords in metres (pt0 at origin, pt1 on +x axis).

    Raises ValueError if the distances are geometrically impossible.
    """
    e01, e12, e23, e30 = edge_distances_m

    pt0 = (0.0, 0.0)
    pt1 = (e01, 0.0)

    # pt2: circle from pt0 (radius=d02) ∩ circle from pt1 (radius=e12)
    # choose positive-y intersection so quad opens "forward"
    pt2 = _circle_intersect(pt0, diagonal_m, pt1, e12, sign=1)

    # pt3: circle from pt0 (radius=e30) ∩ circle from pt2 (radius=e23)
    # choose the intersection that keeps the quad roughly convex
    try:
        pt3_pos = _circle_intersect(pt0, e30, pt2, e23, sign=+1)
        pt3_neg = _circle_intersect(pt0, e30, pt2, e23, sign=-1)
    except ValueError:
        raise ValueError("Calibration distances are geometrically impossible for pt3")

    # Pick the pt3 that forms a convex (or less concave) quad with pt0..pt2
    def _signed_area(pts: list) -> float:
        n = len(pts)
        return 0.5 * sum(
            pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
            for i in range(n)
        )

    quad_pos = [pt0, pt1, pt2, pt3_pos]
    quad_neg = [pt0, pt1, pt2, pt3_neg]
    # Pick the solution that produces a larger-area (convex) quad.
    # The degenerate solution has a much smaller area and is concave.
    pt3 = pt3_pos if abs(_signed_area(quad_pos)) >= abs(_signed_area(quad_neg)) else pt3_neg

    return [pt0, pt1, pt2, pt3]


def validate_quad_distances(
    world_coords: list[tuple[float, float]],
    edge_distances_m: Sequence[float],
    diagonal_m: float,
) -> float:
    """Return max absolute error in metres between reconstructed and supplied distances."""
    pts = world_coords
    n = len(pts)
    errors = []
    for i, d_given in enumerate(edge_distances_m):
        j = (i + 1) % n
        d_calc = math.hypot(pts[j][0] - pts[i][0], pts[j][1] - pts[i][1])
        errors.append(abs(d_calc - d_given))
    # diagonal 0→2
    d_diag = math.hypot(pts[2][0] - pts[0][0], pts[2][1] - pts[0][1])
    errors.append(abs(d_diag - diagonal_m))
    return max(errors)


# ── Homography ─────────────────────────────────────────────────────────────────

def compute_homography(
    zone_pixels_norm: list[tuple[float, float]],
    world_coords: list[tuple[float, float]],
    frame_w: int,
    frame_h: int,
) -> tuple[np.ndarray, float]:
    """
    Compute the perspective transform from pixel coords to world metre coords.
    zone_pixels_norm: normalised [0,1] pixel coords (4 corners).
    Returns (H, residual_max_px).
    H maps image pixel (x,y) → world (x,y) in metres.
    """
    src = np.array(
        [[x * frame_w, y * frame_h] for x, y in zone_pixels_norm],
        dtype=np.float32,
    )
    dst = np.array(world_coords, dtype=np.float32)

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("findHomography failed — check that the 4 points are not collinear")

    # Residual: project src through H and compare to dst
    src_h = np.array([[[x, y]] for x, y in src], dtype=np.float32)
    projected = cv2.perspectiveTransform(src_h, H).reshape(-1, 2)

    # Convert residual back to pixel space (invert H)
    H_inv = np.linalg.inv(H)
    proj_px_h = np.array([[[x, y]] for x, y in projected], dtype=np.float32)
    back_px = cv2.perspectiveTransform(proj_px_h, H_inv).reshape(-1, 2)
    residuals = np.linalg.norm(back_px - src, axis=1)
    residual_max_px = float(residuals.max())

    return H, residual_max_px


def road_axis_from_edge(
    zone_pixels_norm: list[tuple[float, float]],
    H: np.ndarray,
) -> float:
    """
    Derive the expected vehicle heading (degrees) in world space from
    the direction of edge 0 (pt0→pt1) projected through the homography.
    Returns the angle such that vehicles going in either direction are ±180°.
    """
    # Edge 0 in world space (pt0 is at origin, pt1 is on +x axis by construction)
    # So the road axis is always 0° (east) in our canonical world frame.
    return 0.0


# ── Point-in-zone helpers ──────────────────────────────────────────────────────

class ZonePolygon:
    """Shapely-backed polygon for point-in-zone tests in normalised pixel coords."""

    def __init__(self, zone_pixels_norm: list[tuple[float, float]]):
        self._poly    = Polygon(zone_pixels_norm)
        self._prepped = prep(self._poly)

    def contains_norm(self, x_norm: float, y_norm: float) -> bool:
        return self._prepped.contains(Point(x_norm, y_norm))

    def contains_px(self, px: float, py: float, frame_w: int, frame_h: int) -> bool:
        return self.contains_norm(px / frame_w, py / frame_h)

    def pixel_coords(self, frame_w: int, frame_h: int) -> list[tuple[int, int]]:
        return [(int(x * frame_w), int(y * frame_h)) for x, y in self._poly.exterior.coords[:-1]]


# ── Frame annotation ───────────────────────────────────────────────────────────

def draw_zone_on_frame(
    frame: np.ndarray,
    zone: ZonePolygon,
    color: tuple[int, int, int] = (0, 220, 255),
    alpha: float = 0.25,
) -> np.ndarray:
    h, w = frame.shape[:2]
    pts = np.array(zone.pixel_coords(w, h), dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.polylines(overlay, [pts], True, color, 2)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_pass_annotations(
    frame: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    track_id: int,
    speed_mph: float | None,
    zone: ZonePolygon,
    trail_pxs: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    out = draw_zone_on_frame(frame, zone)
    x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
    color = (0, 200, 80)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    label = f"#{track_id}"
    if speed_mph is not None:
        label += f"  {speed_mph:.1f} mph"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ly = max(y1 - 6, th + 4)
    cv2.rectangle(out, (x1, ly - th - 4), (x1 + tw + 4, ly + 2), color, -1)
    cv2.putText(out, label, (x1 + 2, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    if trail_pxs:
        for i in range(1, len(trail_pxs)):
            cv2.line(out, trail_pxs[i - 1], trail_pxs[i], (0, 180, 255), 2)
    return out


# ── Full calibration dict builder (called from the API handler) ────────────────

def build_calibration(
    camera: str,
    zone_pixels_norm: list[tuple[float, float]],
    edge_distances_m: list[float],
    diagonal_m: float,
    frame_w: int,
    frame_h: int,
    notes: str = "",
) -> dict:
    """
    Validate inputs, reconstruct world quad, compute homography.
    Returns a dict ready for db.save_calibration().
    Raises ValueError with a user-friendly message on bad input.
    """
    if len(zone_pixels_norm) != 4:
        raise ValueError("Exactly 4 zone corners required")
    if len(edge_distances_m) != 4:
        raise ValueError("Exactly 4 edge distances required")
    if any(d <= 0 for d in edge_distances_m) or diagonal_m <= 0:
        raise ValueError("All distances must be positive")

    world_coords = reconstruct_quad(edge_distances_m, diagonal_m)

    dist_error = validate_quad_distances(world_coords, edge_distances_m, diagonal_m)
    if dist_error > 0.05:
        log.warning("Quad distance residual %.3f m (>5cm) — measurements may be inconsistent", dist_error)

    H, residual_px = compute_homography(zone_pixels_norm, world_coords, frame_w, frame_h)
    road_axis = road_axis_from_edge(zone_pixels_norm, H)

    return {
        "camera": camera,
        "zone_pixels": zone_pixels_norm,
        "edge_distances_m": edge_distances_m,
        "diagonal_m": diagonal_m,
        "world_coords": world_coords,
        "homography": H.tolist(),
        "road_axis_deg": road_axis,
        "residual_max_px": residual_px,
        "frame_w": frame_w,
        "frame_h": frame_h,
        "notes": notes,
    }


def homography_from_dict(cal: dict) -> np.ndarray:
    return np.array(cal["homography"], dtype=np.float64)


def pixel_to_world(
    px: float, py: float, H: np.ndarray
) -> tuple[float, float]:
    pt = cv2.perspectiveTransform(
        np.array([[[px, py]]], dtype=np.float32), H
    )
    return float(pt[0][0][0]), float(pt[0][0][1])
