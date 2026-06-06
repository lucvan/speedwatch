"""
Speed-distribution statistics + a self-contained inline-SVG renderer.

Server-side SVG keeps the app dependency-free (no JS charting lib, matching the rest of the
UI). `speed_distribution_svg` draws the road-wide speed histogram as a smoothed "bell curve",
with optional threshold/percentile markers and an optional overlay of one vehicle's own passes
so you can see where that vehicle sits against the whole road.
"""
from __future__ import annotations
import bisect
import html
import math


# ── Numeric helpers ──────────────────────────────────────────────────────────

def _clean(values) -> list[float]:
    out = []
    for v in values or []:
        if v is None:
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def percentile(sorted_vals: list[float], p: float) -> float | None:
    """Linear-interpolation percentile (p in 0..100) of an already-sorted list."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


def percentile_rank(sorted_pop: list[float], x: float) -> float | None:
    """Percentage of the population at or below x (0..100)."""
    if not sorted_pop or x is None:
        return None
    i = bisect.bisect_right(sorted_pop, x)
    return 100.0 * i / len(sorted_pop)


def median(sorted_vals: list[float]) -> float | None:
    return percentile(sorted_vals, 50)


def _histogram(values: list[float], lo: float, hi: float, bins: int) -> list[int]:
    counts = [0] * bins
    if hi <= lo:
        return counts
    width = (hi - lo) / bins
    for v in values:
        idx = int((v - lo) / width)
        idx = max(0, min(bins - 1, idx))
        counts[idx] += 1
    return counts


def _kde(values: list[float], xs: list[float], bandwidth: float) -> list[float]:
    """Gaussian kernel density estimate of `values`, evaluated at each point in `xs`."""
    n = len(values)
    if n == 0 or bandwidth <= 0:
        return [0.0] * len(xs)
    norm = 1.0 / (n * bandwidth * math.sqrt(2 * math.pi))
    out = []
    for x in xs:
        acc = 0.0
        for v in values:
            z = (x - v) / bandwidth
            acc += math.exp(-0.5 * z * z)
        out.append(norm * acc)
    return out


def _bandwidth(values: list[float]) -> float:
    """Silverman-ish KDE bandwidth with a 1 mph floor so a few passes don't render needle-thin."""
    n = len(values)
    if n < 2:
        return 1.5
    m = sum(values) / n
    sd = (sum((v - m) ** 2 for v in values) / n) ** 0.5
    if sd <= 0:
        return 1.5
    return max(1.0, 0.9 * sd * n ** (-0.2))


def _nice_ceil(x: float) -> float:
    """Round up to a tidy axis maximum (nearest 5, then 10 for larger values)."""
    if x <= 0:
        return 10.0
    step = 5.0 if x <= 40 else 10.0
    return math.ceil(x / step) * step


def speed_summary(values, warn: float | None = None, limit: float | None = None) -> dict | None:
    """Headline distribution stats for a set of speeds (the road population, or one vehicle)."""
    vals = sorted(_clean(values))
    if not vals:
        return None
    n = len(vals)
    over = sum(1 for v in vals if limit is not None and v > limit)
    return {
        "n": n,
        "min": round(vals[0], 1),
        "mean": round(sum(vals) / n, 1),
        "median": round(percentile(vals, 50), 1),
        "p85": round(percentile(vals, 85), 1),   # classic traffic-engineering metric
        "p95": round(percentile(vals, 95), 1),
        "max": round(vals[-1], 1),
        "over_limit": over,
        "over_pct": round(100.0 * over / n, 1) if limit is not None else None,
    }


# ── Inline-SVG distribution chart ────────────────────────────────────────────

def speed_distribution_svg(
    road,
    highlight=None,
    *,
    warn: float | None = None,
    limit: float | None = None,
    markers: list[tuple] | None = None,
    width: int = 680,
    height: int = 220,
    hi: float | None = None,
    highlight_label: str = "this vehicle",
) -> str:
    """
    Render the road speed distribution as an inline SVG (responsive via viewBox).

    - `road`: every pass speed on the road (the background distribution).
    - `highlight`: optional list of one vehicle's pass speeds, overlaid as ticks on the
      baseline plus a vertical line at their median.
    - `markers`: extra vertical reference lines as (value, css_colour, label).
    Returns "" when there isn't enough data to plot.
    """
    road = sorted(_clean(road))
    hl = sorted(_clean(highlight)) if highlight else []
    if len(road) < 2:
        return ""

    top = _nice_ceil(max(road[-1], limit or 0, hl[-1] if hl else 0))
    bins = max(10, min(40, int(round(top / 2))))   # ~2 mph bins
    bin_w = top / bins
    counts = _histogram(road, 0, top, bins)
    max_c = max(counts) or 1

    # Geometry
    ml, mr, mt, mb = 8, 8, 14, 26      # margins
    pw = width - ml - mr               # plot width
    ph = height - mt - mb              # plot height
    base_y = mt + ph                   # x-axis baseline

    def x_of(speed: float) -> float:
        return ml + (speed / top) * pw

    def bar_h(c: int) -> float:
        return (c / max_c) * ph

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'preserveAspectRatio="xMidYMid meet" role="img" '
        f'style="max-width:{width}px;font-family:inherit">'
    ]
    # Baseline
    parts.append(f'<line x1="{ml}" y1="{base_y:.1f}" x2="{ml + pw}" y2="{base_y:.1f}" '
                 f'stroke="var(--border)" stroke-width="1"/>')

    # Histogram bars (road distribution)
    bw = pw / bins
    for i, c in enumerate(counts):
        if c == 0:
            continue
        h = bar_h(c)
        x = ml + i * bw
        # Colour each bar by its speed band (centre of the bin).
        centre = (i + 0.5) * bin_w
        fill = "#5fd38a"
        if limit is not None and centre > limit:
            fill = "#ff6b6b"
        elif warn is not None and centre >= warn:
            fill = "#e6b84f"
        parts.append(f'<rect x="{x + 0.5:.1f}" y="{base_y - h:.1f}" width="{bw - 1:.1f}" '
                     f'height="{h:.1f}" fill="{fill}" opacity="0.45"/>')

    # Smoothed "bell curve" polyline over the bar tops (3-point moving average).
    if bins >= 3:
        smooth = []
        for i in range(bins):
            window = counts[max(0, i - 1): i + 2]
            avg = sum(window) / len(window)
            smooth.append((ml + (i + 0.5) * bw, base_y - bar_h(avg)))
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in smooth)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--accent)" '
                     f'stroke-width="2" opacity="0.9"/>')

    # Vertical reference markers (thresholds + percentiles)
    all_markers = list(markers or [])
    if warn is not None:
        all_markers.append((warn, "#e6b84f", f"warn {warn:g}"))
    if limit is not None:
        all_markers.append((limit, "#ff6b6b", f"limit {limit:g}"))
    for val, colour, label in all_markers:
        if val is None or val > top:
            continue
        x = x_of(val)
        parts.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{base_y:.1f}" '
                     f'stroke="{colour}" stroke-width="1.4" stroke-dasharray="4 3" opacity="0.85"/>')
        parts.append(f'<text x="{x:.1f}" y="{mt - 3:.1f}" fill="{colour}" font-size="10" '
                     f'text-anchor="middle">{html.escape(str(label))}</text>')

    # Overlay one vehicle's own pass distribution: a filled density (KDE) curve showing the
    # shape of their speeds, the individual passes as ticks, and a median line. Drawn in a
    # distinct colour (violet) so it reads clearly against the band-coloured road bars + curve.
    if hl:
        HL = "#a371f7"
        npts = 96
        xs = [top * i / (npts - 1) for i in range(npts)]
        dens = _kde(hl, xs, _bandwidth(hl))
        dmax = max(dens) or 1.0
        scale = (0.82 * ph) / dmax
        curve = [(x_of(x), base_y - d * scale) for x, d in zip(xs, dens)]
        area = (f"{ml:.1f},{base_y:.1f} "
                + " ".join(f"{x:.1f},{y:.1f}" for x, y in curve)
                + f" {ml + pw:.1f},{base_y:.1f}")
        parts.append(f'<polygon points="{area}" fill="{HL}" opacity="0.20"/>')
        parts.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in curve)}" '
                     f'fill="none" stroke="{HL}" stroke-width="2"/>')
        # Individual passes as ticks straddling the baseline.
        for v in hl:
            if v > top:
                continue
            x = x_of(v)
            parts.append(f'<line x1="{x:.1f}" y1="{base_y - 8:.1f}" x2="{x:.1f}" y2="{base_y + 4:.1f}" '
                         f'stroke="{HL}" stroke-width="2" opacity="0.95"/>')
        hmed = percentile(hl, 50)
        if hmed is not None and hmed <= top:
            x = x_of(hmed)
            parts.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{base_y:.1f}" '
                         f'stroke="{HL}" stroke-width="2" stroke-dasharray="2 2"/>')
            parts.append(f'<text x="{x:.1f}" y="{mt + 9:.1f}" fill="{HL}" font-size="10" '
                         f'text-anchor="middle">{html.escape(highlight_label)}</text>')

    # X-axis ticks
    tick_step = 10 if top > 30 else 5
    t = 0
    while t <= top + 0.001:
        x = x_of(t)
        parts.append(f'<line x1="{x:.1f}" y1="{base_y:.1f}" x2="{x:.1f}" y2="{base_y + 4:.1f}" '
                     f'stroke="var(--muted)" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{base_y + 16:.1f}" fill="var(--muted)" font-size="10" '
                     f'text-anchor="middle">{int(t)}</text>')
        t += tick_step
    parts.append(f'<text x="{ml + pw:.1f}" y="{base_y + 16:.1f}" fill="var(--muted)" '
                 f'font-size="10" text-anchor="end">mph</text>')

    parts.append("</svg>")
    return "".join(parts)
