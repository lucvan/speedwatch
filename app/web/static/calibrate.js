'use strict';

let _camera   = '';
let _points   = [];   // [{x_norm, y_norm}]  normalised 0..1
let _canvas   = null;
let _ctx      = null;
let _img      = null;
let _imgW     = 0;
let _imgH     = 0;

const COLORS  = ['#ff4444', '#44ff44', '#4488ff', '#ffaa00'];
const LABELS  = ['0', '1', '2', '3'];

function initCalibration(snapshotUrl, camera) {
  _camera = camera;
  _canvas = document.getElementById('cal-canvas');
  _ctx    = _canvas.getContext('2d');

  _img = new Image();
  _img.crossOrigin = 'anonymous';
  _img.onload = () => {
    _imgW = _img.naturalWidth;
    _imgH = _img.naturalHeight;
    _canvas.width  = _imgW;
    _canvas.height = _imgH;
    _redraw();
  };
  _img.onerror = () => {
    _ctx.fillStyle = '#333';
    _ctx.fillRect(0, 0, _canvas.width || 640, _canvas.height || 360);
    _ctx.fillStyle = '#aaa';
    _ctx.font = '20px sans-serif';
    _ctx.fillText('Snapshot unavailable', 20, 40);
  };
  _img.src = snapshotUrl + '?t=' + Date.now();

  _canvas.addEventListener('click', _onCanvasClick);

  // Lens tuner: sliders drive a debounced undistort preview of the snapshot.
  _lensImg = new Image();
  _lensImg.crossOrigin = 'anonymous';
  _lensImg.onload = redrawLens;
  ['k1', 'k2', 'f'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', _onLensInput);
  });
}

// ── Lens distortion tuner ────────────────────────────────────────────────────

let _lensMode = false;
let _lensImg  = null;
let _lensTimer = null;

function _lensParams() {
  const g = (id, d) => {
    const el = document.getElementById(id);
    return el ? parseFloat(el.value) : d;
  };
  return { k1: g('k1', 0), k2: g('k2', 0), f: g('f', 0.5) };
}

function _onLensInput() {
  const p = _lensParams();
  const set = (id, txt) => { const el = document.getElementById(id); if (el) el.textContent = txt; };
  set('k1-val', p.k1.toFixed(3));
  set('k2-val', p.k2.toFixed(3));
  set('f-val',  p.f.toFixed(2));
  clearTimeout(_lensTimer);
  _lensTimer = setTimeout(() => updateLensPreview(false), 90);
}

function toggleLensMode() {
  _lensMode = !_lensMode;
  const panel = document.getElementById('lens-panel');
  const btn   = document.getElementById('lens-toggle');
  if (panel) panel.classList.toggle('d-none', !_lensMode);
  if (btn)   btn.textContent = _lensMode ? 'Done' : 'Tune lens…';
  _canvas.style.cursor = _lensMode ? 'default' : 'crosshair';
  if (_lensMode) {
    _onLensInput();              // sync labels
    updateLensPreview(false);    // server grabs + caches a frame if needed
  } else {
    _redraw();                   // back to raw snapshot + corner markers
  }
}

function updateLensPreview(refresh) {
  if (!_lensMode) return;
  const p = _lensParams();
  const q = `camera=${encodeURIComponent(_camera)}&k1=${p.k1}&k2=${p.k2}&f=${p.f}` +
            `&refresh=${refresh ? 1 : 0}&t=${Date.now()}`;
  _lensImg.src = '/api/lens/preview?' + q;
}

function refreshLensFrame() { updateLensPreview(true); }

function resetLens() {
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
  set('k1', 0); set('k2', 0); set('f', 0.5);
  _onLensInput();
}

function redrawLens() {
  if (!_ctx) return;
  _ctx.clearRect(0, 0, _canvas.width, _canvas.height);
  if (_lensImg && _lensImg.complete && _lensImg.naturalWidth > 0) {
    _ctx.drawImage(_lensImg, 0, 0, _canvas.width, _canvas.height);
  }
  const grid = document.getElementById('grid-toggle');
  if (!grid || !grid.checked) return;

  const W = _canvas.width, H = _canvas.height;
  _ctx.lineWidth = 1;
  for (let i = 1; i < 10; i++) {
    const mid = (i === 5);
    _ctx.strokeStyle = mid ? 'rgba(0,255,120,0.9)' : 'rgba(0,255,120,0.45)';
    _ctx.lineWidth   = mid ? 2 : 1;
    let x = Math.round(W * i / 10), y = Math.round(H * i / 10);
    _ctx.beginPath(); _ctx.moveTo(x, 0); _ctx.lineTo(x, H); _ctx.stroke();
    _ctx.beginPath(); _ctx.moveTo(0, y); _ctx.lineTo(W, y); _ctx.stroke();
  }
}

async function _postReapply() {
  const r = await fetch('/api/calibration/reapply-lens', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ camera: _camera }),
  });
  let d = {};
  try { d = await r.json(); } catch (e) { /* non-JSON (e.g. 404 page) */ }
  return { ok: r.ok, status: r.status, data: d };
}

async function saveLens() {
  const p = _lensParams();
  const el = document.getElementById('lens-result');
  el.textContent = 'Saving lens correction…';
  el.className   = 'small mt-2 text-muted';
  try {
    const r = await fetch('/api/lens/save?camera=' + encodeURIComponent(_camera), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(p),
    });
    const d = await r.json();
    if (!(r.ok && d.ok)) {
      el.textContent = `Error: ${d.detail || JSON.stringify(d)}`;
      el.className   = 'small mt-2 text-danger';
      return;
    }
    // Auto-apply to the existing zone calibration so it takes effect with no re-clicking.
    el.textContent = `Lens saved (${d.image_w}×${d.image_h}). Applying to your zone calibration…`;
    const res = await _postReapply();
    if (res.ok && res.data.ok) {
      el.textContent = `Lens saved and applied to your zone calibration ` +
                       `(residual ${res.data.residual_max_px.toFixed(1)} px). ` +
                       `Live on the next vehicle.`;
      el.className   = 'small mt-2 text-success';
    } else if (res.status === 404) {
      el.textContent = 'Lens saved. No zone calibration yet — click the 4 corners below and save to bake it in.';
      el.className   = 'small mt-2 text-warning';
    } else {
      el.textContent = `Lens saved, but auto-apply failed (${res.data.detail || res.status}). ` +
                       `Re-save your zone calibration, or restart the service if the button is new.`;
      el.className   = 'small mt-2 text-warning';
    }
  } catch (err) {
    el.textContent = `Network error: ${err.message}`;
    el.className   = 'small mt-2 text-danger';
  }
}

async function reapplyLens() {
  const el = document.getElementById('reapply-result');
  if (el) { el.textContent = 'Applying…'; el.className = 'small ms-2 text-muted'; }
  try {
    const res = await _postReapply();
    if (!el) return;
    if (res.ok && res.data.ok) {
      el.textContent = `Applied (residual ${res.data.residual_max_px.toFixed(1)} px) — live on the next vehicle.`;
      el.className   = 'small ms-2 text-success';
    } else {
      el.textContent = `Error: ${res.data.detail || res.status}`;
      el.className   = 'small ms-2 text-danger';
    }
  } catch (err) {
    if (el) { el.textContent = `Network error: ${err.message}`; el.className = 'small ms-2 text-danger'; }
  }
}

function _onCanvasClick(e) {
  if (_lensMode) return;            // tuning the lens, not picking corners
  if (_points.length >= 4) return;

  const rect   = _canvas.getBoundingClientRect();
  const scaleX = _canvas.width  / rect.width;
  const scaleY = _canvas.height / rect.height;
  const px     = (e.clientX - rect.left) * scaleX;
  const py     = (e.clientY - rect.top)  * scaleY;

  _points.push({
    x_norm: px / _canvas.width,
    y_norm: py / _canvas.height,
  });

  _redraw();
  _updateStatus();

  if (_points.length === 4) {
    document.getElementById('save-btn').disabled = false;
  }
}

function _redraw() {
  if (!_ctx) return;
  _ctx.clearRect(0, 0, _canvas.width, _canvas.height);

  // Background image
  if (_img && _img.complete && _img.naturalWidth > 0) {
    _ctx.drawImage(_img, 0, 0);
  }

  if (_points.length === 0) return;

  // Draw polygon lines
  if (_points.length >= 2) {
    _ctx.beginPath();
    _ctx.strokeStyle = 'rgba(255, 220, 0, 0.8)';
    _ctx.lineWidth   = 2;
    const p0 = _points[0];
    _ctx.moveTo(p0.x_norm * _canvas.width, p0.y_norm * _canvas.height);
    for (let i = 1; i < _points.length; i++) {
      _ctx.lineTo(_points[i].x_norm * _canvas.width, _points[i].y_norm * _canvas.height);
    }
    if (_points.length === 4) {
      _ctx.lineTo(p0.x_norm * _canvas.width, p0.y_norm * _canvas.height);
    }
    _ctx.stroke();
  }

  // Draw each point
  _points.forEach((p, i) => {
    const cx = p.x_norm * _canvas.width;
    const cy = p.y_norm * _canvas.height;

    _ctx.beginPath();
    _ctx.arc(cx, cy, 8, 0, Math.PI * 2);
    _ctx.fillStyle   = COLORS[i];
    _ctx.strokeStyle = '#fff';
    _ctx.lineWidth   = 2;
    _ctx.fill();
    _ctx.stroke();

    _ctx.fillStyle  = '#fff';
    _ctx.font       = 'bold 13px sans-serif';
    _ctx.textAlign  = 'center';
    _ctx.textBaseline = 'middle';
    _ctx.fillText(LABELS[i], cx, cy);
  });
}

function _updateStatus() {
  const el = document.getElementById('point-status');
  if (!el) return;
  if (_points.length < 4) {
    el.textContent = `${_points.length} / 4 corners selected — click corner ${_points.length}`;
    el.className   = 'text-warning small mt-1';
  } else {
    el.textContent = '4 corners selected. Enter measurements and save.';
    el.className   = 'text-success small mt-1';
  }
}

function resetPoints() {
  _points = [];
  document.getElementById('save-btn').disabled = true;
  document.getElementById('save-result').textContent = '';
  _updateStatus();
  _redraw();
}

// ── ANPR ignore-zone tool ────────────────────────────────────────────────────
// Drag to draw red rectangles over fixed background plates; any plate detected inside
// one is discarded by the ingest pipeline. Rectangles are stored normalised (0..1).
let _mCanvas = null, _mCtx = null, _mImg = null, _mCamera = '';
let _mRects = [], _mDrag = null;

function initAnprMask(snapshotUrl, camera, rects) {
  _mCamera = camera;
  _mRects  = Array.isArray(rects) ? rects.map(r => r.slice()) : [];
  _mCanvas = document.getElementById('mask-canvas');
  if (!_mCanvas) return;
  _mCtx = _mCanvas.getContext('2d');

  _mImg = new Image();
  _mImg.crossOrigin = 'anonymous';
  _mImg.onload = () => {
    _mCanvas.width  = _mImg.naturalWidth;
    _mCanvas.height = _mImg.naturalHeight;
    _mRedraw();
  };
  _mImg.onerror = () => {
    _mCanvas.width = 640; _mCanvas.height = 360;
    _mCtx.fillStyle = '#333'; _mCtx.fillRect(0, 0, 640, 360);
    _mCtx.fillStyle = '#aaa'; _mCtx.font = '18px sans-serif';
    _mCtx.fillText('Snapshot unavailable', 20, 40);
  };
  _mImg.src = snapshotUrl + '?t=' + Date.now();

  _mCanvas.addEventListener('mousedown', _mDown);
  _mCanvas.addEventListener('mousemove', _mMove);
  window.addEventListener('mouseup', _mUp);
  _mUpdateStatus();
}

function _mPos(e) {
  const r = _mCanvas.getBoundingClientRect();
  return {
    x: (e.clientX - r.left) * (_mCanvas.width  / r.width)  / _mCanvas.width,
    y: (e.clientY - r.top)  * (_mCanvas.height / r.height) / _mCanvas.height,
  };
}
function _mDown(e) { const p = _mPos(e); _mDrag = { x0: p.x, y0: p.y, x1: p.x, y1: p.y }; }
function _mMove(e) { if (!_mDrag) return; const p = _mPos(e); _mDrag.x1 = p.x; _mDrag.y1 = p.y; _mRedraw(); }
function _mUp() {
  if (!_mDrag) return;
  const d = _mDrag; _mDrag = null;
  const x1 = Math.min(d.x0, d.x1), y1 = Math.min(d.y0, d.y1);
  const x2 = Math.max(d.x0, d.x1), y2 = Math.max(d.y0, d.y1);
  if (x2 - x1 >= 0.01 && y2 - y1 >= 0.01) { _mRects.push([x1, y1, x2, y2]); _mUpdateStatus(); }
  _mRedraw();
}
function _mDrawRect(r, active) {
  const W = _mCanvas.width, H = _mCanvas.height;
  const x = r[0] * W, y = r[1] * H, w = (r[2] - r[0]) * W, h = (r[3] - r[1]) * H;
  _mCtx.fillStyle = active ? 'rgba(255,60,60,0.20)' : 'rgba(255,60,60,0.30)';
  _mCtx.fillRect(x, y, w, h);
  _mCtx.strokeStyle = '#ff3c3c'; _mCtx.lineWidth = 2; _mCtx.strokeRect(x, y, w, h);
}
function _mRedraw() {
  if (!_mCtx) return;
  _mCtx.clearRect(0, 0, _mCanvas.width, _mCanvas.height);
  if (_mImg && _mImg.complete && _mImg.naturalWidth > 0) _mCtx.drawImage(_mImg, 0, 0);
  _mRects.forEach(r => _mDrawRect(r, false));
  if (_mDrag) _mDrawRect([Math.min(_mDrag.x0, _mDrag.x1), Math.min(_mDrag.y0, _mDrag.y1),
                          Math.max(_mDrag.x0, _mDrag.x1), Math.max(_mDrag.y0, _mDrag.y1)], true);
}
function _mUpdateStatus() {
  const el = document.getElementById('mask-status');
  if (el) el.textContent = `${_mRects.length} ignore zone(s). Drag on the image to add one.`;
}
function maskUndo()  { _mRects.pop(); _mUpdateStatus(); _mRedraw(); }
function maskClear() { _mRects = []; _mUpdateStatus(); _mRedraw(); }
function maskRefresh() { if (_mImg) _mImg.src = '/snapshot/' + encodeURIComponent(_mCamera) + '?t=' + Date.now(); }
async function maskSave() {
  const el = document.getElementById('mask-result');
  el.textContent = 'Saving…'; el.className = 'small mt-2 text-muted';
  try {
    const r = await fetch('/api/anpr-mask/save', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera: _mCamera, rects: _mRects }),
    });
    const d = await r.json();
    if (r.ok && d.ok) {
      el.textContent = `Saved ${d.count} ignore zone(s). Live on the next vehicle.`;
      el.className = 'small mt-2 text-success';
    } else {
      el.textContent = `Error: ${d.detail || JSON.stringify(d)}`;
      el.className = 'small mt-2 text-danger';
    }
  } catch (err) {
    el.textContent = `Network error: ${err.message}`;
    el.className = 'small mt-2 text-danger';
  }
}

async function saveCalibration() {
  if (_points.length < 4) {
    alert('Please click all 4 corners first.');
    return;
  }

  const fields = ['e01', 'e12', 'e23', 'e30', 'd02'];
  const vals   = {};
  for (const f of fields) {
    const v = parseFloat(document.getElementById(f).value);
    if (isNaN(v) || v <= 0) {
      alert(`Please enter a valid positive number for ${f}`);
      return;
    }
    vals[f] = v;
  }

  const notes   = document.getElementById('cal-notes').value.trim();
  const resultEl = document.getElementById('save-result');
  resultEl.textContent = 'Saving…';
  resultEl.className   = 'mt-2 small text-muted';

  const payload = {
    camera:          _camera,
    zone_pixels:     _points.map(p => [p.x_norm, p.y_norm]),
    edge_distances_m:[vals.e01, vals.e12, vals.e23, vals.e30],
    diagonal_m:      vals.d02,
    frame_w:         _imgW || 1280,
    frame_h:         _imgH || 720,
    notes:           notes,
  };

  try {
    const r = await fetch('/api/calibration', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const d = await r.json();
    if (r.ok && d.ok) {
      const px = d.residual_max_px.toFixed(1);
      const warn = d.residual_max_px > 8 ? ' ⚠ residual high — re-check measurements' : '';
      resultEl.textContent = `Saved (id ${d.calibration_id}, residual ${px}px)${warn}`;
      resultEl.className   = d.residual_max_px > 8
        ? 'mt-2 small text-warning'
        : 'mt-2 small text-success';
      setTimeout(() => location.reload(), 1500);
    } else {
      resultEl.textContent = `Error: ${d.detail || JSON.stringify(d)}`;
      resultEl.className   = 'mt-2 small text-danger';
    }
  } catch (err) {
    resultEl.textContent = `Network error: ${err.message}`;
    resultEl.className   = 'mt-2 small text-danger';
  }
}
