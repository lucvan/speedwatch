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
}

function _onCanvasClick(e) {
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
