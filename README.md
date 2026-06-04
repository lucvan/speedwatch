# speedwatch

Standalone vehicle speed-detection and ANPR service for a single fixed camera. Pulls a
4K RTSPS stream, decodes it on the GPU, runs object detection + tracking, estimates
real-world speed via a homography, reads number plates, and serves clips + events over a
local web UI. Runs as a Windows service (NSSM); no Docker.

Replaced a Frigate deployment — it keeps only clean 30 fps clips and drops empty events.

## Pipeline

```
4K RTSPS  ──(ffmpeg -hwaccel d3d11va)──►  GPU decode
   │
   ├─ 30 fps capture ───────────────────►  clip writer (clean 30 fps mp4)
   └─ 15 fps  ──►  YOLO11 (onnxruntime-directml) ─► ByteTrack
                        │
                        ├─ homography ─► real-world speed (mph)
                        └─ fast-alpr ──► number-plate text
                                          │
                              SQLite events + web UI (:8765)
```

- **Decode:** `ffmpeg -hwaccel d3d11va` (4K HEVC, ~40% less CPU than software decode).
- **Detection:** YOLO11 medium ONNX on `DmlExecutionProvider` (DirectML) — ~7 ms/frame
  on the AMD RX 9070 XT vs ~136 ms on CPU (~19× faster).
- **Tracking:** ByteTrack (via `supervision`).
- **Speed:** per-track displacement mapped through a calibrated homography.
- **ANPR:** `fast-alpr` (fast-plate-ocr + open-image-models), reuses the same onnxruntime.

## Prerequisites

- Windows 11 with a DirectML-capable GPU (built for **AMD RX 9070 XT / DirectML** — no
  CUDA). `ONNX_PROVIDER=auto` falls back to CPU if no DML device is present.
- Python 3.11+.
- [NSSM](https://nssm.cc/) on `PATH` (or copied to `C:\Windows\System32\nssm.exe`) for
  service install.
- A camera RTSPS stream (developed against a UniFi UDM Protect stream).

## Setup

```powershell
git clone https://github.com/lucvan/speedwatch.git C:\speedwatch
cd C:\speedwatch

# Fetch the gitignored bits (venv, YOLO model, ffmpeg/ffprobe)
.\bootstrap.ps1

# Configure
Copy-Item .env.example .env
notepad .env            # fill in CAMERA_RTSP*, CAMERA_NAME, paths

# Install + start the Windows service (run elevated)
.\install-service.ps1
nssm start speedwatch
```

UI: <http://localhost:8765>

The model weights, `.venv`, and `ffmpeg.exe`/`ffprobe.exe` are **not** in the repo
(see `.gitignore`); `bootstrap.ps1` fetches them so a clean clone is fully deployable.

## Configuration (`.env`)

| Key | Meaning |
|---|---|
| `FFMPEG_BIN` / `FFPROBE_BIN` | Path to the ffmpeg/ffprobe binaries |
| `CAMERA_RTSP` | Detect stream URL (`rtsps://<ROUTER_IP>:7441/<KEY>?enableSrtp`) |
| `CAMERA_RTSP_HQ` | Record / high-quality stream URL |
| `CAMERA_NAME` | Logical camera name (used in events/clips) |
| `YOLO_MODEL` | Path to the YOLO11 ONNX model (default `models/yolo11m.onnx`) |
| `ONNX_PROVIDER` | `auto` \| `dml` \| `cpu` |
| `OBJECT_CLASSES` | Comma-separated COCO classes to track (`car,truck,bus`) |
| `DETECT_FPS` | Inference frame rate |
| `MIN_DT_SECONDS` | Minimum track interval used in speed estimation |
| `DATA_DIR` | Where clips, logs, and the SQLite DB are written |
| `PORT` | Web UI / API port (default `8765`) |

## Lifecycle (NSSM)

```powershell
nssm start  speedwatch
nssm stop   speedwatch
nssm restart speedwatch
nssm status speedwatch
# logs: <DATA_DIR>\logs\stdout.log  /  stderr.log  (rotated daily / 10 MB)
.\install-service.ps1     # re-run elevated to apply parameter changes
```

> NSSM commands need an **elevated** shell. If `nssm` isn't found, copy it from the
> winget package dir to `C:\Windows\System32\nssm.exe` first.

## Calibration

Speed accuracy depends on two calibrations, in this order.

### 1. Lens intrinsics (distortion correction) — do this once per camera

The zone homography is a planar projective transform: it is exact only for a pinhole
camera. A wide-FOV lens adds radial (barrel) distortion that a homography can't represent,
so the pixel→metre mapping drifts toward the frame edges — exactly where a side-on vehicle
enters and leaves frame.

**Easiest: the on-screen slider tuner (plumb-line method).** On `/calibrate`, click
**Tune lens…**, then drag *Strength* (and optionally *Edge*) until things that are straight
in real life — the kerb, road edge, a fence top or wall — line up straight against the green
reference grid, and **Save lens correction**. No printed board needed; absolute scale is set
later by your tape measurements, so the focal control rarely needs touching. This writes the
same intrinsics file the chessboard method produces.

**Most accurate: chessboard calibration.** Print a board and run:

```powershell
# Print a chessboard (a standard 10x7-square board = 9x6 inner corners). Hold it in the
# camera's view and move it around the WHOLE frame — especially the corners/edges where
# distortion is worst — tilting it at varied angles.

# 1. Capture views through the actual camera stream (saves frames where a board is found):
python -m app.intrinsics capture --count 30 --cols 9 --rows 6

# 2. Estimate K + distortion coefficients (writes data/intrinsics/<camera>.json):
python -m app.intrinsics calibrate --cols 9 --rows 6 --square 0.025
```

Aim for a reprojection RMS < 1.0 px (printed at the end of step 2). The intrinsics are then
**baked into the next zone calibration you save**, and the live pipeline undistorts each
ground point before the homography. Without intrinsics everything still runs — just without
distortion correction (and the `/calibrate` page shows a warning).

### 2. Zone homography

Map the image to the ground plane: open `/calibrate`, click the four zone corners on the
snapshot, and enter the four edge lengths + one diagonal (tape-measured on the road). The
resulting matrix turns per-track pixel displacement into metres, then mph. Re-save the zone
calibration after (re)running lens calibration, and whenever the camera is moved or re-aimed.

> **Timing note:** frame timestamps come from a monotonic frame index (ffmpeg emits CFR at
> `DETECT_FPS`), not wall-clock-at-read. Read-time clock jitter used to corrupt `dt` in
> `speed = displacement / dt`, which scaled with speed and threw off faster vehicles.
