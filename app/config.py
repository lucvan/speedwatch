from __future__ import annotations
import os
from pathlib import Path

APP_VERSION = "0.2.3"

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# ffmpeg binaries (full path avoids PATH lookup failures in service context)
FFMPEG_BIN  = os.getenv("FFMPEG_BIN",  "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")

# Hardware-accelerated decode. d3d11va offloads H.264 decode to the GPU video engine
# (VCN on the RX 9070 XT), cutting ffmpeg CPU ~40% for the constant 4K@30 decode.
# Set empty to fall back to software decode if hwaccel ever misbehaves.
FFMPEG_HWACCEL = os.getenv("FFMPEG_HWACCEL", "d3d11va")

# Camera — single source stream for the whole pipeline (detection, stills, clips).
CAMERA_RTSP    = os.getenv("CAMERA_RTSP",    "")  # required — set in .env
CAMERA_NAME    = os.getenv("CAMERA_NAME",    "camera")

# Pipeline working resolution. The HQ stream is decoded and scaled to this size for the
# in-memory pipeline — detection, tracking, stills, and clips all run at this resolution,
# so snapshots are inherently time-aligned (no separate HQ recording / seek-matching).
# The HQ stream is 16:9, so 1920x1080 preserves aspect exactly. Set 0x0 for native res.
PIPELINE_WIDTH  = int(os.getenv("PIPELINE_WIDTH",  "1920"))
PIPELINE_HEIGHT = int(os.getenv("PIPELINE_HEIGHT", "1080"))

# Safety cap: force-close a motion gate after this many seconds so a stuck gate
# (foliage, a parked car) can't grow the raw-frame buffer until the process OOMs.
MAX_SESSION_SECONDS = float(os.getenv("MAX_SESSION_SECONDS", "20"))

# YOLO / inference
YOLO_MODEL    = os.getenv("YOLO_MODEL", "models/yolo11m.onnx")  # relative to install root
ONNX_PROVIDER = os.getenv("ONNX_PROVIDER", "auto")  # auto | DmlExecutionProvider | CPUExecutionProvider

# Detection
OBJECT_CLASSES = os.getenv("OBJECT_CLASSES", "car,truck,bus")
DETECT_FPS     = float(os.getenv("DETECT_FPS", "12"))  # frames/sec pulled from stream (also clip fps)
# YOLO/tracking is strided down to this fps from DETECT_FPS — clips still use every frame.
DETECT_INFER_FPS = float(os.getenv("DETECT_INFER_FPS", "15"))
# Motion gate runs MOG2 on the frame scaled by this factor (gate needs no full-res detail).
MOTION_SCALE     = float(os.getenv("MOTION_SCALE", "0.25"))

# Speed qualification
MIN_DT_SECONDS = float(os.getenv("MIN_DT_SECONDS", "0.3"))

# Speed banding for the UI. The legal limit is the red threshold; the warn (amber)
# threshold is set lower to absorb measurement margin on this narrow dead-end road.
#   speed < SPEED_WARN_MPH            → green  (within comfortable margin)
#   SPEED_WARN_MPH ≤ speed ≤ LIMIT    → amber  (watch)
#   speed > SPEED_LIMIT_MPH           → red    (over the legal limit)
SPEED_LIMIT_MPH = float(os.getenv("SPEED_LIMIT_MPH", "20"))
SPEED_WARN_MPH  = float(os.getenv("SPEED_WARN_MPH",  "16"))

# ANPR / number-plate recognition (fast-alpr). Runs on CPU (tiny models) so it does not
# contend with YOLO on DML. Applied to the entry + exit car crop of each qualifying pass;
# the higher-confidence read wins (handles front/rear plate by direction automatically).
ENABLE_ALPR        = os.getenv("ENABLE_ALPR", "1") not in ("0", "false", "False", "")
ALPR_DETECTOR_MODEL = os.getenv("ALPR_DETECTOR_MODEL", "yolo-v9-t-512-license-plate-end2end")
ALPR_OCR_MODEL      = os.getenv("ALPR_OCR_MODEL", "european-plates-mobile-vit-v2-model")
ALPR_MIN_CONF       = float(os.getenv("ALPR_MIN_CONF", "0.6"))   # mean per-char OCR confidence
ALPR_CROP_PAD       = int(os.getenv("ALPR_CROP_PAD", "12"))      # px padding around car bbox

# Vehicle vision-description (local Ollama multimodal model). For plates seen more than
# VEHICLE_MIN_PASSES times, describe the car (colour/make/body) from its best car crop.
# Runs locally so neighbours' vehicle data never leaves the box.
OLLAMA_URL           = os.getenv("OLLAMA_URL", "http://localhost:11434")
VEHICLE_VISION_MODEL = os.getenv("VEHICLE_VISION_MODEL", "llama3.2-vision:11b")
VEHICLE_AUTODESCRIBE = os.getenv("VEHICLE_AUTODESCRIBE", "1") not in ("0", "false", "False", "")
VEHICLE_MIN_PASSES   = int(os.getenv("VEHICLE_MIN_PASSES", "3"))        # describe when passes > this
VEHICLE_VISION_TIMEOUT = float(os.getenv("VEHICLE_VISION_TIMEOUT", "180"))
VEHICLE_VISION_PROMPT = os.getenv(
    "VEHICLE_VISION_PROMPT",
    "Identify the car in this image. Respond with ONLY its colour, make and model on a "
    "single line, in exactly this format: Colour Make Model (for example: "
    "Silver Volkswagen Golf). Write Unknown for any part you cannot determine. Do not "
    "include the number plate, body type, year, condition, or any other words or punctuation.",
)
# Max soft edit-distance for suggesting two plates are the same car (OCR-confusable subs cost 0.4).
ALIAS_SUGGEST_MAXDIST = float(os.getenv("ALIAS_SUGGEST_MAXDIST", "1.5"))

# Vehicle Re-ID image embeddings (visual grouping of plateless / look-alike cars).
# Model: OpenVINO OMZ `vehicle-reid-0001` (OSNet, MIT) exported to ONNX by bootstrap.ps1.
# Runs on onnxruntime (CPU by default — the model is tiny, ~2.6 GFLOPs — leaving DML for YOLO).
# If the model file is absent, embedding features no-op and the rest of the pipeline runs normally.
REID_MODEL         = os.getenv("REID_MODEL", "models/vehicle-reid-0001.onnx")
REID_PROVIDER      = os.getenv("REID_PROVIDER", "cpu")     # cpu | dml
# Cosine similarity to call two crops the same car. Needs tuning on real footage: too low
# over-merges distinct look-alikes, too high splits one car into several groups. Start
# conservative (more, purer groups) and lower it if the same car keeps splitting.
REID_SIM_THRESHOLD = float(os.getenv("REID_SIM_THRESHOLD", "0.7"))

def reid_model_path() -> Path:
    p = Path(REID_MODEL)
    if not p.is_absolute():
        p = Path(INSTALL_DIR) / p
    return p

# Storage
DATA_DIR   = os.getenv("DATA_DIR", str(Path(__file__).parent.parent / "data"))
INSTALL_DIR = str(Path(__file__).parent.parent)

# Server
# Bind address. Default 127.0.0.1 (localhost only). Set HOST=0.0.0.0 to expose the UI to
# other devices on the LAN (also needs a Windows firewall rule allowing inbound TCP on PORT).
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8765"))

# Per-stage timing profiler — logs avg ms per pipeline stage every PROFILE_FLUSH_S seconds.
# Zero overhead when off. Turn on temporarily to find CPU hot spots.
PROFILE         = os.getenv("PROFILE", "0") in ("1", "true", "True")
PROFILE_FLUSH_S = float(os.getenv("PROFILE_FLUSH_S", "10"))

# COCO class id map
_COCO_CLASS_IDS = {"car": 2, "truck": 7, "bus": 5}

def object_class_ids() -> list[int]:
    names = [n.strip() for n in OBJECT_CLASSES.split(",")]
    return [_COCO_CLASS_IDS[n] for n in names if n in _COCO_CLASS_IDS]

def model_path() -> Path:
    p = Path(YOLO_MODEL)
    if not p.is_absolute():
        p = Path(INSTALL_DIR) / p
    return p
