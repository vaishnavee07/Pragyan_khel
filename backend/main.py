"""
SentraVision AI Vision Platform
Enterprise-grade modular AI inference engine
Phase 4: RT-DETR Detection Upgrade  |  Phase 5: Video Upload Mode
"""
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from core.ai_engine import AIEngine
from api.websocket_handler import WebSocketHandler
from api.video_ws_handler import VideoWebSocketHandler
from core.video_session import VideoSession
import detection_config as cfg

# ── Video session storage ─────────────────────────────────────────────
_VIDEO_UPLOAD_DIR = Path(__file__).parent / "_uploads"
_VIDEO_UPLOAD_DIR.mkdir(exist_ok=True)

_video_sessions: dict[str, VideoSession] = {}   # session_id → VideoSession

# Import detection module based on config
if cfg.MODEL_TYPE == "rtdetr":
    try:
        from modules.rtdetr_detection import RTDETRDetectionModule as DetectionModule
        print("✓ Using RT-DETR detection module")
    except ImportError:
        from modules.object_detection import ObjectDetectionModule as DetectionModule
        print("⚠ RT-DETR not available, using fallback")
else:
    from modules.object_detection import ObjectDetectionModule as DetectionModule
    print("✓ Using YOLO detection module")

# Import autofocus module
from modules.autofocus_module import AutofocusModule
from modules.yolo_seg_module import YOLOv8SegModule

app = FastAPI(
    title="SentraVision AI Platform",
    description="Modular AI Vision Platform with RT-DETR detection",
    version="2.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI Engine
ai_engine = AIEngine()

# Register detection module with config
detection_module = DetectionModule({
    'confidence_threshold': cfg.CONFIDENCE_THRESHOLD,
    'iou_threshold': cfg.IOU_THRESHOLD,
    'device': cfg.DEVICE,
    'detect_only_person': cfg.DETECT_ONLY_PERSON
})
ai_engine.register_module('object_detection', detection_module)

# Register cinematic autofocus module
autofocus_module = AutofocusModule({
    'bbox_size':         120,
    'mode':              'blur',
    # ── Engine selection ─────────────────────────────────────
    # True  = ProximityBlurEngine  (lightweight, 20-30 FPS, no AI overhead)
    # False = TrackingAutofocusEngine (depth + YOLO-seg, higher quality)
    'proximity_mode':    True,
    # ── Proximity engine tuning ─────────────────────────────
    'min_blur_k':        5,       # kernel for nearest neighbour
    'max_blur_k':        35,      # kernel for far background
    'feather':           10,      # soft edge gradient around subject box
    'grace_period':      0.8,
    'bbox_alpha':        0.70,    # Feature 4: bbox smoothing
    'blur_alpha':        0.80,    # Feature 2: temporal blur smoothing
    'transition_frames': 20,      # Feature 2: rack-focus ramp length
    # ── Full engine tuning (used when proximity_mode=False) ──────
    'focus_radius':      75,
    'blur_ksize':        101,
    'use_segmentation':  True,
    'seg_frame_skip':    0,
    'depth_blur_k':      3.5,
    'rack_focus_duration': 0.40,
})

# Inject detector into autofocus for isolation mode
autofocus_module.set_detector(detection_module)

# Create and inject YOLOv11-seg module for Full Silhouette Lock
yolo_seg_module = YOLOv8SegModule({
    'confidence_threshold': cfg.CONFIDENCE_THRESHOLD,
    'iou_threshold':        cfg.IOU_THRESHOLD,
    'device':               cfg.DEVICE,
    'detect_only_person':   cfg.DETECT_ONLY_PERSON,
    'detection_interval':   getattr(cfg, 'DETECTION_INTERVAL', 2),
    'smooth_alpha':         0.40,  # lower = smoother motion
})
try:
    yolo_seg_module.initialize()
    autofocus_module.set_yolo_seg(yolo_seg_module)
    ai_engine.register_module('yolo_seg', yolo_seg_module)
    print("✓ YOLOv11-seg registered and wired to autofocus")
except Exception as e:
    print(f"⚠ YOLOv11-seg init failed: {e} — autofocus will use geometric fallback")

ai_engine.register_module('autofocus', autofocus_module)

# Initialize WebSocket handler
ws_handler = WebSocketHandler(ai_engine)
video_ws_handler = VideoWebSocketHandler(_video_sessions)

@app.on_event("startup")
async def startup_event():
    """Initialize platform on startup"""
    print("=" * 70)
    print("🔷 SentraVision AI Vision Platform v5.0 [YOLOv11 Cinematic Engine]")
    print("=" * 70)
    print(f"➤ Detector : YOLOv11x-seg  (f/1.4 cinematic DoF)")
    print(f"➤ Device   : {cfg.DEVICE.upper()}  (CUDA auto-probe)")
    print(f"➤ Overlays : DISABLED  (invisible intelligence mode)")
    print(f"➤ Engine   : TrackingAutofocusEngine + MiDaS depth")
    print("➤ Modular AI Engine initialized")
    print(f"➤ Available modes: {', '.join(ai_engine.get_available_modes())}")
    print("➤ Cinematic Autofocus: ENABLED")
    print("➤ Full Silhouette Subject Lock: ENABLED")
    print("   └─ Pixel-accurate person segmentation (MediaPipe)")
    print("   └─ Edge recovery + anti-halo refinement")
    print("   └─ Entire person sharp (arms, legs, hair, clothes)")
    
    # Activate default mode
    if ai_engine.switch_mode('object_detection'):
        print("✓ Detection module active")
    else:
        print("✗ Failed to activate detection")
    print("=" * 70)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    ai_engine.shutdown()
    print("✓ Platform shutdown complete")

@app.get("/")
async def root():
    return {
        "platform": "SentraVision AI Vision Platform",
        "version": "2.0.0",
        "status": "operational",
        "active_mode": ai_engine.get_active_mode(),
        "available_modes": ai_engine.get_available_modes()
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "ai_engine": "operational",
        "active_mode": ai_engine.get_active_mode()
    }

@app.get("/modes")
async def get_modes():
    """Get available AI modes"""
    return {
        "modes": ai_engine.get_available_modes(),
        "active": ai_engine.get_active_mode()
    }

@app.post("/modes/{mode_name}/activate")
async def activate_mode(mode_name: str):
    """Switch to a different AI mode"""
    success = ai_engine.switch_mode(mode_name)
    return {
        "success": success,
        "active_mode": ai_engine.get_active_mode() if success else None
    }

@app.post("/autofocus/mode")
async def set_autofocus_mode(mode: str):
    """
    Switch autofocus mode between 'blur' and 'isolation'.
    
    Args:
        mode: 'blur' (Gaussian blur background with segmentation support) or 
              'isolation' (hard subject cutout)
    
    Returns:
        { "success": bool, "mode": str }
    """
    if mode not in ['blur', 'isolation']:
        return {
            "success": False,
            "error": f"Invalid mode '{mode}'. Use 'blur' or 'isolation'."
        }
    
    autofocus_module.set_mode(mode)
    
    return {
        "success": True,
        "mode": mode
    }

@app.post("/autofocus/segmentation")
async def set_autofocus_segmentation(enabled: bool):
    """
    Enable/disable pixel-accurate segmentation in blur mode.
    
    FULL SILHOUETTE SUBJECT LOCK MODE:
    - When enabled: Uses AI segmentation for pixel-perfect person masking
    - When disabled: Falls back to geometric bounding box masks
    
    Args:
        enabled: True = segmentation on, False = geometric masks only
    
    Returns:
        { "success": bool, "segmentation_enabled": bool }
    """
    autofocus_module.blur_engine.set_segmentation_enabled(enabled)
    
    return {
        "success": True,
        "segmentation_enabled": enabled
    }

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for live video streaming"""
    await ws_handler.handle_connection(websocket, camera_index=0)


# ── Video Upload Mode routes ──────────────────────────────────────────

@app.post("/video/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file (MP4 / MOV / AVI).
    Returns session_id + metadata for the Video Upload Mode UI.
    """
    allowed_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Use: {', '.join(allowed_exts)}",
        )

    session_id = uuid.uuid4().hex[:12]
    save_dir   = _VIDEO_UPLOAD_DIR / session_id
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path  = save_dir / f"video{ext}"

    # Stream to disk without loading into RAM
    with open(save_path, "wb") as f:
        while chunk := await file.read(1024 * 256):   # 256 KB chunks
            f.write(chunk)

    # Open a video session
    session = VideoSession(str(save_path))
    if not session.open():
        shutil.rmtree(save_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail="Could not decode video file")

    _video_sessions[session_id] = session

    return {
        "session_id": session_id,
        "metadata":   session.metadata,
    }


@app.get("/video/{session_id}/metadata")
async def get_video_metadata(session_id: str):
    """Return metadata for an existing video session."""
    session = _video_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.metadata


@app.delete("/video/{session_id}")
async def delete_video_session(session_id: str):
    """Close and clean up a video session."""
    session = _video_sessions.pop(session_id, None)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session.close()
    save_dir = _VIDEO_UPLOAD_DIR / session_id
    shutil.rmtree(save_dir, ignore_errors=True)
    return {"deleted": session_id}


@app.get("/video/exports/{filename}")
async def download_export(filename: str):
    """Download a rendered export file."""
    # Search all session export dirs
    for session in _video_sessions.values():
        candidate = Path(session.file_path).parent / "exports" / filename
        if candidate.is_file():
            return FileResponse(
                str(candidate),
                media_type="video/mp4",
                filename=filename,
            )
    raise HTTPException(status_code=404, detail="Export not found")


@app.websocket("/ws/video-upload/{session_id}")
async def video_upload_ws(websocket: WebSocket, session_id: str):
    """WebSocket stream for Video Upload Mode playback and processing."""
    await video_ws_handler.handle_connection(websocket, session_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
