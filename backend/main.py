"""
SentraVision AI Vision Platform
Enterprise-grade modular AI inference engine
Phase 4: RT-DETR Detection Upgrade
"""
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from core.ai_engine import AIEngine
from api.websocket_handler import WebSocketHandler
import detection_config as cfg

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
    'focus_radius': 150,
    'blur_ksize':   51,
    'feather':      40,
})
ai_engine.register_module('autofocus', autofocus_module)

# Initialize WebSocket handler
ws_handler = WebSocketHandler(ai_engine)

@app.on_event("startup")
async def startup_event():
    """Initialize platform on startup"""
    print("=" * 60)
    print("🔷 SentraVision AI Vision Platform v3.0 [Cinematic Autofocus]")
    print("=" * 60)
    print(f"➤ Detector: {cfg.MODEL_TYPE.upper()}")
    print(f"➤ Device: {cfg.DEVICE.upper()}")
    print(f"➤ Person-only mode: {cfg.DETECT_ONLY_PERSON}")
    print("➤ Modular AI Engine initialized")
    print(f"➤ Available modes: {', '.join(ai_engine.get_available_modes())}")
    print("➤ Cinematic Autofocus: ENABLED")
    
    # Activate default mode
    if ai_engine.switch_mode('object_detection'):
        print("✓ Detection module active")
    else:
        print("✗ Failed to activate detection")
    print("=" * 60)

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

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for video streaming"""
    await ws_handler.handle_connection(websocket, camera_index=0)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
