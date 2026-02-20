"""
SentraVision Backend - Camera Streaming Demo (No AI for now)
Works without TFLite model - just streams camera feed
"""
import asyncio
import base64
import cv2
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from camera import Camera
from config import Config

app = FastAPI(title="SentraVision API - Demo Mode")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
# camera = None  # Remove global camera

@app.get("/")
async def root():
    return {
        "status": "SentraVision API Running - DEMO MODE (Camera Only)",
        "version": "1.0.0-demo",
        "message": "Object detection disabled - showing camera feed only"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "mode": "demo"}

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for video streaming - demo without AI"""
    camera = None  # Local variable for this connection
    
    await websocket.accept()
    print("✓ WebSocket connected")
    
    # Initialize camera
    camera = Camera(Config.CAMERA_INDEX)
    if not camera.is_opened():
        await websocket.send_json({"error": "Camera not available"})
        await websocket.close()
        return
    
    print("✓ Camera opened successfully")
    
    frame_count = 0
    skip_frames = Config.FRAME_SKIP
    
    try:
        while True:
            # Check for client messages (for graceful disconnect)
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                if message == "close":
                    break
            except asyncio.TimeoutError:
                pass
            
            # Read frame
            frame = camera.read()
            if frame is None:
                await websocket.send_json({"error": "Failed to read frame"})
                break
            
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % skip_frames != 0:
                await asyncio.sleep(0.001)
                continue
            
            # Add DEMO text overlay
            cv2.putText(frame, "DEMO MODE - Camera Feed Only", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "AI Detection: Install TensorFlow", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Calculate FPS
            fps = camera.get_fps()
            
            # Prepare payload (no detections in demo mode)
            payload = {
                "frame": frame_base64,
                "fps": round(fps, 1),
                "detections": 0,
                "inference_time": 0,
                "objects": [],
                "demo_mode": True
            }
            
            # Send to client
            await websocket.send_json(payload)
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)
            
    except WebSocketDisconnect:
        print("✓ WebSocket disconnected")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if camera:
            camera.release()
            camera = None
            print("✓ Camera released")

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🔷 SentraVision Backend - DEMO MODE")
    print("=" * 60)
    print("➤ Camera streaming enabled")
    print("➤ AI detection disabled (TensorFlow not available)")
    print("➤ Server starting on http://0.0.0.0:8000")
    print("➤ WebSocket endpoint: ws://localhost:8000/ws/video")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
