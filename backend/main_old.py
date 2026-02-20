"""
SentraVision Backend - FastAPI + WebSocket Server
Real-time object detection streaming
"""
import asyncio
import base64
import cv2
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from camera import Camera
from detector import ObjectDetector
from config import Config

app = FastAPI(title="SentraVision API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
camera = None
detector = None

@app.on_event("startup")
async def startup_event():
    """Initialize detector on startup"""
    global detector
    detector = ObjectDetector(Config.MODEL_PATH, Config.CONFIDENCE_THRESHOLD)
    print(f"✓ Object detector loaded")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global camera
    if camera:
        camera.release()
    print("✓ Camera released")

@app.get("/")
async def root():
    return {"status": "SentraVision API Running", "version": "1.0.0"}

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for video streaming"""
    global camera, detector
    
    await websocket.accept()
    print("✓ WebSocket connected")
    
    # Initialize camera
    camera = Camera(Config.CAMERA_INDEX)
    if not camera.is_opened():
        await websocket.send_json({"error": "Camera not available"})
        await websocket.close()
        return
    
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
            
            # Detect objects
            detections, inference_time = detector.detect(frame)
            
            # Draw bounding boxes
            annotated_frame = detector.draw_detections(frame, detections)
            
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Calculate FPS
            fps = camera.get_fps()
            
            # Prepare payload
            payload = {
                "frame": frame_base64,
                "fps": round(fps, 1),
                "detections": len(detections),
                "inference_time": round(inference_time, 2),
                "objects": [
                    {
                        "class": det["class"],
                        "confidence": round(det["confidence"], 2),
                        "bbox": det["bbox"]
                    }
                    for det in detections
                ]
            }
            
            # Send to client
            await websocket.send_json(payload)
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)
            
    except WebSocketDisconnect:
        print("✓ WebSocket disconnected")
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        if camera:
            camera.release()
            camera = None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
