"""
WebSocket Handler - Manages WebSocket connections and streaming
Supports autofocus click events and cinematic composited frames.
"""
import asyncio
import base64
import json
import cv2
from fastapi import WebSocket, WebSocketDisconnect
from typing import Optional
from core.ai_engine import AIEngine
from services.camera_service import CameraService
from services.performance_service import PerformanceService

class WebSocketHandler:
    """Handles WebSocket connections for video streaming"""
    
    def __init__(self, ai_engine: AIEngine):
        self.ai_engine = ai_engine
        self.active_connections = []
        
    async def handle_connection(self, websocket: WebSocket, camera_index: int = 0):
        """Handle a WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✓ WebSocket connected (total: {len(self.active_connections)})")
        
        camera = CameraService(camera_index)
        performance = PerformanceService()
        
        if not camera.open():
            await websocket.send_json({"error": "Camera not available"})
            await websocket.close()
            return
        
        frame_count = 0
        
        try:
            while True:
                # Check for client messages
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                    await self._handle_message(websocket, raw)
                except asyncio.TimeoutError:
                    pass
                
                # Read frame
                frame = camera.read()
                if frame is None:
                    await websocket.send_json({"error": "Failed to read frame"})
                    break
                
                frame_count += 1
                
                # Performance-based frame skipping
                if performance.should_skip_frame(frame_count):
                    await asyncio.sleep(0.001)
                    continue
                
                # Process frame with AI engine
                inference_result = self.ai_engine.process_frame(frame, camera.get_fps())
                
                if inference_result:
                    # Use composited frame when autofocus module is active
                    active_module = self.ai_engine.active_module
                    composited = None
                    if hasattr(active_module, 'get_composited_frame'):
                        composited = active_module.get_composited_frame()

                    if composited is not None:
                        output_frame = composited
                    else:
                        output_frame = self._annotate_frame(frame, inference_result)

                    if frame_count % 30 == 0 and composited is not None:
                        print(f"[WS] autofocus frame  shape={composited.shape}  "
                              f"focus={getattr(active_module, '_focus_point', '?')}")

                    # Update performance metrics
                    performance.update(inference_result.fps, inference_result.inference_time)
                    
                    # Encode frame
                    _, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Prepare payload
                    payload = {
                        "frame": frame_base64,
                        "mode": inference_result.mode,
                        "fps": round(inference_result.fps, 1),
                        "detections": len(inference_result.detections),
                        "inference_time": round(inference_result.inference_time, 2),
                        "alert_level": inference_result.alert_level,
                        "objects": [
                            {
                                "class": det.get("class", det.get("class_name", "")),
                                "confidence": round(det["confidence"], 2),
                                "bbox": det["bbox"],
                                "track_id": det.get("track_id"),
                                "class_id": det.get("class_id")
                            }
                            for det in inference_result.detections
                        ],
                        "metrics": inference_result.metrics,
                        "performance": performance.get_metrics(),
                        # Autofocus extras
                        "autofocus": inference_result.metrics if inference_result.mode == "autofocus" else None,
                    }
                    
                    # Send to client
                    await websocket.send_json(payload)
                else:
                    # No AI module active - send frame only
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    await websocket.send_json({
                        "frame": frame_base64,
                        "mode": "none",
                        "fps": round(camera.get_fps(), 1),
                        "detections": 0,
                        "alert_level": "normal"
                    })
                
                # Adaptive frame skip adjustment
                if frame_count % 60 == 0:
                    performance.adjust_frame_skip()
                
                await asyncio.sleep(0.01)
                
        except WebSocketDisconnect:
            print("✓ WebSocket disconnected")
        except Exception as e:
            print(f"✗ Error in WebSocket handler: {e}")
            import traceback
            traceback.print_exc()
        finally:
            camera.release()
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            print(f"✓ Connection cleaned up (remaining: {len(self.active_connections)})")
    
    def _annotate_frame(self, frame, inference_result):
        """Add AI mode indicator and alert border"""
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Add mode badge
        mode_text = f"Mode: {inference_result.mode.upper()}"
        cv2.rectangle(annotated, (10, 10), (200, 40), (0, 0, 0), -1)
        cv2.putText(annotated, mode_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add alert border
        if inference_result.alert_level == "critical":
            color = (0, 0, 255)  # Red
            thickness = 8
        elif inference_result.alert_level == "warning":
            color = (0, 165, 255)  # Orange
            thickness = 6
        else:
            color = (0, 255, 0)  # Green
            thickness = 4
        
        cv2.rectangle(annotated, (0, 0), (w-1, h-1), color, thickness)
        
        return annotated
    
    async def _handle_message(self, websocket: WebSocket, raw: str):
        """Dispatch incoming WebSocket message strings."""
        if raw == "close":
            await websocket.close()
            return

        # Try JSON message first (autofocus / config events)
        try:
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if msg_type == "switch_mode":
                mode = msg.get("mode", "")
                success = self.ai_engine.switch_mode(mode)
                await websocket.send_json({"type": "mode_switched", "mode": mode, "success": success})

            elif msg_type == "autofocus_click":
                x, y = int(msg.get("x", 0)), int(msg.get("y", 0))
                module = self.ai_engine.active_module
                if hasattr(module, 'on_click'):
                    module.on_click(x, y)
                    await websocket.send_json({"type": "autofocus_ack", "x": x, "y": y})

            elif msg_type == "autofocus_double_click":
                module = self.ai_engine.active_module
                if hasattr(module, 'on_double_click'):
                    module.on_double_click()
                    await websocket.send_json({"type": "autofocus_reset"})

            elif msg_type == "autofocus_config":
                module = self.ai_engine.active_module
                if hasattr(module, 'set_focus_radius') and "focus_radius" in msg:
                    module.set_focus_radius(int(msg["focus_radius"]))
                if hasattr(module, 'set_blur_strength') and "blur_strength" in msg:
                    module.set_blur_strength(float(msg["blur_strength"]))
                await websocket.send_json({"type": "config_ack"})

            return
        except (json.JSONDecodeError, TypeError):
            pass

        # Legacy plain-text commands
        if raw.startswith("switch_mode:"):
            mode = raw.split(":", 1)[1]
            success = self.ai_engine.switch_mode(mode)
            await websocket.send_json({"type": "mode_switched", "mode": mode, "success": success})

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        for connection in self.active_connections[:]:
            try:
                await connection.send_json(message)
            except:
                self.active_connections.remove(connection)
