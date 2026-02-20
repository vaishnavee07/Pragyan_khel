# SentraVision AI Vision Platform v2.0

## 🔷 Enterprise-Grade Modular AI Architecture

SentraVision has been completely rebuilt from a basic camera demo into a **scalable, production-ready AI Vision Platform** with modular architecture, real-time performance monitoring, and intelligent alert management.

---

## 🏗️ Architecture Overview

```
SentraVision/
├── backend/
│   ├── core/                    # Core AI infrastructure
│   │   ├── base_module.py       # Abstract base class for all AI modules
│   │   ├── ai_engine.py         # Central AI controller & mode switcher
│   │   └── alert_engine.py      # System-wide alert management
│   │
│   ├── modules/                 # Pluggable AI modules
│   │   ├── object_detection.py  # Object detection module
│   │   └── [future modules...]  # Face recognition, motion tracking, etc.
│   │
│   ├── services/                # Support services
│   │   ├── camera_service.py    # Enhanced camera management
│   │   └── performance_service.py  # Performance monitoring & optimization
│   │
│   ├── api/                     # API layer
│   │   └── websocket_handler.py # WebSocket streaming handler
│   │
│   └── main.py                  # FastAPI application entry point
│
└── frontend_v2.html             # Enterprise-grade UI
```

---

## ✨ Key Features

### 1. **Modular AI Engine**
- **Abstract Base Class**: All AI modules inherit from `BaseAIModule`
- **Hot-Swappable Modes**: Switch between AI modes without restarting
- **Standardized Interface**: Consistent `InferenceResult` format across all modules
- **Easy Extension**: Add new AI capabilities by implementing `BaseAIModule`

### 2. **Intelligent Alert System**
- **3-Level Alerts**: NORMAL → WARNING → CRITICAL
- **Rule-Based Engine**: Customizable alert conditions
- **Visual Feedback**: Color-coded borders and status indicators
- **Alert History**: Track all alert events

### 3. **Performance Intelligence**
- **Adaptive Frame Skipping**: Automatically adjust based on system load
- **Real-Time Metrics**: FPS, inference time, CPU, memory usage
- **Performance Graphs**: Visual performance monitoring
- **Resource Optimization**: Dynamic throttling to maintain smooth operation

### 4. **Enterprise UI**
- **Modern Design**: Gradient backgrounds, backdrop blur, smooth animations
- **Real-Time Stats**: Live FPS, detections, inference time, system metrics
- **Mode Selector**: Visual AI mode switching
- **Alert Banners**: Prominent alert notifications
- **Performance Bars**: Visual system health indicators

---

## 🚀 How It Works

### AI Engine Flow

```
1. Register Modules
   └─> ai_engine.register_module('object_detection', module)

2. Activate Mode
   └─> ai_engine.switch_mode('object_detection')

3. Process Frame
   └─> result = ai_engine.process_frame(frame, fps)
       ├─> Module inference
       ├─> Alert evaluation
       └─> Performance tracking

4. Return Structured Result
   └─> InferenceResult {
         mode, fps, detections, metrics,
         alert_level, inference_time, timestamp
       }
```

### Base Module Interface

```python
class BaseAIModule(ABC):
    @abstractmethod
    def initialize(self) -> bool:
        """Load models and allocate resources"""
        pass
    
    @abstractmethod
    def process_frame(self, frame) -> InferenceResult:
        """Run inference and return structured results"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Release resources"""
        pass
```

### Adding New AI Modules

```python
from core.base_module import BaseAIModule, InferenceResult

class FaceRecognitionModule(BaseAIModule):
    def initialize(self) -> bool:
        # Load face recognition model
        self.model = load_model()
        return True
    
    def process_frame(self, frame) -> InferenceResult:
        # Run face recognition
        faces = self.model.detect(frame)
        
        return self._create_result(
            detections=[...],
            metrics={...},
            alert_level="normal",
            inference_time=inference_ms
        )
    
    def cleanup(self):
        del self.model

# Register and activate
ai_engine.register_module('face_recognition', FaceRecognitionModule())
ai_engine.switch_mode('face_recognition')
```

---

## 📊 Performance Monitoring

### Adaptive Frame Skipping
The system automatically adjusts frame processing rate based on inference performance:

- **Inference < 50ms**: Process every frame (frame_skip = 1)
- **Inference 50-100ms**: Skip 1-2 frames (frame_skip = 2-3)
- **Inference > 100ms**: Skip 3-5 frames (frame_skip = 4-5)

### Metrics Tracked
- **FPS**: Current, average, min, max
- **Inference Time**: Current, average (milliseconds)
- **System Resources**: CPU usage, memory usage
- **Alert Statistics**: Alert frequency, duration

---

## 🔔 Alert System

### Alert Levels

| Level | Trigger | Color | Border |
|-------|---------|-------|--------|
| **NORMAL** | 0-2 objects | Green | Thin (4px) |
| **WARNING** | 3-4 objects | Orange | Medium (6px) |
| **CRITICAL** | 5+ objects or dangerous items | Red | Thick (8px) |

### Alert Customization

```python
from core.alert_engine import AlertEngine, AlertLevel

alert_engine = AlertEngine()

# Add custom rule
def custom_rule(inference_result):
    if 'person' in [d['class'] for d in inference_result.detections]:
        return AlertLevel.WARNING
    return AlertLevel.NORMAL

alert_engine.add_rule('person_detector', custom_rule)
```

---

## 🎯 API Endpoints

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Platform info and status |
| `/health` | GET | Health check |
| `/modes` | GET | List available AI modes |
| `/modes/{mode}/activate` | POST | Switch AI mode |

### WebSocket API

**Connect**: `ws://localhost:8000/ws/video`

**Receive Format**:
```json
{
  "frame": "base64_encoded_jpeg",
  "mode": "object_detection",
  "fps": 28.5,
  "detections": 2,
  "inference_time": 45.3,
  "alert_level": "normal",
  "objects": [
    {"class": "person", "confidence": 0.95, "bbox": [x1, y1, x2, y2]}
  ],
  "metrics": {
    "brightness": 65,
    "motion": 0,
    "avg_inference": 42.1
  },
  "performance": {
    "fps": {"current": 28.5, "average": 27.8},
    "inference": {"current": 45.3, "average": 42.1},
    "system": {"cpu": 35, "memory": 62}
  }
}
```

**Send Commands**:
- `switch_mode:object_detection` - Switch AI mode
- `close` - Close connection

---

## 🛠️ Installation & Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Requirements**:
- fastapi==0.109.0
- uvicorn[standard]==0.27.0
- opencv-python==4.9.0.80
- numpy==1.26.3
- websockets==12.0
- python-multipart==0.0.6
- psutil==5.9.8

### 2. Run Backend

```bash
python main.py
```

Server starts on `http://localhost:8000`

### 3. Open Frontend

Open `frontend_v2.html` in your browser or navigate to `http://localhost:8000`

---

## 🎨 UI Features

### Status Bar
- **Connection Status**: Green dot when connected
- **Active Mode**: Currently running AI mode
- **Alert Level**: Current system alert level

### Video Feed
- **Live Camera Stream**: Real-time video with AI annotations
- **FPS Counter**: Top-right corner
- **Alert Border**: Color-coded based on alert level
- **Mode Badge**: Shows active AI mode

### Controls
- **Start Vision**: Connect to backend and start streaming
- **Stop Vision**: Disconnect and stop streaming

### AI Modes Panel
- **Mode Selector**: Switch between available AI modes
- **Visual Feedback**: Highlighted active mode
- **Mode Descriptions**: Info about each mode

### Detections Panel
- **Detection Count**: Total objects detected
- **Inference Time**: AI processing time
- **Object List**: Detailed list of detected objects with confidence scores

### Performance Panel
- **FPS Bar**: Frame rate visualization
- **CPU Bar**: Processor usage
- **Memory Bar**: RAM usage

---

## 🔮 Future Enhancements

### Planned AI Modules
1. **Face Recognition** - Identify and track faces
2. **Motion Tracking** - Detect movement patterns
3. **Pose Estimation** - Human pose detection
4. **Anomaly Detection** - Detect unusual behavior
5. **Crowd Analysis** - Count and track people

### Platform Features
- [ ] Multi-camera support
- [ ] Recording & playback
- [ ] Cloud storage integration
- [ ] Mobile app
- [ ] Email/SMS alerts
- [ ] Database logging
- [ ] Analytics dashboard
- [ ] User management
- [ ] ROI (Region of Interest) configuration

---

## 🏆 Advantages Over Previous Version

| Feature | Old Demo | New Platform |
|---------|----------|--------------|
| **Architecture** | Monolithic | Modular |
| **AI Modes** | Fixed | Pluggable |
| **Mode Switching** | Requires restart | Hot-swap |
| **Alerts** | None | 3-level intelligent system |
| **Performance** | Fixed frame rate | Adaptive throttling |
| **Monitoring** | Basic FPS only | Full system metrics |
| **Extensibility** | Hard to extend | Abstract base class |
| **UI** | Basic HTML | Enterprise-grade |
| **Code Organization** | 2 files | 11+ modular files |

---

## 📝 Code Quality

### Design Patterns
- **Abstract Factory**: `BaseAIModule` for creating AI modules
- **Strategy Pattern**: Switchable AI modes
- **Observer Pattern**: Alert engine monitoring
- **Service Layer**: Separated concerns (camera, performance, API)

### Best Practices
- ✅ Type hints and dataclasses
- ✅ Abstract base classes (ABC)
- ✅ Comprehensive docstrings
- ✅ Error handling and logging
- ✅ Resource cleanup (context managers)
- ✅ Async/await for concurrency
- ✅ Dependency injection

---

## 🐛 Troubleshooting

### Camera Not Opening
```python
# Check DirectShow backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
```

### Port Already in Use
```bash
# Windows
netstat -ano | findstr ":8000"
taskkill /F /PID <PID>

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### Module Import Errors
```bash
# Ensure you're in backend directory
cd backend
python main.py
```

---

## 📄 License

SentraVision AI Vision Platform  
© 2024 - All Rights Reserved

---

## 🙏 Acknowledgments

Built with:
- **FastAPI** - Modern web framework
- **OpenCV** - Computer vision
- **WebSockets** - Real-time communication
- **psutil** - System monitoring

---

## 📬 Support

For issues, questions, or contributions, please refer to the project documentation or contact the development team.

---

**🔷 SentraVision - Enterprise AI Vision, Simplified**
