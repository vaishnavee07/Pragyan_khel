# 🔷 SentraVision AI Vision Platform v2.0

**Enterprise-Grade Modular AI Surveillance System**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-red.svg)](https://opencv.org/)
[![Status](https://img.shields.io/badge/Status-Operational-success.svg)]()

---

## 📖 Overview

SentraVision is a **production-ready AI vision platform** built with modular architecture, enabling hot-swappable AI modes, intelligent alert management, and real-time performance monitoring. Designed for scalability and extensibility.

### 🎯 Key Achievements
- ✅ **Modular Architecture**: Pluggable AI modules with abstract base class
- ✅ **Real-Time Streaming**: WebSocket-based video streaming at 30 FPS
- ✅ **Intelligent Alerts**: 3-level alert system (Normal/Warning/Critical)
- ✅ **Performance Intelligence**: Adaptive frame skipping based on system load
- ✅ **Enterprise UI**: Modern, animated interface with live metrics
- ✅ **Hot-Swappable Modes**: Switch AI modes without restarting
- ✅ **Resource Management**: Automatic cleanup and optimization

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SentraVision Platform                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐         ┌─────────────────┐            │
│  │   Frontend     │◄────────┤   WebSocket     │            │
│  │  (HTML/JS)     │         │    Handler      │            │
│  └────────────────┘         └────────┬────────┘            │
│                                       │                      │
│                            ┌──────────▼──────────┐          │
│                            │    AI Engine        │          │
│                            │  (Mode Switcher)    │          │
│                            └──┬────────────────┬─┘          │
│                               │                │             │
│              ┌────────────────▼───┐     ┌─────▼─────────┐  │
│              │ Object Detection   │     │ Future Modules │  │
│              │     Module         │     │  (Face/Motion) │  │
│              └────────────────────┘     └────────────────┘  │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Camera    │  │  Performance │  │  Alert Engine    │  │
│  │   Service   │  │   Service    │  │  (3 Levels)      │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 📁 Project Structure

```
SentraVision/
├── backend/
│   ├── core/                    # Core AI infrastructure
│   │   ├── base_module.py       # Abstract base class
│   │   ├── ai_engine.py         # AI controller
│   │   └── alert_engine.py      # Alert management
│   │
│   ├── modules/                 # AI modules
│   │   └── object_detection.py  # Object detection
│   │
│   ├── services/                # Support services
│   │   ├── camera_service.py    # Camera management
│   │   └── performance_service.py  # Performance monitoring
│   │
│   ├── api/                     # API layer
│   │   └── websocket_handler.py # WebSocket handler
│   │
│   ├── main.py                  # FastAPI app
│   └── requirements.txt         # Dependencies
│
├── frontend_v2.html             # Enterprise UI
├── ARCHITECTURE.md              # Detailed documentation
├── QUICKSTART.md                # Quick start guide
└── README.md                    # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# 1. Clone or navigate to project
cd SentraVision

# 2. Install dependencies
cd backend
pip install -r requirements.txt

# 3. Run backend
python main.py
```

### Usage

1. **Start Backend**: Run `python main.py` in `/backend`
2. **Open Frontend**: Open `frontend_v2.html` in browser
3. **Start Vision**: Click "Start Vision" button
4. **Watch Live Feed**: Camera streams with AI annotations

---

## ✨ Features

### 🎯 Modular AI Engine
- **Abstract Base Class**: All modules inherit from `BaseAIModule`
- **Standardized Interface**: Consistent `InferenceResult` format
- **Hot-Swap Modes**: Switch AI modes without restart
- **Easy Extension**: Add new AI capabilities in minutes

### 🔔 Intelligent Alert System
- **3-Level Alerts**: NORMAL → WARNING → CRITICAL
- **Visual Feedback**: Color-coded borders and banners
- **Customizable Rules**: Add custom alert conditions
- **Alert History**: Track all events

### 📊 Performance Intelligence
- **Adaptive Throttling**: Auto-adjust frame rate based on load
- **Real-Time Metrics**: FPS, inference time, CPU, memory
- **Performance Graphs**: Visual monitoring
- **Resource Optimization**: Dynamic frame skipping

### 🎨 Enterprise UI
- **Modern Design**: Gradient backgrounds, animations
- **Live Stats Dashboard**: Real-time metrics
- **Mode Selector**: Visual AI mode switching
- **Performance Bars**: System health visualization

---

## 🔧 API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Platform status |
| `/health` | GET | Health check |
| `/modes` | GET | Available AI modes |
| `/modes/{mode}/activate` | POST | Switch AI mode |

### WebSocket Endpoint

**Connect**: `ws://localhost:8000/ws/video`

**Receive Data**:
```json
{
  "frame": "base64_jpeg",
  "mode": "object_detection",
  "fps": 28.5,
  "detections": 2,
  "inference_time": 45.3,
  "alert_level": "normal",
  "objects": [...],
  "metrics": {...},
  "performance": {...}
}
```

**Send Commands**:
- `switch_mode:object_detection`
- `close`

---

## 🧩 Extending the Platform

### Add New AI Module

```python
# 1. Create module file: backend/modules/face_recognition.py
from core.base_module import BaseAIModule, InferenceResult

class FaceRecognitionModule(BaseAIModule):
    def initialize(self) -> bool:
        # Load model
        return True
    
    def process_frame(self, frame) -> InferenceResult:
        # Run inference
        faces = detect_faces(frame)
        
        return self._create_result(
            detections=faces,
            metrics={},
            alert_level="normal",
            inference_time=50
        )
    
    def cleanup(self):
        pass

# 2. Register in main.py
face_recognition = FaceRecognitionModule()
ai_engine.register_module('face_recognition', face_recognition)
```

### Add Custom Alert Rule

```python
from core.alert_engine import AlertEngine, AlertLevel

alert_engine = AlertEngine()

def custom_rule(inference_result):
    if len(inference_result.detections) > 3:
        return AlertLevel.WARNING
    return AlertLevel.NORMAL

alert_engine.add_rule('my_rule', custom_rule)
```

---

## 📊 Performance

### Benchmarks (Intel i5, 8GB RAM)
- **FPS**: 25-30 (1080p camera)
- **Inference**: 40-60ms (object detection)
- **CPU**: 30-40% average
- **Memory**: ~500MB

### Optimizations
- **DirectShow Backend**: Reliable Windows camera access
- **Adaptive Frame Skip**: Maintains performance under load
- **JPEG Compression**: 85% quality, optimized bandwidth
- **Buffer Management**: Size 1 for minimal latency

---

## 🔮 Roadmap

### Phase 1: Core Platform ✅
- [x] Modular architecture
- [x] AI engine controller
- [x] Alert system
- [x] Performance monitoring
- [x] Enterprise UI

### Phase 2: AI Capabilities
- [ ] TensorFlow Lite integration
- [ ] Face recognition module
- [ ] Motion tracking module
- [ ] Pose estimation
- [ ] Anomaly detection

### Phase 3: Production Features
- [ ] Multi-camera support
- [ ] Recording & playback
- [ ] Cloud storage
- [ ] Mobile app
- [ ] Email/SMS alerts
- [ ] Database logging
- [ ] Analytics dashboard
- [ ] User management

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python 3.9+, FastAPI |
| **Computer Vision** | OpenCV, DirectShow |
| **WebSocket** | FastAPI WebSocket |
| **Async** | asyncio, uvicorn |
| **Monitoring** | psutil |
| **Frontend** | HTML5, JavaScript, CSS3 |
| **UI** | Gradient design, animations |

---

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)**: Get started in 3 steps
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Detailed architecture guide
- **Code Comments**: Comprehensive inline documentation

---

## 🐛 Troubleshooting

### Camera Issues
```bash
# Test camera
python backend/test_camera.py

# Use DirectShow backend (Windows)
cv2.VideoCapture(0, cv2.CAP_DSHOW)
```

### Port Conflicts
```bash
# Find process using port 8000
netstat -ano | findstr ":8000"

# Kill process
taskkill /F /PID <PID>
```

### Import Errors
```bash
# Ensure backend directory as working dir
cd backend
python main.py
```

---

## 🏆 Comparison

| Feature | v1.0 Demo | v2.0 Platform |
|---------|-----------|---------------|
| Architecture | Monolithic | Modular |
| AI Modes | Fixed | Pluggable |
| Alerts | None | 3-level system |
| Performance | Fixed rate | Adaptive |
| Monitoring | FPS only | Full metrics |
| UI | Basic | Enterprise |
| Extensibility | Hard | Easy |
| Code Structure | 2 files | 11+ modules |

---

## 📄 License

© 2024 SentraVision AI Vision Platform. All Rights Reserved.

---

## 🙏 Credits

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [OpenCV](https://opencv.org/) - Computer vision
- [psutil](https://github.com/giampaolo/psutil) - System monitoring

---

## 📬 Contact

For questions, issues, or contributions:
- **Documentation**: See `ARCHITECTURE.md`
- **Quick Start**: See `QUICKSTART.md`
- **Issues**: Check troubleshooting section

---

<div align="center">

**🔷 SentraVision - Enterprise AI Vision, Simplified**

*Transform any camera into an intelligent vision system*

[Quick Start](QUICKSTART.md) • [Architecture](ARCHITECTURE.md) • [API Docs](#-api-reference)

</div>
