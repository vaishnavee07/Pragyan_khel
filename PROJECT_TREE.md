# 🔷 SentraVision - Complete Project Tree

## Directory Structure

```
SentraVision/
│
├── backend/                           # Python FastAPI Backend
│   ├── main.py                       # FastAPI server + WebSocket endpoint
│   ├── camera.py                     # Camera capture with FPS tracking
│   ├── detector.py                   # TensorFlow Lite object detection
│   ├── config.py                     # Configuration settings
│   ├── requirements.txt              # Python dependencies
│   ├── .gitignore                    # Python gitignore
│   └── models/
│       ├── .gitkeep                  # Placeholder for models folder
│       └── detect.tflite            # TFLite model (download separately)
│
├── frontend/                          # React + Next.js Frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx             # Main dashboard (camera + stats)
│   │   │   ├── layout.tsx           # Root layout with metadata
│   │   │   └── globals.css          # Global styles + Tailwind
│   │   └── components/
│   │       ├── VideoFeed.tsx        # WebSocket video stream component
│   │       └── Stats.tsx            # Performance metrics display
│   ├── package.json                  # npm dependencies
│   ├── next.config.js                # Next.js configuration
│   ├── tailwind.config.js            # Tailwind CSS config
│   ├── postcss.config.js             # PostCSS config
│   ├── tsconfig.json                 # TypeScript configuration
│   ├── .eslintrc.json                # ESLint rules
│   └── .gitignore                    # Node gitignore
│
├── SETUP.md                           # Complete installation guide
├── START_SENTRAVISION.bat            # Windows startup script
├── start_sentravision.sh             # Linux/macOS startup script
│
└── [Legacy files - can be archived]
    ├── AndroidProject/
    ├── WebApp/
    └── [Various .md files]
```

## File Count Summary

**Backend:** 6 files  
**Frontend:** 11 files  
**Documentation:** 3 files  
**Total New Files:** 20

## Core Files Description

### Backend

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | 135 | FastAPI server, WebSocket streaming, frame processing |
| `camera.py` | 65 | OpenCV camera capture, FPS tracking, resource management |
| `detector.py` | 145 | TFLite model loading, object detection, bounding boxes |
| `config.py` | 18 | Configuration constants (camera, model, performance) |
| `requirements.txt` | 7 | Python package dependencies |

### Frontend

| File | Lines | Purpose |
|------|-------|---------|
| `page.tsx` | 75 | Main dashboard with video feed and stats panel |
| `layout.tsx` | 22 | Root layout with Inter font and metadata |
| `globals.css` | 50 | Tailwind directives, custom scrollbar, dark theme |
| `VideoFeed.tsx` | 120 | WebSocket client, canvas rendering, start/stop controls |
| `Stats.tsx` | 95 | Performance metrics, detected objects list, system info |
| `package.json` | 27 | Dependencies and scripts |
| `tailwind.config.js` | 17 | Tailwind theme customization |
| `next.config.js` | 6 | Next.js optimization settings |
| `tsconfig.json` | 26 | TypeScript compiler options |

## Technology Stack

### Backend

```python
fastapi==0.109.0           # Modern async web framework
uvicorn[standard]==0.27.0  # ASGI server
opencv-python==4.9.0.80    # Camera capture
numpy==1.26.3              # Array operations
tensorflow==2.15.0         # TFLite inference
websockets==12.0           # WebSocket protocol
```

### Frontend

```json
next: ^14.1.0              // React framework
react: ^18.2.0             // UI library
typescript: ^5.3.3         // Type safety
tailwindcss: ^3.4.1        // Styling
framer-motion: ^11.0.3     // Animations
```

## WebSocket Protocol

**Endpoint:** `ws://localhost:8000/ws/video`

**Client → Server:**
```json
"close"  // Graceful disconnect
```

**Server → Client:**
```json
{
  "frame": "base64_encoded_jpeg",
  "fps": 25.3,
  "detections": 2,
  "inference_time": 42.5,
  "objects": [
    {
      "class": "person",
      "confidence": 0.87,
      "bbox": [120, 50, 300, 400]
    }
  ]
}
```

## Performance Characteristics

- **Video Resolution:** 640x480
- **Target FPS:** 20-30
- **Frame Skip:** Every 2nd frame processed
- **Inference Time:** 30-50ms per frame
- **WebSocket Latency:** <20ms
- **Memory Usage:** ~200MB (backend) + ~150MB (frontend)
- **Model Size:** ~4MB (MobileNet SSD)

## Installation Flow

1. **Download model** → Place in `backend/models/detect.tflite`
2. **Backend setup** → `pip install -r requirements.txt`
3. **Frontend setup** → `npm install`
4. **Start backend** → `python main.py` (port 8000)
5. **Start frontend** → `npm run dev` (port 3000)
6. **Access UI** → `http://localhost:3000`

## Quick Start Commands

```bash
# Windows
START_SENTRAVISION.bat

# Linux/macOS
chmod +x start_sentravision.sh
./start_sentravision.sh
```

## Model Download

**MobileNet SSD v1 (Quantized):**
```
https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
```

Extract `detect.tflite` → Place in `backend/models/`

## Architecture Diagram

```
┌─────────────┐         WebSocket          ┌──────────────┐
│   Browser   │ ←────────────────────────→ │   FastAPI    │
│  (React UI) │    Base64 JPEG Frames      │   (Python)   │
└─────────────┘                             └──────────────┘
                                                    │
                                                    ▼
                                            ┌──────────────┐
                                            │   OpenCV     │
                                            │   Camera     │
                                            └──────────────┘
                                                    │
                                                    ▼
                                            ┌──────────────┐
                                            │ TensorFlow   │
                                            │ Lite Model   │
                                            └──────────────┘
```

## Modular Architecture

**Clean separation of concerns:**
- `camera.py` → Camera hardware interaction
- `detector.py` → AI inference logic
- `main.py` → API endpoints and streaming
- `config.py` → Centralized settings

**Frontend components:**
- `VideoFeed.tsx` → WebSocket client + canvas
- `Stats.tsx` → UI metrics display
- `page.tsx` → Layout orchestration

## Extension Points

**Backend:**
- Add motion detection in `camera.py`
- Integrate new models in `detector.py`
- Add REST endpoints in `main.py`

**Frontend:**
- Add new metric cards in `Stats.tsx`
- Implement recording in `VideoFeed.tsx`
- Create settings page in `app/settings/`

## Clean Foundation ✓

✅ No bloat, no unnecessary features  
✅ Modular and extensible  
✅ Production-ready error handling  
✅ Graceful resource cleanup  
✅ Type-safe frontend  
✅ Async backend  
✅ Real-time streaming  
✅ Modern UI with animations  

**Ready for expansion.**
