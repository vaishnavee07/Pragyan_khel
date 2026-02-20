# 🔷 SentraVision - Real-time AI Surveillance System

A clean, scalable Python + React foundation for real-time object detection with camera feed streaming.

## 📂 Project Structure

```
SentraVision/
├── backend/                   # Python FastAPI server
│   ├── main.py               # FastAPI + WebSocket server
│   ├── camera.py             # Camera capture logic
│   ├── detector.py           # TensorFlow object detection
│   ├── config.py             # Configuration settings
│   ├── requirements.txt      # Python dependencies
│   └── models/
│       └── detect.tflite     # Download model file
│
├── frontend/                  # React + Next.js UI
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx      # Main dashboard
│   │   │   ├── layout.tsx    # Root layout
│   │   │   └── globals.css   # Global styles
│   │   └── components/
│   │       ├── VideoFeed.tsx # Video stream display
│   │       └── Stats.tsx     # Performance metrics
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   └── tsconfig.json
│
└── README.md
```

## 🚀 Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- Webcam connected

### Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download TensorFlow Lite model
# Download from: https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
# Extract and place detect.tflite in backend/models/
```

### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install
```

## ▶️ Running the Application

### Start Backend (Terminal 1)

```bash
cd backend
venv\Scripts\activate  # Windows
python main.py
```

Backend runs on: `http://localhost:8000`

### Start Frontend (Terminal 2)

```bash
cd frontend
npm run dev
```

Frontend runs on: `http://localhost:3000`

### Access Application

Open browser: `http://localhost:3000`

Click **"Start Camera"** to begin real-time detection.

## 🎯 Features

✅ Real-time webcam capture  
✅ TensorFlow Lite object detection  
✅ WebSocket streaming (20-30 FPS)  
✅ Bounding box visualization  
✅ Live FPS counter  
✅ Detection confidence display  
✅ Modern dark UI with animations  
✅ Graceful camera release  
✅ Non-blocking inference  

## 🔧 Configuration

Edit `backend/config.py` to customize:

```python
CAMERA_INDEX = 0               # Change camera source
CONFIDENCE_THRESHOLD = 0.5     # Detection confidence (0-1)
FRAME_SKIP = 2                 # Process every Nth frame
```

## 🛠️ Tech Stack

**Backend:**
- FastAPI - Modern async Python web framework
- OpenCV - Camera capture and image processing
- TensorFlow Lite - Lightweight object detection
- WebSocket - Real-time bidirectional communication

**Frontend:**
- Next.js 14 - React framework with App Router
- TypeScript - Type-safe JavaScript
- Tailwind CSS - Utility-first styling
- Framer Motion - Smooth animations

## 📊 Performance

- **Target FPS:** 20-30 FPS
- **Inference Time:** 30-50ms per frame
- **Model:** MobileNet SSD (lightweight)
- **Input Size:** 640x480
- **Memory:** Efficient frame processing

## 🔒 Security Notes

- Backend accepts connections only from `localhost:3000`
- WebSocket connections are isolated per client
- Camera access requires user permission
- No data storage or cloud transmission

## 🐛 Troubleshooting

**Camera not opening:**
- Check camera permissions
- Verify camera index in `config.py`
- Ensure no other app is using camera

**WebSocket connection failed:**
- Verify backend is running on port 8000
- Check firewall settings
- Ensure no port conflicts

**Model not found:**
- Download `detect.tflite` model
- Place in `backend/models/` folder
- Verify path in `config.py`

**Low FPS:**
- Increase `FRAME_SKIP` in config
- Close resource-intensive applications
- Use GPU acceleration if available

## 📝 Next Steps

This is a clean foundation. You can extend with:
- Multi-camera support
- Motion detection
- Alert system
- Database integration
- Cloud deployment
- Mobile app

## 📄 License

MIT License - Build and extend as needed.
