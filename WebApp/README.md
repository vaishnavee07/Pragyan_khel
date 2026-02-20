# SentraVision Web Application

## Quick Start

### Method 1: Using Node.js (Recommended)

1. Install dependencies:
```bash
npm install
```

2. Start the server:
```bash
npm start
```

3. Open browser:
```
http://localhost:3000
```

### Method 2: Using Python HTTP Server

```bash
python -m http.server 8000
```

Then open: `http://localhost:8000`

### Method 3: Using Python 3

```bash
python3 -m http.server 8000
```

Then open: `http://localhost:8000`

## Features

✅ Real-time object detection using TensorFlow.js
✅ Motion detection and tracking
✅ Brightness analysis
✅ Behavior analysis (loitering, motion spikes)
✅ Color-coded alert system (Normal/Warning/Critical)
✅ FPS and performance monitoring
✅ 100% browser-based processing
✅ No server-side processing needed

## Requirements

- Modern web browser (Chrome, Firefox, Edge, Safari)
- Webcam access
- HTTPS or localhost (required for camera access)

## Camera Permissions

The browser will request camera access on first load. You must allow this for the application to work.

## Technology Stack

- TensorFlow.js 4.11.0
- COCO-SSD object detection model
- WebRTC for camera access
- Canvas API for rendering
- Express.js for local server

## Performance

- Target: 20-30 FPS
- Automatic frame skipping based on device performance
- Optimized for real-time processing
