# ============================================
# PHASE 4 - RT-DETR INSTALLATION GUIDE
# Windows Compatible Installation
# ============================================

## STEP 1: Install PyTorch (Windows)

### Option A: CPU-only (Faster installation, slower inference)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Option B: GPU-accelerated (Requires NVIDIA GPU + CUDA 11.8)
```bash
# Install CUDA 11.8 from: https://developer.nvidia.com/cuda-11-8-0-download-archive
# Then install PyTorch with CUDA support:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Verify PyTorch Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## STEP 2: Install RT-DETR (Ultralytics)

```bash
cd backend
pip install -r requirements_rtdetr.txt
```

This installs:
- ultralytics>=8.1.0 (includes RT-DETR)
- opencv-python>=4.9.0
- numpy>=1.24.0
- psutil>=5.9.0

## STEP 3: Download RT-DETR Model

On first run, RT-DETR will auto-download the model (~100MB):
```bash
python -c "from ultralytics import RTDETR; model = RTDETR('rtdetr-l.pt'); print('Model downloaded')"
```

### Available Models:
- `rtdetr-l.pt` - Large (recommended, balanced speed/accuracy)
- `rtdetr-x.pt` - Extra Large (best accuracy, slower)

## STEP 4: Configure Detection

Edit `backend/detection_config.py`:

```python
# Use RT-DETR
MODEL_TYPE = "rtdetr"

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.4  # Lower = more detections
IOU_THRESHOLD = 0.6         # Higher = less overlapping boxes

# Device selection
DEVICE = "cuda"  # or "cpu"

# Person-only mode (recommended to avoid false detections)
DETECT_ONLY_PERSON = True  # Only detect persons, ignore other objects
```

## STEP 5: Verify Installation

```bash
cd backend
python test_rtdetr.py
```

Expected output:
```
✓ RT-DETR installed
✓ Model loaded: rtdetr-l.pt
✓ GPU available: True
✓ Test inference: 45.2 ms
✓ FPS: 22.1
```

## STEP 6: Start Backend

```bash
cd backend
python main.py
```

Expected output:
```
🔷 SentraVision AI Vision Platform v2.1 [RT-DETR]
➤ Detector: RTDETR
➤ Device: CUDA
➤ Person-only mode: True
✓ RT-DETR model loaded successfully
✓ Detection module active
```

## TROUBLESHOOTING

### Issue: "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### Issue: "CUDA not available"
- Install CUDA Toolkit 11.8
- Or use CPU mode: Set `DEVICE = "cpu"` in config

### Issue: Model download fails
```bash
# Manual download:
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-l.pt
move rtdetr-l.pt backend/
```

### Issue: Low FPS (<15)
- Use GPU: Set `DEVICE = "cuda"`
- Use smaller model: Change to `rtdetr-l.pt` (not rtdetr-x.pt)
- Enable person-only: Set `DETECT_ONLY_PERSON = True`

## PERFORMANCE TARGETS

| Hardware | Expected FPS | Inference Time |
|----------|--------------|----------------|
| CPU (i5) | 10-15 | 70-100ms |
| CPU (i7) | 15-20 | 50-70ms |
| GPU (RTX 3060) | 40-60 | 20-30ms |
| GPU (RTX 4090) | 100+ | 10-15ms |

## VERIFICATION CHECKLIST

- [ ] PyTorch installed
- [ ] CUDA available (if using GPU)
- [ ] ultralytics package installed
- [ ] RT-DETR model downloaded
- [ ] Backend starts without errors
- [ ] Frontend connects successfully
- [ ] Video stream shows detections
- [ ] FPS >= 15
- [ ] Track IDs stable across frames
- [ ] Tap-to-select works (if testing)
- [ ] No memory leaks after 10 minutes

## NEXT STEPS

After successful installation:
1. Test with webcam: Click "Start Vision" in frontend
2. Verify person detection accuracy
3. Check FPS meets target (>=15)
4. Test tracking stability
5. Verify blur engine receives correct masks
