let video, canvas, ctx;
let model = null;
let previousFrameData = null;
let isProcessing = false;

// Frame history for behavior analysis
const frameHistory = [];
const MAX_HISTORY = 100;
let loiteringStartTime = 0;
let motionBaseline = 20;

// Performance tracking
let lastFrameTime = Date.now();
let fps = 0;
let frameCount = 0;
let frameSkip = 0;
let frameSkipCounter = 0;

// Alert states
const AlertState = {
    NORMAL: { name: 'NORMAL', color: '#4CAF50', class: 'status-normal' },
    WARNING: { name: 'WARNING', color: '#FFC107', class: 'status-warning' },
    CRITICAL: { name: 'CRITICAL', color: '#F44336', class: 'status-critical' }
};

let currentAlert = AlertState.NORMAL;

async function init() {
    try {
        video = document.getElementById('video');
        canvas = document.getElementById('canvas');
        ctx = canvas.getContext('2d');

        // Request camera access
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        });

        video.srcObject = stream;
        
        await new Promise(resolve => {
            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                resolve();
            };
        });

        // Load COCO-SSD model
        updateLoadingMessage('Loading object detection model...');
        model = await cocoSsd.load();

        document.getElementById('loading').style.display = 'none';
        
        // Start processing
        requestAnimationFrame(processFrame);
        
    } catch (error) {
        showError('Camera access denied or model loading failed: ' + error.message);
    }
}

function updateLoadingMessage(message) {
    const loadingDiv = document.getElementById('loading');
    if (loadingDiv) {
        loadingDiv.querySelector('p').textContent = message;
    }
}

function showError(message) {
    document.getElementById('loading').style.display = 'none';
    const errorDiv = document.getElementById('error');
    errorDiv.style.display = 'block';
    document.getElementById('errorMessage').textContent = message;
}

async function processFrame() {
    if (!model || !video.readyState === 4) {
        requestAnimationFrame(processFrame);
        return;
    }

    // Frame skipping for performance
    frameSkipCounter++;
    if (frameSkipCounter <= frameSkip) {
        requestAnimationFrame(processFrame);
        return;
    }
    frameSkipCounter = 0;

    if (isProcessing) {
        requestAnimationFrame(processFrame);
        return;
    }

    isProcessing = true;
    const startTime = performance.now();

    try {
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Run object detection
        const predictions = await model.detect(video);

        // Compute motion and brightness
        const motionIntensity = computeMotion();
        const brightness = computeBrightness();

        // Analyze behavior
        const alertState = analyzeBehavior(predictions, motionIntensity);

        // Draw results
        drawPredictions(predictions);
        drawBorder(alertState);

        // Update UI
        updateUI(predictions, motionIntensity, brightness, alertState, performance.now() - startTime);

        // Adjust frame skip based on performance
        const inferenceTime = performance.now() - startTime;
        if (inferenceTime > 50) {
            frameSkip = Math.min(frameSkip + 1, 3);
        } else if (inferenceTime < 25) {
            frameSkip = Math.max(frameSkip - 1, 0);
        }

    } catch (error) {
        console.error('Processing error:', error);
    }

    isProcessing = false;
    requestAnimationFrame(processFrame);
}

function computeMotion() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 160;
    tempCanvas.height = 120;
    const tempCtx = tempCanvas.getContext('2d');
    
    tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
    const currentData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    
    if (!previousFrameData) {
        previousFrameData = currentData;
        return 0;
    }

    let diffSum = 0;
    let motionPixels = 0;
    const threshold = 30;

    for (let i = 0; i < currentData.data.length; i += 4) {
        const diff = Math.abs(
            (currentData.data[i] + currentData.data[i + 1] + currentData.data[i + 2]) / 3 -
            (previousFrameData.data[i] + previousFrameData.data[i + 1] + previousFrameData.data[i + 2]) / 3
        );
        
        if (diff > threshold) {
            motionPixels++;
        }
        diffSum += diff;
    }

    previousFrameData = currentData;

    const motionScore = (motionPixels / (tempCanvas.width * tempCanvas.height)) * 100;
    return Math.min(motionScore, 100);
}

function computeBrightness() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 100;
    tempCanvas.height = 100;
    const tempCtx = tempCanvas.getContext('2d');
    
    tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
    const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    
    let sum = 0;
    for (let i = 0; i < imageData.data.length; i += 4) {
        sum += (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
    }
    
    const avg = sum / (tempCanvas.width * tempCanvas.height);
    return Math.round((avg / 255) * 100);
}

function analyzeBehavior(predictions, motionIntensity) {
    const timestamp = Date.now();
    const personCount = predictions.filter(p => p.class === 'person').length;

    // Store frame data
    frameHistory.push({
        timestamp,
        personCount,
        motionIntensity,
        detectionCount: predictions.length
    });

    if (frameHistory.length > MAX_HISTORY) {
        frameHistory.shift();
    }

    // Detect loitering
    const loitering = detectLoitering(personCount, timestamp);

    // Detect motion spike
    const motionSpike = detectMotionSpike(motionIntensity);

    // Determine alert state
    if (motionSpike || motionIntensity > 70) {
        return AlertState.CRITICAL;
    } else if (loitering) {
        return AlertState.WARNING;
    } else {
        return AlertState.NORMAL;
    }
}

function detectLoitering(personCount, timestamp) {
    if (personCount > 0) {
        if (loiteringStartTime === 0) {
            loiteringStartTime = timestamp;
        }
        const duration = timestamp - loiteringStartTime;
        return duration > 15000; // 15 seconds
    } else {
        loiteringStartTime = 0;
        return false;
    }
}

function detectMotionSpike(motionIntensity) {
    if (frameHistory.length < 30) return false;

    const threshold = motionBaseline * 2.5;
    const recentFrames = frameHistory.slice(-5);
    const sustainedSpike = recentFrames.filter(f => f.motionIntensity > threshold).length >= 3;

    if (!sustainedSpike) {
        motionBaseline = motionBaseline * 0.95 + motionIntensity * 0.05;
    }

    return sustainedSpike;
}

function drawPredictions(predictions) {
    predictions.forEach(prediction => {
        const [x, y, width, height] = prediction.bbox;

        // Color based on class
        let color = '#2196F3'; // blue for person
        if (prediction.class === 'handbag' || prediction.class === 'backpack' || prediction.class === 'suitcase') {
            color = '#9C27B0'; // purple for bags
        } else if (prediction.class !== 'person') {
            color = '#607D8B'; // gray for other objects
        }

        // Draw bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);

        // Draw filled background for label
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.3;
        ctx.fillRect(x, y, width, height);
        ctx.globalAlpha = 1.0;

        // Draw label
        const label = `${prediction.class} ${Math.round(prediction.score * 100)}%`;
        ctx.font = '16px Arial';
        ctx.fillStyle = color;
        ctx.fillRect(x, y - 25, ctx.measureText(label).width + 10, 25);
        ctx.fillStyle = 'white';
        ctx.fillText(label, x + 5, y - 7);
    });
}

function drawBorder(alertState) {
    ctx.strokeStyle = alertState.color;
    ctx.lineWidth = 12;
    ctx.strokeRect(6, 6, canvas.width - 12, canvas.height - 12);
}

function updateUI(predictions, motionIntensity, brightness, alertState, inferenceTime) {
    // Update status
    const statusDiv = document.getElementById('status');
    statusDiv.textContent = alertState.name;
    statusDiv.className = alertState.class;

    // Update metrics
    document.getElementById('brightness').textContent = brightness + '%';
    document.getElementById('motion').textContent = Math.round(motionIntensity) + '%';
    document.getElementById('detections').textContent = predictions.length;
    document.getElementById('inference').textContent = Math.round(inferenceTime) + 'ms';

    // Calculate FPS
    frameCount++;
    const now = Date.now();
    if (now - lastFrameTime >= 1000) {
        fps = frameCount;
        frameCount = 0;
        lastFrameTime = now;
    }
    document.getElementById('fps').textContent = fps;

    // Update current alert
    currentAlert = alertState;
}

// Start on page load
window.addEventListener('load', init);
